import numpy as np
from tqdm import tqdm

from src.domain.analysis.complexity.shared import (
    aggregate_min_mean_max,
    build_approx_mst,
    make_null_row,
)
from src.core.utils import timed


_N_KEYS = ("n1", "n2", "n3", "n4")


def _n1_vec(c_mask: np.ndarray, j_mask: np.ndarray, edges_uv: np.ndarray) -> float:
    """N1: fraction of cluster-c samples sharing an MST edge with the j population."""
    n_c = int(c_mask.sum())
    if n_c == 0 or edges_uv.shape[0] == 0:
        return 0.0
    u, v = edges_uv[:, 0], edges_uv[:, 1]
    boundary_u = u[c_mask[u] & j_mask[v]]
    boundary_v = v[c_mask[v] & j_mask[u]]
    if boundary_u.size == 0 and boundary_v.size == 0:
        return 0.0
    return float(np.unique(np.concatenate([boundary_u, boundary_v])).size / n_c)


def _n2_vec(
    nbs: np.ndarray,
    nb_dists: np.ndarray,
    in_c: np.ndarray,
    in_j: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """N2: mean intra/(intra+inter) NN distance ratio over cluster-c samples."""
    valid = in_c.any(axis=1) & in_j.any(axis=1)
    if not valid.any():
        # No sample has both an intra- and an inter-class neighbour: cluster c is
        # cleanly separated from j, the easy end of the ratio. Returning 0.0 (not
        # the 0.5 midpoint) keeps the degenerate case consistent with N3/N4, which
        # already return 0.0 for the same "no c∪j neighbour" condition.
        return 0.0
    rows = np.arange(nbs.shape[0])
    intra_d = nb_dists[rows, in_c.argmax(axis=1)]
    inter_d = nb_dists[rows, in_j.argmax(axis=1)]
    ratios = intra_d[valid] / (intra_d[valid] + inter_d[valid] + eps)
    return float(ratios.mean())


def _n3_vec(
    nbs: np.ndarray, j_mask: np.ndarray, in_c: np.ndarray, in_j: np.ndarray
) -> float:
    """N3: 1-NN error rate restricted to c ∪ j neighbours."""
    in_cj = in_c | in_j
    valid = in_cj.any(axis=1)
    if not valid.any():
        return 0.0
    rows = np.arange(nbs.shape[0])
    nb_idx = nbs[rows, in_cj.argmax(axis=1)]
    misclassified = j_mask[nb_idx] & valid
    return float(misclassified.sum() / valid.sum())


def _n4_vec(in_c: np.ndarray, in_j: np.ndarray) -> float:
    """N4: k-NN majority-vote error rate restricted to c ∪ j neighbours."""
    c_votes = in_c.sum(axis=1)
    j_votes = in_j.sum(axis=1)
    valid = (c_votes + j_votes) > 0
    if not valid.any():
        return 0.0
    return float(((j_votes > c_votes) & valid).sum() / valid.sum())


def _pair_metrics(
    nbs: np.ndarray,
    nb_dists: np.ndarray,
    c_mask: np.ndarray,
    j_mask: np.ndarray,
    edges_uv: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute (n1, n2, n3, n4) for cluster c against population j."""
    in_c = c_mask[nbs]
    in_j = j_mask[nbs]
    n1 = _n1_vec(c_mask, j_mask, edges_uv)
    n2 = _n2_vec(nbs, nb_dists, in_c, in_j)
    n3 = _n3_vec(nbs, j_mask, in_c, in_j)
    n4 = _n4_vec(in_c, in_j)
    return n1, n2, n3, n4


def _aggregate_pairs(
    nbs: np.ndarray,
    nb_dists: np.ndarray,
    c_mask: np.ndarray,
    population_masks: list[np.ndarray],
    edges_uv: np.ndarray,
) -> dict[str, list[float]]:
    """Run _pair_metrics over each population mask, skipping empty ones."""
    out: dict[str, list[float]] = {k: [] for k in _N_KEYS}
    for j_mask in population_masks:
        if not j_mask.any():
            continue
        n1, n2, n3, n4 = _pair_metrics(nbs, nb_dists, c_mask, j_mask, edges_uv)
        out["n1"].append(n1)
        out["n2"].append(n2)
        out["n3"].append(n3)
        out["n4"].append(n4)
    return out


@timed
def compute_n_measures(
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    cluster_mask: dict[str, np.ndarray],
    top_k_map: dict[str, list[str]],
    *,
    metric: str = "cosine",
) -> dict[str, dict[str, float | None]]:
    """N1-N4 per cluster vs the top-K adversarial clusters, as min/mean/max.

    Builds a global approximate MST once (for N1), then derives N2-N4 from the
    k-NN graph via vectorised boolean masks. `metric` is forwarded to the MST so
    it stays consistent with the k-NN graph.
    """
    edges_uv = build_approx_mst(knn_idx, knn_dist, X_num, X_cat, metric=metric)

    result: dict[str, dict[str, float | None]] = {}
    for cid_str, c_mask in tqdm(
        cluster_mask.items(), desc="N measures", unit="cluster", leave=False
    ):
        row = make_null_row(_N_KEYS)
        c_full_idx = np.where(c_mask)[0]
        if c_full_idx.size == 0:
            result[cid_str] = row
            continue

        nbs = knn_idx[c_full_idx]
        nb_dists = knn_dist[c_full_idx]

        cluster_pops = [
            cluster_mask[ac] for ac in top_k_map.get(cid_str, []) if ac in cluster_mask
        ]

        agg = _aggregate_pairs(nbs, nb_dists, c_mask, cluster_pops, edges_uv)
        for nk in _N_KEYS:
            mn, me, mx = aggregate_min_mean_max(agg[nk])
            row[f"{nk}_min"] = mn
            row[f"{nk}_mean"] = me
            row[f"{nk}_max"] = mx

        result[cid_str] = row

    return result
