import numpy as np
from tqdm import tqdm

from src.domain.analysis.complexity.shared import aggregate_min_mean_max, make_null_row, _l2_normalize
from src.core.utils import timed


def _f1_pair(X_c: np.ndarray, X_j: np.ndarray, eps: float = 1e-8) -> float:
    """F1: 1/(1 + max_feature Fisher ratio). Higher = harder (more overlap)."""
    mu_c = X_c.mean(axis=0)
    mu_j = X_j.mean(axis=0)
    var_c = X_c.var(axis=0)
    var_j = X_j.var(axis=0)
    fisher = (mu_c - mu_j) ** 2 / (var_c + var_j + eps)
    return float(1.0 / (1.0 + np.max(fisher)))


def _f2_pair(X_c: np.ndarray, X_j: np.ndarray, eps: float = 1e-8) -> float:
    """F2: mean over features of per-feature bounding-box overlap ratio. Higher = harder."""
    min_c, max_c = X_c.min(axis=0), X_c.max(axis=0)
    min_j, max_j = X_j.min(axis=0), X_j.max(axis=0)
    overlap = np.maximum(0.0, np.minimum(max_c, max_j) - np.maximum(min_c, min_j))
    total_range = np.maximum(max_c, max_j) - np.minimum(min_c, min_j) + eps
    return float(np.mean(overlap / total_range))


def _f3_pair(X_c: np.ndarray, X_j: np.ndarray) -> float:
    """F3: min over features of the fraction of cluster-c samples in the overlap region.

    Measures how well a single feature can separate cluster c from cluster j.
    Higher = harder (no single feature separates well).
    """
    min_c, max_c = X_c.min(axis=0), X_c.max(axis=0)
    min_j, max_j = X_j.min(axis=0), X_j.max(axis=0)
    lo = np.maximum(min_c, min_j)
    hi = np.minimum(max_c, max_j)
    fracs = []
    for f in range(X_c.shape[1]):
        if hi[f] >= lo[f]:
            frac = float(np.mean((X_c[:, f] >= lo[f]) & (X_c[:, f] <= hi[f])))
        else:
            frac = 0.0
        fracs.append(frac)
    return float(np.min(fracs))


def _f4_pair(X_c: np.ndarray, X_j: np.ndarray) -> float:
    """F4: fraction of cluster-c samples in the overlap region on ALL features simultaneously.

    Higher = harder (many samples overlap jointly in all features).
    """
    min_c, max_c = X_c.min(axis=0), X_c.max(axis=0)
    min_j, max_j = X_j.min(axis=0), X_j.max(axis=0)
    lo = np.maximum(min_c, min_j)
    hi = np.minimum(max_c, max_j)
    if np.any(hi < lo):
        return 0.0
    in_all = np.ones(len(X_c), dtype=bool)
    for f in range(X_c.shape[1]):
        in_all &= (X_c[:, f] >= lo[f]) & (X_c[:, f] <= hi[f])
    return float(np.mean(in_all))


_F_KEYS = ("f1", "f2", "f3", "f4")


def _pair_block(X_c: np.ndarray, X_others: list[np.ndarray]) -> dict[str, list[float]]:
    """Compute F1-F4 for cluster X_c against each population in X_others.

    Skips populations with fewer than 2 samples.
    """
    out: dict[str, list[float]] = {k: [] for k in _F_KEYS}
    for X_o in X_others:
        if len(X_o) < 2:
            continue
        out["f1"].append(_f1_pair(X_c, X_o))
        out["f2"].append(_f2_pair(X_c, X_o))
        out["f3"].append(_f3_pair(X_c, X_o))
        out["f4"].append(_f4_pair(X_c, X_o))
    return out


@timed
def compute_f_measures(
    X_num: np.ndarray,
    y_cluster: np.ndarray,
    top_k_map: dict[str, list[str]],
    *,
    metric: str = "cosine",
) -> dict[str, dict[str, float | None]]:
    """F1-F4 per cluster vs the top-K adversarial clusters, as min/mean/max.

    Noise (-1) is excluded. metric="cosine" L2-normalises samples first (angular
    space, matching the Gower-cosine k-NN); "euclidean" uses raw samples.
    Output keys: f{1..4}_{min,mean,max}.
    """
    mask_valid = y_cluster != -1
    X_raw = X_num[mask_valid]
    X_v = _l2_normalize(X_raw) if metric == "cosine" else X_raw
    yk_v = y_cluster[mask_valid]

    cluster_block: dict[str, np.ndarray] = {
        str(int(cid)): X_v[yk_v == cid] for cid in np.unique(yk_v)
    }

    result: dict[str, dict[str, float | None]] = {}
    for cid_str, X_c in tqdm(
        cluster_block.items(), desc="F measures", unit="cluster", leave=False
    ):
        row = make_null_row(_F_KEYS)
        if len(X_c) < 2 or X_c.shape[1] == 0:
            result[cid_str] = row
            continue

        cluster_blocks = [
            cluster_block[ac]
            for ac in top_k_map.get(cid_str, [])
            if ac in cluster_block
        ]

        vals = _pair_block(X_c, cluster_blocks)
        for fk in _F_KEYS:
            mn, me, mx = aggregate_min_mean_max(vals[fk])
            row[f"{fk}_min"] = mn
            row[f"{fk}_mean"] = me
            row[f"{fk}_max"] = mx

        result[cid_str] = row

    return result
