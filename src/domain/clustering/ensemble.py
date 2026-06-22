from collections.abc import Callable

import hdbscan
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from src.domain.clustering.base import ClusterFn

ConsensusReporter = Callable[[dict], None]


def _coassociation_matrix(
    labels_arr_sub: np.ndarray, *, weight_voters: bool = True
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Symmetric (s, s) co-association in [0, 1] + pairwise per-algo agreement.

    Noise = no-vote at variable denominator. With reliability weights w_m, for a
    pair (i, j):
        num[i,j] = sum_m w_m * [label_m(i) == label_m(j) AND both != -1]
        den[i,j] = sum_m w_m * [both != -1]
        co[i,j]  = num/den if den > 0 else 0.0

    Each voter weight `w_m = coverage_m` (its non-noise fraction): a voter that
    abstains on most points contributes little signal and is down-weighted toward 0
    (soft floor). Coverage is assumption-free — unlike "agreement with the other
    voters", which penalises a lone correct voter when the majority is wrong (e.g.
    spectral on concentric circles). With `weight_voters=False` all weights are 1
    (unweighted, legacy behaviour).

    Returns `(co_assoc, pairwise, voter_weights)`. `pairwise` is `{(i,j): agreement}`
    for each algo pair (i < j), the fraction of both-classified point-pairs on which
    the two algorithms agree on "in-same-cluster" (XNOR).
    """
    M, s = labels_arr_sub.shape
    valid = labels_arr_sub != -1
    same_cluster: list[np.ndarray] = []
    for m in range(M):
        l = labels_arr_sub[m]
        v = valid[m]
        pair_valid = v[:, None] & v[None, :]
        sc = (l[:, None] == l[None, :]) & pair_valid
        same_cluster.append(sc)

    triu = np.triu_indices(s, k=1)
    pairwise: dict[tuple[int, int], float] = {}
    for i in range(M):
        for j in range(i + 1, M):
            both_valid = valid[i][:, None] & valid[i][None, :] & valid[j][:, None] & valid[j][None, :]
            both_pairs = both_valid[triu]
            agree = (same_cluster[i] == same_cluster[j])[triu]
            agree_on_valid = agree[both_pairs]
            pairwise[(i, j)] = float(agree_on_valid.mean()) if agree_on_valid.size else 0.0

    if weight_voters and M > 1:
        weights = valid.mean(axis=1).astype(np.float32)  # coverage = 1 - noise_ratio
    else:
        weights = np.ones(M, dtype=np.float32)

    num = np.zeros((s, s), dtype=np.float32)
    den = np.zeros((s, s), dtype=np.float32)
    for m in range(M):
        w = float(weights[m])
        pair_valid = valid[m][:, None] & valid[m][None, :]
        den += w * pair_valid.astype(np.float32)
        num += w * same_cluster[m].astype(np.float32)
    den_safe = np.where(den == 0, 1.0, den)
    co = (num / den_safe).astype(np.float32)
    co[den == 0] = 0.0
    np.fill_diagonal(co, 1.0)

    return co, pairwise, weights


def _agreement_with_sub(labels_arr: np.ndarray, sub_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    """Co-association agreement of points j_idx vs all sub_idx points.

    Same noise = no-vote convention. Returns (len(j_idx), len(sub_idx)) float32.
    """
    sub_labels = labels_arr[:, sub_idx]
    j_labels = labels_arr[:, j_idx]
    sub_valid = sub_labels != -1
    j_valid = j_labels != -1
    match = (
        (j_labels[:, :, None] == sub_labels[:, None, :])
        & j_valid[:, :, None]
        & sub_valid[:, None, :]
    )
    both_valid = j_valid[:, :, None] & sub_valid[:, None, :]
    num = match.sum(axis=0).astype(np.float32)
    den = both_valid.sum(axis=0).astype(np.float32)
    return np.where(den > 0, num / np.where(den == 0, 1.0, den), 0.0).astype(np.float32)


def _propagate_labels(
    labels_arr: np.ndarray,
    sub_idx: np.ndarray,
    sub_labels: np.ndarray,
    non_sub: np.ndarray,
    floor: float,
    full_labels: np.ndarray,
    batch_size: int = 1000,
) -> dict:
    """Assign non-sub points to the cluster of the sub-point with highest agreement."""
    max_agreements: list[float] = []
    low_conf = 0
    for start in range(0, len(non_sub), batch_size):
        batch = non_sub[start : start + batch_size]
        agree = _agreement_with_sub(labels_arr, sub_idx, batch)
        best = agree.argmax(axis=1)
        max_vals = agree[np.arange(len(batch)), best]
        max_agreements.extend(max_vals.tolist())
        best_labels = sub_labels[best]
        noise_mask = (best_labels == -1) | (max_vals == 0) | (max_vals < floor)
        full_labels[batch] = np.where(noise_mask, -1, best_labels)
        low_conf += int(noise_mask.sum())
    if not max_agreements:
        return {"mean_max": 0.0, "low_conf_ratio": 0.0}
    return {
        "mean_max": float(np.mean(max_agreements)),
        "low_conf_ratio": low_conf / len(max_agreements),
    }


def _refine_geometry(
    X_sub: np.ndarray,
    labels: np.ndarray,
    *,
    margin: float = 0.8,
    n_neighbors: int = 10,
) -> tuple[np.ndarray, int]:
    """Reassign geometric stragglers via a local kNN-majority vote in feature space.

    Consensus clusters live in co-association space; this corrects points whose
    feature-space neighbourhood contradicts the assignment. A non-noise point moves
    to cluster `b` only if at least `margin` of its k nearest non-noise neighbours
    belong to `b` while at most `1 - margin` share its own label — i.e. it sits
    inside another cluster's territory. Local (not centroid-based), so it respects
    non-convex shapes; conservative, so coherent assignments stay put. Noise (-1)
    is left untouched. Returns `(labels, n_refined)`.

    Operates on the consensus subsample (X_sub already metric-adjusted upstream:
    L2-normalised when metric=cosine, so Euclidean neighbours match cosine order).
    """
    non_noise = labels != -1
    cids = np.unique(labels[non_noise])
    idx = np.where(non_noise)[0]
    if cids.size < 2 or idx.size <= n_neighbors:
        return labels, 0

    k = min(n_neighbors, idx.size - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_sub[idx])
    nbr = nn.kneighbors(X_sub[idx], return_distance=False)[:, 1:]  # drop self

    code = np.searchsorted(cids, labels[idx])           # (m,) own cluster as 0..K-1
    nbr_code = code[nbr]                                 # (m, k) neighbour clusters
    counts = np.zeros((idx.size, cids.size), dtype=np.int32)
    np.add.at(counts, (np.arange(idx.size)[:, None], nbr_code), 1)
    frac = counts / k

    rows = np.arange(idx.size)
    best = frac.argmax(axis=1)
    best_frac = frac[rows, best]
    own_frac = frac[rows, code]
    reassign = (best != code) & (best_frac >= margin) & (own_frac <= 1.0 - margin)
    if not reassign.any():
        return labels, 0
    refined = labels.copy()
    refined[idx[reassign]] = cids[best[reassign]]
    return refined, int(reassign.sum())


def _build_diagnostics(
    labels_arr: np.ndarray,
    sub_idx: np.ndarray,
    sub_labels: np.ndarray,
    co: np.ndarray,
    pairwise_idx: dict,
    voter_weights: np.ndarray,
    s: int,
    n: int,
    M: int,
) -> dict:
    """Compute intra/inter agreement, histogram, ARIs from the post-HDBSCAN sub partition."""
    triu = np.triu_indices(s, k=1)
    co_triu = co[triu]
    sub_pairs = sub_labels[triu[0]] == sub_labels[triu[1]]
    non_noise_pairs = (sub_labels[triu[0]] != -1) & (sub_labels[triu[1]] != -1)
    intra_mask = sub_pairs & non_noise_pairs
    inter_mask = (~sub_pairs) & non_noise_pairs
    mean_intra = float(co_triu[intra_mask].mean()) if intra_mask.any() else 0.0
    mean_inter = float(co_triu[inter_mask].mean()) if inter_mask.any() else 0.0

    histo_int = np.clip(np.round(co_triu * M).astype(int), 0, M)
    counts = np.bincount(histo_int, minlength=M + 1).tolist()

    return {
        "subsample_fraction": s / n if n > 0 else 1.0,
        "mean_intra_cluster_agreement": mean_intra,
        "mean_inter_cluster_agreement": mean_inter,
        "consensus_separation": mean_intra - mean_inter,
        "agreement_histogram": counts,
        "pairwise_algo_agreement_idx": {
            f"{i}-{j}": v for (i, j), v in pairwise_idx.items()
        },
        "algorithm_consensus_ari_idx": [
            float(adjusted_rand_score(sub_labels, labels_arr[m, sub_idx]))
            for m in range(M)
        ],
        "voter_weights_idx": [float(w) for w in voter_weights],
    }


def compute_coassociation_labels(
    labels_list: list[np.ndarray],
    *,
    threshold: float,
    min_cluster_size: int,
    max_fit_samples: int,
    random_state: int = 0,
    propagation_confidence_floor: float = 0.0,
    X_fit: np.ndarray | None = None,
    weight_voters: bool = True,
    refine_geometry: bool = True,
    refine_margin: float = 0.8,
) -> tuple[np.ndarray, dict]:
    """Consensus clustering via co-association matrix + HDBSCAN(precomputed).

    On a subsample of `max_fit_samples` points: build a (reliability-weighted when
    `weight_voters`) co-association, run HDBSCAN with
    `cluster_selection_epsilon = 1 - threshold`. When `refine_geometry` and `X_fit`
    is given, a kNN-majority feature-space pass reassigns geometric stragglers in
    the seed partition; the remaining points are then propagated via co-association.
    """
    if not labels_list:
        raise ValueError("compute_coassociation_labels: empty labels_list")
    n = labels_list[0].shape[0]
    if any(l.shape[0] != n for l in labels_list):
        raise ValueError(
            f"inconsistent label-array lengths: {[l.shape[0] for l in labels_list]}"
        )

    labels_arr = np.stack(labels_list)
    M = labels_arr.shape[0]

    s = min(n, max_fit_samples)
    if n > max_fit_samples:
        rng = np.random.default_rng(random_state)
        sub_idx = np.sort(rng.choice(n, s, replace=False))
    else:
        sub_idx = np.arange(n)

    sub = labels_arr[:, sub_idx]
    co, pairwise_idx, voter_weights = _coassociation_matrix(sub, weight_voters=weight_voters)

    dist = (1.0 - co).astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    eps = max(0.0, float(1.0 - threshold))
    mcs = max(2, int(min_cluster_size))
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=mcs,
        cluster_selection_epsilon=eps,
    )
    sub_labels = clusterer.fit_predict(dist).astype(np.int64)

    diagnostics = _build_diagnostics(
        labels_arr, sub_idx, sub_labels, co, pairwise_idx, voter_weights, s, n, M
    )

    # Refine the seed partition (subsample only → cost bounded by max_fit_samples);
    # propagation then follows the refined structure.
    if refine_geometry and X_fit is not None:
        sub_labels, n_refined = _refine_geometry(
            X_fit[sub_idx], sub_labels, margin=refine_margin
        )
        diagnostics["n_refined"] = n_refined
    else:
        diagnostics["n_refined"] = 0

    full_labels = np.full(n, -1, dtype=np.int64)
    full_labels[sub_idx] = sub_labels
    if n > s:
        non_sub = np.setdiff1d(np.arange(n), sub_idx, assume_unique=True)
        prop_stats = _propagate_labels(
            labels_arr, sub_idx, sub_labels, non_sub,
            propagation_confidence_floor, full_labels,
        )
        diagnostics["propagation_mean_max_agreement"] = prop_stats["mean_max"]
        diagnostics["propagation_low_confidence_ratio"] = prop_stats["low_conf_ratio"]
    else:
        diagnostics["propagation_mean_max_agreement"] = None
        diagnostics["propagation_low_confidence_ratio"] = None

    return full_labels, diagnostics


def make_ensemble_cluster_fn(
    cluster_fns: list[ClusterFn],
    *,
    threshold: float = 0.5,
    min_consensus_size: int = 1,
    max_fit_samples: int = 10_000,
    random_state: int = 0,
    consensus_reporter: ConsensusReporter | None = None,
    propagation_confidence_floor: float = 0.0,
    weight_voters: bool = True,
    refine_geometry: bool = True,
    refine_margin: float = 0.8,
    algorithms: dict[str, dict]
) -> ClusterFn:
    """Compose multiple ClusterFns via co-association + HDBSCAN(precomputed).

    `min_consensus_size` is the absolute HDBSCAN(precomputed) min_cluster_size
    on the co-association matrix. `consensus_reporter` receives the diagnostics
    dict after each consensus computation. `weight_voters`, `refine_geometry`
    and `refine_margin` control the reliability weighting and the kNN-majority
    feature-space refinement of the consensus.
    """

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None = None) -> np.ndarray:
        labels_and_models_list = [fn(X_num, X_cat) for fn in cluster_fns] # Each is a tuple (labels, best_model)

        labels_list = [_[0] for _ in labels_and_models_list] # Only the labels

        clustering_name_to_model = {list(algorithms.keys())[i] : labels_and_models_list[i][1] for i in range(len(algorithms))} # The ordering is kept between the cluster_fns and the algorithms

        labels, diagnostics = compute_coassociation_labels(
            labels_list,
            threshold=threshold,
            min_cluster_size=int(min_consensus_size),
            max_fit_samples=max_fit_samples,
            random_state=random_state,
            propagation_confidence_floor=propagation_confidence_floor,
            X_fit=X_num,
            weight_voters=weight_voters,
            refine_geometry=refine_geometry,
            refine_margin=refine_margin,
        )
        if consensus_reporter is not None:
            consensus_reporter(diagnostics)
        return labels, clustering_name_to_model

    return _fn
