import itertools
import time
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.base import ClusterMixin
from tqdm import tqdm

from src.core.utils import timed
from src.domain.analysis.complexity.shared import (
    hybrid_row_batch,
    hybrid_row_batch_euclidean,
    l2_normalize,
)

FitFn = Callable[..., np.ndarray]  # (X_num, X_cat=None, **params) -> labels
ClusterFn = Callable[
    [np.ndarray, np.ndarray | None], np.ndarray
]  # (X_num, X_cat) -> labels
SilhouetteFn = Callable[
    [np.ndarray, np.ndarray | None, np.ndarray], float
]  # (X_num, X_cat, labels) -> sil


def cluster_size_balance(labels: np.ndarray) -> float:
    """Normalized entropy of non-noise cluster sizes in [0, 1] (1 = uniform)."""
    mask = labels != -1
    if not mask.any():
        return 0.0
    _, counts = np.unique(labels[mask], return_counts=True)
    k = counts.size
    if k < 2:
        return 0.0
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    return h / float(np.log(k))


def _measure(labels: np.ndarray, score: float, combo: dict, duration_s: float, model : ClusterMixin) -> dict:
    """Headline metrics for a single clustering candidate."""
    n = int(labels.shape[0])
    n_noise = int((labels == -1).sum())
    n_clusters = int(np.unique(labels[labels != -1]).size) if n - n_noise > 0 else 0
    return {
        "combo": combo,
        "score": score,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / n if n > 0 else 0.0,
        "size_balance": cluster_size_balance(labels),
        "duration_s": duration_s,
        "model" : model
    }


def _subsample(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    max_samples: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Random subsample of X_num (and X_cat) to at most max_samples rows."""
    n = X_num.shape[0]
    if n <= max_samples:
        return X_num, X_cat
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X_num[idx], (X_cat[idx] if X_cat is not None else None)


def _score_silhouette(
    X_num: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Silhouette on non-noise points only. Returns -inf on failure or < 2 clusters."""
    mask = labels != -1
    if mask.sum() < 2:
        return float("-inf")
    unique = np.unique(labels[mask])
    if len(unique) < 2:
        return float("-inf")
    try:
        return float(silhouette_score(X_num[mask], labels[mask], metric=metric))
    except Exception:
        return float("-inf")


def pairwise_hybrid_distance(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    metric: str = "cosine",
) -> np.ndarray:
    """Full pairwise Gower-hybrid distance matrix in [0, 1].

    Same formula as the complexity k-NN graph (cosine or range-normalised
    Manhattan on numerics + Hamming on categorics), materialised densely —
    only suitable for scoring subsamples (O(n^2) memory).
    """
    d_num = X_num.shape[1]
    d_cat = X_cat.shape[1] if X_cat is not None else 0
    if metric == "cosine":
        X_norm = l2_normalize(X_num)
        return hybrid_row_batch(X_norm, X_cat, X_norm, X_cat, d_num, d_cat)
    feat_ranges = X_num.max(axis=0) - X_num.min(axis=0)
    return hybrid_row_batch_euclidean(
        X_num, X_cat, X_num, X_cat, d_num, d_cat, feat_ranges
    )


def assign_nearest_centroid(
    X_num: np.ndarray,
    centroids: dict,
    *,
    metric: str,
    candidate_ids: Sequence[int] | None = None,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Assign each row to the nearest centroid id (batched).

    Inductive counterpart of fitting a clusterer: maps new points onto an
    existing set of centroids. If `candidate_ids` is given, only those centroids
    are eligible. Centroid keys (str or int) are coerced to int64 so the output
    matches the `cluster` column dtype.
    """
    items = [(int(k), v) for k, v in centroids.items()]
    if candidate_ids is not None:
        cand = {int(c) for c in candidate_ids}
        items = [(k, v) for k, v in items if k in cand]
    id_arr = np.array([k for k, _ in items], dtype=np.int64)
    C = np.array([np.asarray(v, dtype=np.float64) for _, v in items], dtype=np.float64)
    result = np.empty(len(X_num), dtype=np.int64)
    for start in range(0, len(X_num), batch_size):
        batch = X_num[start : start + batch_size]
        D = pairwise_distances(batch, C, metric=metric)
        result[start : start + batch_size] = id_arr[D.argmin(axis=1)]
    return result


def assign_clusters_within_class(
    X_num: np.ndarray,
    y_class: np.ndarray,
    centroids: dict,
    cluster_to_class: dict[int, int],
    *,
    metric: str,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Assign each row to the nearest centroid among clusters of its own class.

    Label-aware, inductive assignment: a row is matched only against centroids of
    clusters belonging to its class. Rows whose class has no train cluster get -1
    (noise sentinel) — guarded so an empty candidate set never reaches
    `pairwise_distances`.
    """
    result = np.full(len(X_num), -1, dtype=np.int64)
    for cls in np.unique(y_class):
        candidate_ids = [cid for cid, c in cluster_to_class.items() if c == cls]
        if not candidate_ids:
            continue
        rows = np.where(y_class == cls)[0]
        result[rows] = assign_nearest_centroid(
            X_num[rows],
            centroids,
            metric=metric,
            candidate_ids=candidate_ids,
            batch_size=batch_size,
        )
    return result


def make_hybrid_silhouette_fn(
    metric: str = "cosine",
    max_scoring_samples: int = 5_000,
    random_state: int = 0,
) -> SilhouetteFn:
    """Build a silhouette scorer on the Gower-hybrid distance (mixed features).

    Operates on a random subsample of at most `max_scoring_samples` non-noise
    points to bound the dense pairwise matrix. Returns -inf on degenerate input.
    """

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None, labels: np.ndarray) -> float:
        idx = np.where(labels != -1)[0]
        if idx.size < 2:
            return float("-inf")
        if idx.size > max_scoring_samples:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(idx, size=max_scoring_samples, replace=False)
        sub_labels = labels[idx]
        if np.unique(sub_labels).size < 2:
            return float("-inf")
        dm = pairwise_hybrid_distance(
            X_num[idx], X_cat[idx] if X_cat is not None else None, metric=metric
        )
        np.fill_diagonal(dm, 0.0)
        try:
            return float(silhouette_score(dm, sub_labels, metric="precomputed"))
        except Exception:
            return float("-inf")

    return _fn


@timed
def grid_search(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    fit_fn: FitFn,
    param_grid: dict[str, list],
    *,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    noise_penalty: float = 3.0,
    resolution_weight: float = 0.1,
    silhouette_fn: SilhouetteFn | None = None,
    **fixed_params,
) -> tuple[dict, ClusterMixin]:
    """Grid search scored by silhouette − noise_penalty·noise_ratio + resolution.

    The silhouette is monotone-decreasing in k on weakly-structured data, so on
    its own it favours coarse partitions and starves the downstream
    complexity→failure regression of cluster-level support. A gentle
    `resolution_weight`-scaled tilt, increasing with the cluster count and
    self-normalised by the finest partition in the sweep, is added so a flat
    silhouette drifts to the fine end of the (data-relative) `n_clusters` band;
    a genuine silhouette peak still dominates the small tilt, so well-separated
    data keeps its natural resolution. The tilt has no fixed target — the
    resolution band lives in the candidate grid (see `compose._n_clusters_grid`),
    not here.

    Methods without noise (k-means, birch) score on the plain silhouette.
    `silhouette_fn` overrides the silhouette term (e.g. Gower-hybrid for
    mixed-feature algorithms); default is Euclidean on `X_num`. fixed_params are
    forwarded to fit_fn unchanged.

    Returns {"best": entry, "sweep": [entry, ...]}; failed fits get score=-inf,
    error=True. Raises if no candidate produced a clustering.
    """
    sub_num, sub_cat = _subsample(X_num, X_cat, max_fit_samples, random_state)

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    sweep: list[dict] = []

    # Pass 1: fit each combo, record its silhouette (provisional score = silhouette).
    for combo_values in tqdm(
        itertools.product(*values),
        total=int(np.prod([len(v) for v in values])),
        desc="Grid search",
    ):
        combo = dict(zip(keys, combo_values))
        t0 = time.perf_counter()
        try:
            labels, model = fit_fn(sub_num, X_cat=sub_cat, **combo, **fixed_params)
        except Exception:
            sweep.append(
                {
                    "combo": combo,
                    "score": float("-inf"),
                    "n_clusters": 0,
                    "n_noise": 0,
                    "noise_ratio": 0.0,
                    "size_balance": 0.0,
                    "duration_s": time.perf_counter() - t0,
                    "error": True,

                    # Save exclusively the first model, just in case all of the entries have errors, as you still need a best_model to save to you'll just use that one.
                    # saving all of the models and then popping them before returning is an unnecessary overhead.
                    "model" : model if not sweep else "No model saved as this entry has errors.",                     
                }
            )
            continue

        duration = time.perf_counter() - t0
        sil = (
            silhouette_fn(sub_num, sub_cat, labels)
            if silhouette_fn is not None
            else _score_silhouette(sub_num, labels)
        )
        entry = _measure(labels, sil, combo, duration, model)
        entry["silhouette"] = sil
        sweep.append(entry)

    # Pass 2: final score = silhouette − noise_penalty·noise_ratio + resolution
    # tilt, where the tilt grows with the cluster count, self-normalised by the
    # finest partition in the sweep. Done after the loop so HDBSCAN (whose k is
    # only known post-fit) shares the same normalisation as the n_clusters grids.
    valid = [e for e in sweep if not e.get("error")]
    max_k = max((e["n_clusters"] for e in valid), default=0)
    best_score = float("-inf")
    best_entry: dict | None = None
    best_model : ClusterMixin = None
    for e in valid:
        tilt = resolution_weight * (e["n_clusters"] / max_k) if max_k > 0 else 0.0
        e["resolution_tilt"] = tilt
        e["score"] = e["silhouette"] - noise_penalty * e["noise_ratio"] + tilt
        current_entry_model = e.pop("model") # Must pop anyway, or the metadata in json will save the entire model and break as it's not JSON serializable
        if e["score"] > best_score:
            best_score = e["score"]
            best_model = current_entry_model
            best_entry = e
            
    first_entry_model = sweep[0].pop("model", "Already popped") # Remove only first in case it is the one with errors
    if best_entry is None:
        # All of the entries were invalid. Must use the first, which has errors. 
        # If it has errors, the model key had not been popped yet, so first_entry_model will be the actual model
        best_model = first_entry_model
        best_entry = sweep[0] if sweep else None

    if best_entry is None:
        raise RuntimeError(
            "grid_search: no valid clustering found across all parameter combinations."
        )

    return {"best": best_entry, "sweep": sweep}, best_model
