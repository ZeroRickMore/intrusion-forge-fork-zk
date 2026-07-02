import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.core.utils import timed


def aggregate_min_mean_max(
    vals: list[float],
) -> tuple[float | None, float | None, float | None]:
    """Aggregate a list of values into (min, mean, max). Returns Nones if empty."""
    if not vals:
        return None, None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.min()), float(arr.mean()), float(arr.max())


def make_null_row(metric_keys: tuple[str, ...]) -> dict[str, float | None]:
    """Build a dict of `f"{metric}_{stat}": None` for the standard pairwise
    output schema (stats min/mean/max)."""
    return {
        f"{m}_{stat}": None
        for m in metric_keys
        for stat in ("min", "mean", "max")
    }


def _l2_normalize(X_num: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2-normalize so that Euclidean on unit vectors maps to cosine."""
    norms = np.linalg.norm(X_num, axis=1, keepdims=True)
    return X_num / np.maximum(norms, eps)


def _hybrid_row_batch(
    X_num_norm: np.ndarray,
    X_cat: np.ndarray | None,
    query_num_norm: np.ndarray,
    query_cat: np.ndarray | None,
    d_num: int,
    d_cat: int,
) -> np.ndarray:
    """Gower-cosine hybrid distance (cosine for numerics, Hamming for categorics).

    d = ( d_num * cos_dist(query_num, ref_num) + Σ_j [query_cat_j != ref_cat_j] ) / (d_num + d_cat)

    cos_dist = clip(||q/||q|| - r/||r||||^2 / 2, 0, 1) ∈ [0, 1].
    The numeric block is weighted by `d_num` to preserve Gower proportionality.
    Returns shape (b, n) distance matrix in [0, 1].
    """
    euclid = cdist(query_num_norm, X_num_norm, metric="euclidean")
    dist = np.clip(euclid**2 / 2, 0.0, 1.0) * d_num

    if X_cat is not None and d_cat > 0:
        for f in range(d_cat):
            dist += (query_cat[:, f : f + 1] != X_cat[:, f]).astype(np.float64)

    return dist / (d_num + d_cat)


def _hybrid_row_batch_euclidean(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    query_num: np.ndarray,
    query_cat: np.ndarray | None,
    d_num: int,
    d_cat: int,
    feat_ranges: np.ndarray,
) -> np.ndarray:
    """Gower-Euclidean hybrid distance (per-feature range-normalised Manhattan + Hamming).

    d = ( Σ_f |x_f - y_f| / range_f + Σ_j [x_cat_j != y_cat_j] ) / (d_num + d_cat)

    feat_ranges[f] = max(X_num[:, f]) - min(X_num[:, f]) for each numerical feature f.
    Returns shape (b, n) distance matrix in [0, 1].
    """
    dist = np.zeros((query_num.shape[0], X_num.shape[0]), dtype=np.float64)
    for f in range(d_num):
        r = max(float(feat_ranges[f]), 1e-8)
        dist += np.abs(query_num[:, f : f + 1] - X_num[:, f]) / r

    if X_cat is not None and d_cat > 0:
        for f in range(d_cat):
            dist += (query_cat[:, f : f + 1] != X_cat[:, f]).astype(np.float64)

    return dist / (d_num + d_cat)


@timed
def build_knn_graph(
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    k: int,
    *,
    metric: str = "cosine",
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a k-NN graph via batched Gower-hybrid distance, never materialising
    the full n×n matrix. Returns (indices, distances), both shape (n, k).

    metric="cosine": cosine on L2-normalised numerics + Hamming on categorics.
    metric="euclidean": range-normalised Manhattan on numerics + Hamming.
    """
    n, d_num = X_num.shape
    d_cat = X_cat.shape[1] if X_cat is not None else 0
    effective_k = min(k, n - 1)

    if metric == "cosine":
        X_num_norm = _l2_normalize(X_num)

        def batch_dists(start: int, end: int) -> np.ndarray:
            q_cat = X_cat[start:end] if X_cat is not None else None
            return _hybrid_row_batch(
                X_num_norm, X_cat, X_num_norm[start:end], q_cat, d_num, d_cat
            )
    else:
        feat_ranges = X_num.max(axis=0) - X_num.min(axis=0)

        def batch_dists(start: int, end: int) -> np.ndarray:
            q_cat = X_cat[start:end] if X_cat is not None else None
            return _hybrid_row_batch_euclidean(
                X_num, X_cat, X_num[start:end], q_cat, d_num, d_cat, feat_ranges
            )

    indices = np.empty((n, effective_k), dtype=np.int64)
    distances = np.empty((n, effective_k), dtype=np.float64)
    for start in tqdm(
        range(0, n, batch_size), desc="k-NN graph", unit="batch", leave=False
    ):
        end = min(start + batch_size, n)
        dists = batch_dists(start, end)
        batch_idx = np.arange(end - start)
        dists[batch_idx, start + batch_idx] = np.inf

        part = np.argpartition(dists, effective_k, axis=1)[:, :effective_k]
        part_d = np.take_along_axis(dists, part, axis=1)
        order = np.argsort(part_d, axis=1)
        indices[start:end] = np.take_along_axis(part, order, axis=1)
        distances[start:end] = np.take_along_axis(part_d, order, axis=1)

    return indices, distances


def _to_sparse_csr(
    indices: np.ndarray, distances: np.ndarray, n: int
) -> scipy.sparse.csr_matrix:
    """Symmetrise the directed k-NN graph keeping the minimum distance per edge."""
    k = indices.shape[1]
    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    data = distances.ravel()

    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    sym_data = np.concatenate([data, data])

    order = np.lexsort((sym_data, sym_cols, sym_rows))
    sym_rows = sym_rows[order]
    sym_cols = sym_cols[order]
    sym_data = sym_data[order]

    unique_mask = np.ones(len(sym_rows), dtype=bool)
    unique_mask[1:] = (sym_rows[1:] != sym_rows[:-1]) | (sym_cols[1:] != sym_cols[:-1])

    mat = scipy.sparse.csr_matrix(
        (sym_data[unique_mask], (sym_rows[unique_mask], sym_cols[unique_mask])),
        shape=(n, n),
    )
    mat.setdiag(0)
    mat.eliminate_zeros()
    return mat


def _bridge_disconnected(
    mat: scipy.sparse.csr_matrix,
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    d_num: int,
    d_cat: int,
    metric: str = "cosine",
    feat_ranges: np.ndarray | None = None,
) -> scipy.sparse.csr_matrix:
    """Add one cross-component bridge edge per disconnected component.

    Used when the k-NN graph is disconnected (categoricals can partition the
    space such that the k neighbours of every node sit in the same category).
    metric controls which distance function is used for bridge distances.
    """
    n_comp, comp_labels = scipy.sparse.csgraph.connected_components(mat, directed=False)
    if n_comp == 1:
        return mat

    mat = mat.tolil()
    ref = int(np.where(comp_labels == 0)[0][0])

    if metric == "cosine":
        X_num_norm = _l2_normalize(X_num)
        q_num_prep = X_num_norm[ref : ref + 1]
        q_cat = X_cat[ref : ref + 1] if X_cat is not None else None
        dists_row = _hybrid_row_batch(
            X_num_norm, X_cat, q_num_prep, q_cat, d_num, d_cat
        )[0]
    else:
        if feat_ranges is None:
            feat_ranges = X_num.max(axis=0) - X_num.min(axis=0)
        q_num = X_num[ref : ref + 1]
        q_cat = X_cat[ref : ref + 1] if X_cat is not None else None
        dists_row = _hybrid_row_batch_euclidean(
            X_num, X_cat, q_num, q_cat, d_num, d_cat, feat_ranges
        )[0]

    for ci in range(1, n_comp):
        nodes_ci = np.where(comp_labels == ci)[0]
        best_j = int(nodes_ci[dists_row[nodes_ci].argmin()])
        d = max(float(dists_row[best_j]), 1e-10)
        mat[ref, best_j] = d
        mat[best_j, ref] = d
        comp_labels[nodes_ci] = 0
    return mat.tocsr()


def build_approx_mst(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    X_num: np.ndarray,
    X_cat: np.ndarray | None,
    *,
    metric: str = "cosine",
) -> np.ndarray:
    """Approximate MST on the sparse k-NN graph. Returns an (E, 2) int64 array
    of MST edges (row, col). One bridge edge is added per disconnected
    component so the MST connects every cluster.
    metric is forwarded to _bridge_disconnected for the bridge distance computation.
    """
    n, d_num = X_num.shape
    d_cat = X_cat.shape[1] if X_cat is not None else 0

    feat_ranges = (
        X_num.max(axis=0) - X_num.min(axis=0) if metric == "euclidean" else None
    )

    graph = _to_sparse_csr(knn_indices, knn_distances, n)
    graph = _bridge_disconnected(
        graph, X_num, X_cat, d_num, d_cat, metric=metric, feat_ranges=feat_ranges
    )
    mst = scipy.sparse.csgraph.minimum_spanning_tree(graph).tocoo()
    if mst.nnz == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack((mst.row, mst.col)).astype(np.int64, copy=False)


def topk_adversarial_clusters(
    centroid_matrix: np.ndarray,
    cluster_ids: list[str],
    id_to_class: dict[str, int],
    top_k: int,
    *,
    metric: str = "euclidean",
) -> dict[str, list[str]]:
    """For each cluster, return the top-K nearest cluster IDs of a different class.

    Result is sorted by ascending centroid distance and capped at `top_k`
    (or all available adversarial clusters if fewer).
    metric: "euclidean" (default) or "cosine".
    """
    if centroid_matrix.shape[0] == 0:
        return {}
    pw = cdist(centroid_matrix, centroid_matrix, metric=metric)
    np.fill_diagonal(pw, np.inf)
    classes = np.array(
        [id_to_class.get(cid, -1) for cid in cluster_ids], dtype=np.int64
    )
    out: dict[str, list[str]] = {}
    for i, cid in enumerate(cluster_ids):
        cls_c = id_to_class.get(cid)
        if cls_c is None:
            out[cid] = []
            continue
        adv_idx = np.where(classes != cls_c)[0]
        if adv_idx.size == 0:
            out[cid] = []
            continue
        order = np.argsort(pw[i, adv_idx])
        take = min(top_k, adv_idx.size)
        out[cid] = [cluster_ids[int(adv_idx[j])] for j in order[:take]]
    return out
