import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples

from src.core.utils import timed


def _approx_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    max_samples: int = 10_000,
    min_per_cluster: int = 50,
    random_state: int = 42,
) -> np.ndarray | None:
    """Approximate silhouette scores with stratified subsampling.

    Non-sampled points receive NaN. Returns None if < 2 unique labels.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None

    n = len(X)
    if n <= max_samples:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(random_state)
        idx_parts: list[np.ndarray] = []
        for lbl in unique_labels:
            members = np.where(labels == lbl)[0]
            take = min(len(members), max(min_per_cluster, 1))
            idx_parts.append(rng.choice(members, size=take, replace=False))
        guaranteed = np.concatenate(idx_parts)
        remaining = max_samples - len(guaranteed)
        if remaining > 0:
            pool = np.setdiff1d(np.arange(n), guaranteed)
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            idx = np.concatenate([guaranteed, extra])
        else:
            idx = guaranteed

    try:
        scores = silhouette_samples(X[idx], labels[idx], metric=metric)
    except ValueError:
        return None

    full = np.full(n, np.nan)
    full[idx] = scores
    return full


def _dispersion(
    samples: np.ndarray, centroid: np.ndarray, metric: str
) -> tuple[float | None, float | None]:
    """(max_dispersion, p95_dispersion) for given metric."""
    if len(samples) == 0:
        return None, None
    dists = pairwise_distances(samples, centroid.reshape(1, -1), metric=metric).ravel()
    return float(np.max(dists)), float(np.percentile(dists, 95))


def _nearest_other(pw_row: np.ndarray) -> float | None:
    """Min centroid distance to any other cluster (diagonal already set to inf)."""
    finite = pw_row[np.isfinite(pw_row)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


def _spherical_centroid(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Fréchet mean under cosine distance: mean of L2-normalised samples, re-normalised."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.maximum(norms, eps)
    sph = X_norm.mean(axis=0)
    sph_norm = np.linalg.norm(sph)
    return sph / max(sph_norm, eps)


@timed
def compute_cluster_geometry(
    X_num: np.ndarray,
    y_cluster: np.ndarray,
    centroids: dict[str, list[float]],
    *,
    metric: str = "cosine",
    random_state: int = 42,
) -> dict[str, dict[str, float | None]]:
    """Geometry measures per cluster under a single metric ("cosine"/"euclidean").

    Noise (-1) is excluded. metric controls centroid type, pairwise distances and
    silhouette. Output keys: max_dispersion, p95_dispersion,
    dist_to_nearest_centroid, p5_silhouette, frac_at_risk.
    """
    mask_valid = y_cluster != -1
    X_v = X_num[mask_valid]
    yk_v = y_cluster[mask_valid]

    present_ids = [str(cid) for cid in np.unique(yk_v) if str(cid) in centroids]
    if not present_ids:
        return {}

    if metric == "cosine":
        centroid_matrix = np.stack(
            [_spherical_centroid(X_v[yk_v == int(cid)]) for cid in present_ids]
        )
    else:
        centroid_matrix = np.stack(
            [np.asarray(centroids[cid], dtype=np.float64) for cid in present_ids]
        )

    id_to_idx = {cid: i for i, cid in enumerate(present_ids)}

    pw = pairwise_distances(centroid_matrix, metric=metric)
    np.fill_diagonal(pw, np.inf)

    sil = _approx_silhouette(X_v, yk_v, metric=metric, random_state=random_state)

    result: dict[str, dict[str, float | None]] = {}

    for cid in present_ids:
        idx_c = id_to_idx[cid]
        mask_cid = yk_v == int(cid)
        samples = X_v[mask_cid]
        centroid = centroid_matrix[idx_c]

        max_disp, p95_disp = _dispersion(samples, centroid, metric)
        dist_nearest = _nearest_other(pw[idx_c])

        if sil is None:
            p5_sil, frac_at_risk = None, None
        else:
            sil_c = sil[mask_cid]
            sil_finite = sil_c[np.isfinite(sil_c)]
            if len(sil_finite) == 0:
                p5_sil, frac_at_risk = None, None
            else:
                p5_sil = float(np.percentile(sil_finite, 5))
                frac_at_risk = float(np.mean(sil_finite < 0))

        result[cid] = {
            "max_dispersion": max_disp,
            "p95_dispersion": p95_disp,
            "dist_to_nearest_centroid": dist_nearest,
            "p5_silhouette": p5_sil,
            "frac_at_risk": frac_at_risk,
        }

    return result
