import numpy as np
import hdbscan
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import Birch, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from src.domain.clustering import ClusteringFactory
from src.domain.clustering.base import _subsample

# out-of-sample prediction batch: bounds transient memory (hdbscan's
# approximate_predict allocates n × 2·min_samples distance/index arrays)
_PREDICT_CHUNK = 200_000


@ClusteringFactory.register("hdbscan")
def fit_hdbscan(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_method: str = "leaf",
    cluster_selection_epsilon: float = 0.0,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    **fixed_params,
) -> np.ndarray:
    """Fit HDBSCAN with Euclidean distance and return labels (n,).

    Noise (-1) is returned as-is: residual noise points are reassigned to
    per-class pseudo-clusters downstream and excluded from the failure meta-model.
    Combo quality (including the coverage/noise trade-off) is judged by the
    noise-penalised silhouette in `grid_search`, not by a hard acceptance gate.
    """
    n = X_num.shape[0]

    clf = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
        prediction_data=True,
    )

    if n > max_fit_samples:
        sub_num, _ = _subsample(X_num, None, max_fit_samples, random_state)
        clf.fit(sub_num)
        labels = np.concatenate(
            [
                hdbscan.approximate_predict(clf, X_num[start : start + _PREDICT_CHUNK])[0]
                for start in range(0, n, _PREDICT_CHUNK)
            ]
        )
    else:
        clf.fit(X_num)
        labels = clf.labels_

    return labels, clf


@ClusteringFactory.register("kprototypes")
def fit_kprototypes(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    gamma: float | None = None,
    max_fit_samples: int = 20_000,
    random_state: int = 0,
    **_,
) -> np.ndarray:
    """Fit k-prototypes on mixed numeric + categorical features and return labels (n,).

    `gamma` weighs the categorical matching dissimilarity against the numeric
    distance (None = kmodes default, half the mean std of the numerics).
    Requires categorical columns; use kmeans for numeric-only data.
    """
    if X_cat is None or X_cat.shape[1] == 0:
        raise ValueError(
            "fit_kprototypes requires categorical features (X_cat is empty); "
            "use kmeans for numeric-only data."
        )
    n, d_num = X_num.shape
    n_clusters = max(2, min(int(n_clusters), n - 1))
    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    cat_idx = list(range(d_num, d_num + X_cat.shape[1]))

    def _mixed(num: np.ndarray, cat: np.ndarray) -> np.ndarray:
        return np.concatenate([num.astype(object), cat.astype(object)], axis=1)

    model = KPrototypes(
        n_clusters=n_clusters,
        gamma=gamma,
        init="Cao",
        n_init=1,
        random_state=random_state,
    )
    if n > max_fit_samples:
        sub_num, sub_cat = _subsample(X_num, X_cat, max_fit_samples, random_state)
        model.fit(_mixed(sub_num, sub_cat), categorical=cat_idx)
        labels = np.concatenate(
            [
                model.predict(
                    _mixed(
                        X_num[start : start + _PREDICT_CHUNK],
                        X_cat[start : start + _PREDICT_CHUNK],
                    ),
                    categorical=cat_idx,
                )
                for start in range(0, n, _PREDICT_CHUNK)
            ]
        )
    else:
        labels = model.fit_predict(_mixed(X_num, X_cat), categorical=cat_idx)
    return np.asarray(labels, dtype=np.int64), model


@ClusteringFactory.register("kmeans")
def fit_kmeans(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    random_state: int = 0,
    **_,
) -> np.ndarray:
    """Fit K-means on X and return labels (n,)."""
    n_clusters = max(2, min(n_clusters, X_num.shape[0] - 1))
    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X_num)
    return labels, model


@ClusteringFactory.register("gmm")
def fit_gmm(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_components: int = 4,
    covariance_type: str = "full",
    random_state: int = 0,
    **_,
) -> np.ndarray:
    """Fit GMM on X and return predicted cluster labels (n,)."""
    n_components = max(2, min(n_components, X_num.shape[0] - 1))
    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    labels = model.fit_predict(X_num)
    return labels, model


@ClusteringFactory.register("birch")
def fit_birch(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    threshold: float = 0.5,
    branching_factor: int = 50,
    max_fit_samples: int = 50_000,
    random_state: int = 0,
    **_,
) -> np.ndarray:
    """Fit BIRCH with `n_clusters` (AgglomerativeClustering on CF-tree leaves)."""
    n = X_num.shape[0]
    n_clusters = max(2, min(int(n_clusters), n - 1))
    X_num = np.ascontiguousarray(X_num, dtype=np.float64)
    clf = Birch(
        threshold=threshold,
        branching_factor=branching_factor,
        n_clusters=n_clusters,
    )
    if n > max_fit_samples:
        sub_num, _sub = _subsample(X_num, None, max_fit_samples, random_state)
        clf.fit(sub_num)
        labels = clf.predict(X_num)
    else:
        clf.fit(X_num)
        labels = clf.labels_
    return labels, clf


@ClusteringFactory.register("spectral")
def fit_spectral(
    X_num: np.ndarray,
    *,
    X_cat: np.ndarray | None = None,
    n_clusters: int = 8,
    affinity: str = "rbf",
    gamma: float | None = None,
    n_neighbors: int = 10,
    max_fit_samples: int = 10_000,
    random_state: int = 0,
    **_,
) -> np.ndarray:
    """Spectral clustering; subsample + 1-NN propagation in feature space for n > max_fit_samples."""
    n = X_num.shape[0]
    n_clusters = max(2, min(int(n_clusters), n - 1))
    X_num = np.ascontiguousarray(X_num, dtype=np.float64)

    spec_kwargs = dict(
        n_clusters=n_clusters,
        affinity=affinity,
        assign_labels="kmeans",
        random_state=random_state,
        eigen_solver="arpack",
    )
    if gamma is not None and affinity == "rbf":
        spec_kwargs["gamma"] = float(gamma)
    if affinity == "nearest_neighbors":
        spec_kwargs["n_neighbors"] = int(n_neighbors)

    if n <= max_fit_samples:
        return SpectralClustering(**spec_kwargs).fit_predict(X_num)

    sub_num, _ = _subsample(X_num, None, max_fit_samples, random_state)
    sub_labels = SpectralClustering(**spec_kwargs).fit_predict(sub_num)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(sub_num)
    _, idx = nn.kneighbors(X_num, n_neighbors=1, return_distance=True)
    return sub_labels[idx.ravel()], "This is a placeholder for the Spectral model."

    # TODO HOW TO RETURN THE MODEL HERE?
