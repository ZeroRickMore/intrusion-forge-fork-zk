from collections.abc import Callable

import numpy as np

from src.domain.clustering import ClusteringFactory
from src.domain.clustering.base import ClusterFn, grid_search, make_hybrid_silhouette_fn

Reporter = Callable[[str, dict], None]  # (algo_name, grid_search_result) -> None


def _split_grid_fixed(params: dict) -> tuple[dict, dict]:
    """Split params into ({list → grid}, {scalar → fixed})."""
    grid = {k: v for k, v in params.items() if isinstance(v, list)}
    fixed = {k: v for k, v in params.items() if not isinstance(v, list)}
    return grid, fixed


# algorithms clustering on mixed features: scored with the Gower-hybrid silhouette
_HYBRID_SCORED_ALGOS = ("kprototypes",)

# algorithms parameterised by an explicit cluster count: their `n_clusters` grid
# is derived per class from the target average cluster size (see _n_clusters_grid)
# instead of a static, dataset-blind list.
_N_CLUSTERS_ALGOS = ("kmeans", "spectral", "birch", "kprototypes")


def _n_clusters_grid(
    n_class: int, target_size: int, k_cap: int, levels: int = 7
) -> list[int]:
    """Data-relative `n_clusters` candidates for a class of `n_class` points.

    Spans a geometric band of average cluster sizes from `target_size` (finest)
    upward by powers of two, so the finest candidate always targets ~target_size
    points/cluster regardless of class size — large classes are no longer capped
    at a static ceiling. Each candidate is clamped to [2, k_cap] (k_cap is the
    subsample-support floor) and de-duplicated.
    """
    sizes = (target_size * (2**i) for i in range(levels))
    ks = {min(k_cap, max(2, round(n_class / s))) for s in sizes}
    return sorted(ks)


def _make_single_cluster_fn(
    name: str,
    params: dict,
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter | None = None,
    metric: str = "cosine",
    max_clusters: int | None = None,
    grid_target_cluster_size: int | None = None,
    resolution_weight: float = 0.1,
) -> ClusterFn:
    """Build a ClusterFn for a single registered algorithm."""
    fit_fn = ClusteringFactory.get(name)
    # An algorithm with no explicit params (e.g. kmeans, whose n_clusters grid is
    # derived per class) parses from YAML as None.
    grid, fixed = _split_grid_fixed(params or {})
    silhouette_fn = (
        make_hybrid_silhouette_fn(metric=metric, random_state=random_state)
        if name in _HYBRID_SCORED_ALGOS
        else None
    )

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None = None) -> np.ndarray:
        common = {
            "max_fit_samples": max_fit_samples,
            "random_state": random_state,
            **fixed,
        }
        algo_grid = dict(grid)
        if name in _N_CLUSTERS_ALGOS and grid_target_cluster_size:
            # Per-class n_clusters band, sized so the finest candidate targets
            # ~grid_target_cluster_size points/cluster. Capped so each cluster
            # keeps ≥25 subsample points (a meaningful silhouette) and, when
            # `max_clusters` is set, so the total cluster count stays within the
            # complexity budget. Replaces any static n_clusters list from config.
            k_cap = max(2, max_fit_samples // 25)
            if max_clusters is not None:
                k_cap = min(k_cap, max_clusters)
            algo_grid["n_clusters"] = _n_clusters_grid(
                X_num.shape[0], grid_target_cluster_size, k_cap
            )
        if algo_grid:
            result, best_model = grid_search(
                X_num,
                X_cat,
                fit_fn,
                algo_grid,
                resolution_weight=resolution_weight,
                silhouette_fn=silhouette_fn,
                **common,
            )
            if reporter is not None:
                reporter(name, result)
            return fit_fn(X_num, X_cat=X_cat, **result["best"]["combo"], **common)
        return fit_fn(X_num, X_cat=X_cat, **common)

    return _fn


def build_cluster_fn(
    algorithms: dict[str, dict],
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter | None = None,
    metric: str = "cosine",
    max_clusters: int | None = None,
    grid_target_cluster_size: int | None = None,
    resolution_weight: float = 0.1,
) -> ClusterFn:
    """Build a ClusterFn from a single {algorithm_name: params} entry.

    `reporter` is invoked once with the algorithm's grid-search result. Exactly
    one algorithm is supported; degenerate K values are handled post-hoc by the
    small-cluster absorption in the pipeline, not by a pre-grid prune.
    """
    if len(algorithms) != 1:
        raise ValueError(
            f"build_cluster_fn expects exactly one algorithm, got {len(algorithms)}: "
            f"{list(algorithms)}"
        )
    (name, params), = algorithms.items()
    return _make_single_cluster_fn(
        name,
        params,
        max_fit_samples,
        random_state,
        reporter=reporter,
        metric=metric,
        max_clusters=max_clusters,
        grid_target_cluster_size=grid_target_cluster_size,
        resolution_weight=resolution_weight,
    )
