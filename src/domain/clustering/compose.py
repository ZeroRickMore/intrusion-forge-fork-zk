from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from src.domain.clustering import ClusteringFactory
from src.domain.clustering.base import ClusterFn, grid_search, make_hybrid_silhouette_fn
from src.domain.clustering.ensemble import make_ensemble_cluster_fn

if TYPE_CHECKING:
    from src.domain.clustering.ensemble import ConsensusReporter

Reporter = Callable[[str, dict], None]  # (algo_name, grid_search_result) -> None


def _split_grid_fixed(params: dict) -> tuple[dict, dict]:
    """Split params into ({list → grid}, {scalar → fixed})."""
    grid = {k: v for k, v in params.items() if isinstance(v, list)}
    fixed = {k: v for k, v in params.items() if not isinstance(v, list)}
    return grid, fixed


_K_GRID_KEYS = ("n_clusters", "n_components")


def _prune_grid_by_floor(
    grid: dict[str, list], effective_n: int, floor: int
) -> tuple[dict[str, list], dict[str, list]]:
    """Drop K candidates whose average cluster size would fall below `floor`.

    Only applies to cluster-count keys (`n_clusters`, `n_components`); the
    smallest candidate is always kept so the grid never empties. Returns
    `(pruned_grid, {key: dropped_values})`.
    """
    out = dict(grid)
    dropped: dict[str, list] = {}
    for key in _K_GRID_KEYS:
        values = out.get(key)
        if not values:
            continue
        keep = [k for k in values if effective_n / k >= floor]
        if not keep:
            keep = [min(values)]
        if len(keep) < len(values):
            out[key] = keep
            dropped[key] = [k for k in values if k not in keep]
    return out, dropped


# algorithms clustering on mixed features: scored with the Gower-hybrid silhouette
_HYBRID_SCORED_ALGOS = ("kprototypes",)


def _make_single_cluster_fn(
    name: str,
    params: dict,
    max_fit_samples: int,
    random_state: int,
    reporter: Reporter | None = None,
    min_cluster_floor: int = 50,
    metric: str = "cosine",
) -> ClusterFn:
    """Build a ClusterFn for a single registered algorithm."""
    fit_fn = ClusteringFactory.get(name)
    grid_raw, fixed = _split_grid_fixed(params)
    silhouette_fn = (
        make_hybrid_silhouette_fn(metric=metric, random_state=random_state)
        if name in _HYBRID_SCORED_ALGOS
        else None
    )

    def _fn(X_num: np.ndarray, X_cat: np.ndarray | None = None) -> np.ndarray:
        effective_n = min(X_num.shape[0], max_fit_samples)
        grid, pruned = _prune_grid_by_floor(grid_raw, effective_n, min_cluster_floor)
        common = {
            "max_fit_samples": max_fit_samples,
            "random_state": random_state,
            **fixed,
        }
        try:
            if grid:
                result, best_model = grid_search(
                    X_num,
                    X_cat,
                    fit_fn,
                    grid,
                    min_cluster_floor=min_cluster_floor,
                    silhouette_fn=silhouette_fn,
                    **common,
                )
                if pruned:
                    result["pruned_by_floor"] = pruned
                if reporter is not None:
                    reporter(name, result)
                return fit_fn(X_num, X_cat=X_cat, **result["best"]["combo"], **common)
            return fit_fn(X_num, X_cat=X_cat, **common)
        except Exception as e:
            # Graceful degradation: this algorithm abstains from the ensemble vote.
            # Co-association (noise = no-vote, variable denominator) handles -1 cleanly,
            # so the consensus continues with M-1 effective voters.
            if reporter is not None:
                reporter(name, {"best": None, "sweep": [], "error": f"{type(e).__name__}: {e}"})
            return np.full(X_num.shape[0], -1, dtype=np.int64)

    return _fn


def build_cluster_fn(
    algorithms: dict[str, dict],
    consensus_threshold: float,
    max_fit_samples: int,
    random_state: int,
    min_consensus_size: int = 1,
    reporter: Reporter | None = None,
    consensus_reporter: "ConsensusReporter | None" = None,
    propagation_confidence_floor: float = 0.0,
    weight_voters: bool = True,
    refine_geometry: bool = True,
    refine_margin: float = 0.8,
    min_cluster_floor: int = 50,
    metric: str = "cosine",
) -> ClusterFn:
    """Build a ClusterFn from {algorithm_name: params}; ensembles when >1 key.

    `min_consensus_size` is the absolute HDBSCAN(precomputed) min_cluster_size
    on the co-association matrix; ignored for a single algorithm. `reporter`
    is invoked once per algorithm with its grid-search result;
    `consensus_reporter` receives the consensus diagnostics on each ensemble
    call. `weight_voters`, `refine_geometry` and `refine_margin` configure the
    ensemble's voter weighting and feature-space refinement.
    """
    if not algorithms:
        raise ValueError("build_cluster_fn: algorithms is empty")
    fns = [
        _make_single_cluster_fn(
            name,
            params,
            max_fit_samples,
            random_state,
            reporter=reporter,
            min_cluster_floor=min_cluster_floor,
            metric=metric,
        )
        for name, params in algorithms.items()
    ]
    if len(fns) == 1:
        return fns[0]
    return make_ensemble_cluster_fn(
        fns,
        threshold=consensus_threshold,
        min_consensus_size=min_consensus_size,
        max_fit_samples=max_fit_samples,
        random_state=random_state,
        consensus_reporter=consensus_reporter,
        propagation_confidence_floor=propagation_confidence_floor,
        weight_voters=weight_voters,
        refine_geometry=refine_geometry,
        refine_margin=refine_margin,
        algorithms=algorithms,
    )
