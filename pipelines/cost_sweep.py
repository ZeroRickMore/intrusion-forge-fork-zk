"""Complexity cost analysis: a deterministic k-NN build cost model per cell, plus a
cross-run aggregator of the cost↔quality trade-off.

Three modes (dispatched in `main`):
  * default — fit T(m)=c·m^alpha by timing `build_knn_graph` over a controlled m grid on
    one cell's train set, and report the complexity share of end-to-end wall time.
    Writes `shared/cost_model.json`.
  * `aggregate=<root>` — scan finished runs under <root> and assemble a tidy table of
    (n_clusters, complexity build time, rho) from existing artifacts. Writes
    `<root>/cost_quality_table.json`.
  * `summary=<root>` — roll up the per-cell `cost_model.json` files under <root> into
    alpha (mean±std per distance, the Θ(m²) robustness check) plus the cosine/euclidean c
    ratio per dataset. Writes `<root>/cost_model_summary.json`.

The build is brute-force kNN = Θ(m²) distance evals, so the cap on complexity samples
decouples cost from corpus size N. After tying cluster count to the cap (cap // floor),
m tracks the cap directly, so the cost model is measured by subsampling m deliberately
(no per-cluster floor) rather than by sweeping a production clustering.
"""

import logging
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from omegaconf import OmegaConf

from src.core.config import load_config
from src.core.io import load_df
from src.core.utils import load_from_json, save_to_json
from src.domain.analysis.complexity.shared import build_knn_graph
from pipelines.common import paths_from_cfg

logger = logging.getLogger(__name__)

# Controlled m grid for the cost model; truncated to the cell's train size.
DEFAULT_M_GRID = [5000, 10000, 20000, 40000, 80000]
_N_REPEATS = 3

# Complexity-stage function names in a timing.json (graph build + measures).
_COMPLEXITY_FNS = {
    "build_knn_graph", "prepare_complexity_graph",
    "compute_f_measures", "compute_n_measures", "compute_network_measures",
    "compute_t_measures", "compute_cluster_geometry",
    "compute_complexity_from_graph", "compute_cluster_complexity",
    "compute_class_complexity",
}


def _fit_cost_model(points: list[tuple[int, float]]) -> dict:
    """Fit T(m) = c·m^alpha to (m, build_s) pairs in log-log space."""
    pts = {int(m): float(t) for m, t in points if m and t > 0}
    if len(pts) < 3:
        return {"alpha": None, "c": None, "r2": None, "n_points": len(pts)}
    m = np.array(sorted(pts), dtype=float)
    lm, lt = np.log(m), np.log([pts[int(x)] for x in m])
    alpha, log_c = np.polyfit(lm, lt, 1)
    ss_res = float(np.sum((lt - (alpha * lm + log_c)) ** 2))
    ss_tot = float(np.sum((lt - lt.mean()) ** 2))
    return {
        "alpha": float(alpha),
        "c": float(np.exp(log_c)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "n_points": len(pts),
    }


def _cost_units(m: int, k: int, d: int) -> dict:
    """Hardware-neutral k-NN build cost: m² distance evals over d features, m·k edges."""
    return {
        "distance_evals": m * (m - 1),
        "feature_ops": m * (m - 1) * d,
        "knn_edges": m * k,
    }


def _timing_rows(timing_path: Path) -> list[dict]:
    try:
        return load_from_json(timing_path)
    except (FileNotFoundError, OSError):
        return []


def _stage_seconds(timing_path: Path, exclude: set[str] | None = None) -> float:
    """Sum durations in a timing.json, skipping `exclude`d function names."""
    exclude = exclude or set()
    return float(
        sum(r.get("duration_s", 0.0) for r in _timing_rows(timing_path)
            if r.get("function") not in exclude)
    )


def _complexity_seconds(timing_path: Path) -> float:
    """Sum durations of the complexity-stage functions in a timing.json."""
    return float(
        sum(r.get("duration_s", 0.0) for r in _timing_rows(timing_path)
            if r.get("function") in _COMPLEXITY_FNS)
    )


# --------------------------------------------------------------------------- #
# Mode 1: per-cell cost model harness                                          #
# --------------------------------------------------------------------------- #
def _time_build(
    X_num: np.ndarray, X_cat: np.ndarray | None, m: int, k: int, metric: str,
    rng: np.random.Generator, repeats: int = _N_REPEATS,
) -> float:
    """Median wall time of `build_knn_graph` on a random m-row subsample."""
    n = X_num.shape[0]
    times: list[float] = []
    for _ in range(repeats):
        idx = rng.choice(n, size=m, replace=False)
        xs_cat = X_cat[idx] if X_cat is not None else None
        t0 = perf_counter()
        build_knn_graph(X_num[idx], xs_cat, k, metric=metric)
        times.append(perf_counter() - t0)
    return float(median(times))


def _pipeline_cost(paths, cost_model: dict, m_prod: int) -> dict:
    """Complexity build share of end-to-end wall time, at the production cap m_prod.

    The build time is the cost model's prediction c·m_prod^alpha; prep+classify come
    from the cell's own timing.json files.
    """
    prep_clustering_s = _stage_seconds(paths.shared / "timing.json", exclude=_COMPLEXITY_FNS)
    classify_s = _stage_seconds(paths.outputs / "timing.json")
    non_complexity_s = prep_clustering_s + classify_s
    alpha, c = cost_model.get("alpha"), cost_model.get("c")
    build_s = float(c * m_prod ** alpha) if alpha is not None and c is not None else None
    total = (build_s + non_complexity_s) if build_s is not None else None
    return {
        "prep_clustering_s": prep_clustering_s,
        "classify_s": classify_s,
        "non_complexity_s": non_complexity_s,
        "m_prod": m_prod,
        "complexity_build_s_pred": build_s,
        "complexity_share_pred": (build_s / total) if total else None,
    }


def _run_cost_model(cfg, paths) -> None:
    """Fit and persist the per-cell k-NN build cost model."""
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    train_df = load_df(str(paths.processed_data / f"train.{cfg.data.extension}"))
    n = len(train_df)
    X_num = (
        train_df[num_cols].to_numpy(dtype=np.float64)
        if num_cols else np.empty((n, 0))
    )
    X_cat = train_df[cat_cols].to_numpy() if cat_cols else None
    k = int(cfg.complexity.k)
    metric = cfg.complexity.distance
    d_feat = X_num.shape[1] + (X_cat.shape[1] if X_cat is not None else 0)

    grid = OmegaConf.select(cfg, "costmodel.m_grid")
    grid = [int(x) for x in grid] if grid else list(DEFAULT_M_GRID)
    m_grid = sorted({min(int(m), n) for m in grid})  # truncate to n, dedup

    logger.info(
        "Cost model on %s [%s], k=%d, n_train=%d, m_grid=%s",
        cfg.data.file_name, metric, k, n, m_grid,
    )
    rng = np.random.default_rng(cfg.seed)
    measurements: list[dict] = []
    for m in m_grid:
        build_s = _time_build(X_num, X_cat, m, k, metric, rng)
        measurements.append({"m": m, "build_s": build_s, "cost_units": _cost_units(m, k, d_feat)})
        logger.info("m=%d: build=%.3fs (median of %d)", m, build_s, _N_REPEATS)

    cost_model = _fit_cost_model([(r["m"], r["build_s"]) for r in measurements])
    m_prod = min(int(cfg.complexity.max_complexity_samples), n)
    out = {
        "dataset": cfg.data.file_name,
        "distance": metric,
        "k": k,
        "n_train": n,
        "cost_model": cost_model,
        "m_grid": measurements,
        "pipeline_cost": _pipeline_cost(paths, cost_model, m_prod),
    }
    save_to_json(out, paths.shared / "cost_model.json")
    logger.info(
        "Cost model written -> %s (alpha=%s, r2=%s)",
        paths.shared / "cost_model.json",
        cost_model.get("alpha"), cost_model.get("r2"),
    )


# --------------------------------------------------------------------------- #
# Mode 2: cross-run aggregator                                                 #
# --------------------------------------------------------------------------- #
def _aggregate_runs(root: Path) -> None:
    """Assemble a tidy cost↔quality table from finished runs under `root`.

    A run is any directory holding `shared/metadata/clusters_meta.json`. Per classifier
    sub-run it emits one row joining genuine cluster count, complexity build time and the
    failure-classifier rho. Incomplete runs are skipped.
    """
    rows: list[dict] = []
    for meta_path in sorted(root.glob("**/shared/metadata/clusters_meta.json")):
        run = meta_path.parents[2]
        try:
            meta = load_from_json(meta_path)
            noise = meta.get("noise_cluster_ids", []) or []
            n_genuine = len(meta.get("clusters_distribution", {})) - len(noise)
            build_s = _complexity_seconds(run / "shared/timing.json")
            cfg_c = load_from_json(run / "shared/config_composed.json")
            dataset = cfg_c.get("data", {}).get("file_name")
            distance = cfg_c.get("distance") or cfg_c.get("complexity", {}).get("distance")
            cap = cfg_c.get("complexity", {}).get("max_complexity_samples")
        except (FileNotFoundError, OSError, KeyError) as exc:
            logger.warning("skip %s (shared artifacts): %s", run, exc)
            continue

        for res_path in sorted(run.glob("*/outputs/analysis/classifier_results.json")):
            try:
                res = load_from_json(res_path)
            except (FileNotFoundError, OSError):
                continue
            rows.append({
                "run": str(run.relative_to(root)),
                "dataset": dataset,
                "distance": distance,
                "classifier": res_path.parents[2].name,
                "n_clusters_genuine": n_genuine,
                "n_clusters_used": res.get("n_clusters_used"),
                "max_complexity_samples": cap,
                "complexity_build_s": build_s,
                "rho": res.get("spearman"),
                "r2": res.get("r2"),
                "mae": res.get("mae"),
            })

    out_path = root / "cost_quality_table.json"
    save_to_json({"n_rows": len(rows), "rows": rows}, out_path)
    logger.info("Aggregated %d rows -> %s", len(rows), out_path)


def _summarize_cost_models(root: Path) -> None:
    """Roll up per-cell `cost_model.json` files under `root` into a robustness summary.

    Groups fits by (dataset, distance): reports alpha mean±std across seeds (the Θ(m²)
    check) and c mean±std, then the euclidean/cosine c ratio per dataset. Degenerate fits
    (alpha=None) are skipped. Writes `<root>/cost_model_summary.json`.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for cm_path in sorted(root.glob("**/shared/cost_model.json")):
        try:
            cm = load_from_json(cm_path)
        except (FileNotFoundError, OSError):
            continue
        model = cm.get("cost_model", {})
        if model.get("alpha") is None:
            logger.warning("skip %s (degenerate fit, alpha=None)", cm_path)
            continue
        key = (cm.get("dataset"), cm.get("distance"))
        groups.setdefault(key, []).append(model)

    by_distance: list[dict] = []
    c_by_dataset: dict[str, dict[str, float]] = {}
    for (dataset, distance), fits in sorted(
        groups.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")
    ):
        alphas = np.array([f["alpha"] for f in fits], dtype=float)
        cs = np.array([f["c"] for f in fits], dtype=float)
        r2s = [f["r2"] for f in fits if f.get("r2") is not None]
        row = {
            "dataset": dataset,
            "distance": distance,
            "n_seeds": len(fits),
            "alpha_mean": float(alphas.mean()),
            "alpha_std": float(alphas.std(ddof=1)) if len(fits) > 1 else 0.0,
            "c_mean": float(cs.mean()),
            "c_std": float(cs.std(ddof=1)) if len(fits) > 1 else 0.0,
            "r2_min": min(r2s) if r2s else None,
        }
        by_distance.append(row)
        c_by_dataset.setdefault(dataset, {})[distance] = row["c_mean"]

    c_ratio = [
        {"dataset": ds, "euclidean_over_cosine": d["euclidean"] / d["cosine"]}
        for ds, d in sorted(c_by_dataset.items())
        if d.get("cosine") and d.get("euclidean")
    ]

    n_models = sum(len(v) for v in groups.values())
    out_path = root / "cost_model_summary.json"
    save_to_json(
        {"n_models": n_models, "by_distance": by_distance, "c_ratio": c_ratio}, out_path
    )
    logger.info("Summarised %d cost models -> %s", n_models, out_path)


def main():
    """Dispatch: `aggregate=<root>` → cross-run table; `summary=<root>` → cost-model
    roll-up; else per-cell cost model."""
    argv = sys.argv[1:]
    agg = [a for a in argv if a.startswith("aggregate=")]
    if agg:
        _aggregate_runs(Path(agg[0].split("=", 1)[1]))
        return
    summ = [a for a in argv if a.startswith("summary=")]
    if summ:
        _summarize_cost_models(Path(summ[0].split("=", 1)[1]))
        return

    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=argv,
    )
    _run_cost_model(cfg, paths_from_cfg(cfg))


if __name__ == "__main__":
    main()
