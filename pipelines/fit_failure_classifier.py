import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from tqdm import tqdm

from src.core.config import load_config, save_config, to_container
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from pipelines.common import paths_from_cfg
from src.core.utils import flush_timing, load_from_json, timed
from src.domain.analysis.selective_prediction import (
    selective_prediction_metrics,
    selective_recall_metrics,
)

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)


def _max_safe_splits(n_minority: int, n_splits_cfg: int) -> int:
    """Largest k <= n_splits_cfg such that StratifiedKFold(k) won't degenerate."""
    k = min(n_splits_cfg, n_minority)
    return k if k >= 2 else 0


def _run_outer_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    inner_cv: KFold | None,
    param_grid: dict,
    random_state: int,
) -> dict:
    """Run one outer CV fold. When inner_cv is None, skip GridSearchCV and use default RF."""
    if inner_cv is None:
        best = RandomForestRegressor(random_state=random_state)
        best.fit(X_train, y_train)
    else:
        grid = GridSearchCV(
            estimator=RandomForestRegressor(random_state=random_state),
            param_grid=param_grid,
            cv=inner_cv,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

    y_pred = best.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "importances": best.feature_importances_,
        "y_pred": y_pred.tolist(),
        "indices": X_test.index.tolist(),
    }


def _quantile_strata(y: pd.Series, q: int) -> pd.Series | None:
    """Integer quantile-bin codes of the continuous target for stratified folding
    (None if the target can't be binned into at least two groups)."""
    try:
        bins = pd.qcut(y, q=min(q, len(y)), duplicates="drop")
    except (ValueError, IndexError):
        return None
    if bins.nunique() < 2:
        return None
    return bins.cat.codes


def build_cluster_summary(
    complexity: dict,
    class_complexity: dict,
    predictions: dict,
) -> dict:
    """Merge per-cluster complexity with class-level complexity (joined on the
    cluster's class via `cluster_class`) and per-classifier failure rates.

    Output schema per cluster:
        cluster_<measure>  — cluster-level complexity (vs top-K adversarial clusters)
        class_<measure>    — class-level complexity of the cluster's class
        cluster_class, is_noise_cluster, n_test, failure_rate
    """
    cluster_errors = predictions.get("clusters", {}).get("global", {}) or {}
    summary: dict[str, dict] = {}
    for cid, cluster_measures in complexity.items():
        class_id = cluster_measures.get("cluster_class")
        class_measures = (
            class_complexity.get(str(class_id), {}) if class_id is not None else {}
        )
        cluster_feats = {
            f"cluster_{k}": v
            for k, v in cluster_measures.items()
            if k not in ("cluster_class", "is_noise_cluster")
        }
        class_feats = {
            f"class_{k}": v
            for k, v in class_measures.items()
            if k != "is_noise_cluster"
        }
        error_entry = cluster_errors.get(str(cid), {})
        summary[str(cid)] = {
            **cluster_feats,
            **class_feats,
            "cluster_class": class_id,
            "is_noise_cluster": int(cluster_measures.get("is_noise_cluster", False)),
            "n_test": error_entry.get("n_total", 0),
            "failure_rate": error_entry.get("error_rate"),
        }
    return summary


def _run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: StratifiedKFold | KFold,
    outer_k: int,
    split_labels: pd.Series | None,
    inner_cv: KFold | None,
    param_grid: dict,
    random_state: int,
) -> dict:
    """Run the outer CV loop and collect per-fold scores + out-of-fold predictions."""
    fold_r2s: list[float] = []
    fold_maes: list[float] = []
    fold_importances: list[np.ndarray] = []
    oof_y_true: list[float] = []
    oof_y_pred: list[float] = []
    oof_indices: list = []

    for train_idx, test_idx in tqdm(
        outer_cv.split(X, split_labels), total=outer_k, desc="Outer CV"
    ):
        fold = _run_outer_fold(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[test_idx],
            inner_cv,
            param_grid,
            random_state,
        )
        fold_r2s.append(fold["r2"])
        fold_maes.append(fold["mae"])
        fold_importances.append(fold["importances"])
        oof_y_true.extend(y.iloc[test_idx].tolist())
        oof_y_pred.extend(fold["y_pred"])
        oof_indices.extend(fold["indices"])

    return {
        "fold_r2s": fold_r2s,
        "fold_maes": fold_maes,
        "fold_importances": fold_importances,
        "y_true": np.array(oof_y_true),
        "y_pred": np.array(oof_y_pred),
        "indices": oof_indices,
    }


def _aggregate_oof_results(oof: dict, feature_cols: list[str]) -> dict:
    """Aggregate out-of-fold predictions into the published regression-metrics block."""
    y_true, y_pred = oof["y_true"], oof["y_pred"]
    mean_importances = np.mean(oof["fold_importances"], axis=0)
    # headline metric: rank correlation between predicted and observed failure rate
    rho = spearmanr(y_pred, y_true)

    return {
        "spearman": float(rho.statistic),
        "spearman_pvalue": float(rho.pvalue),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_std": float(np.nanstd(oof["fold_r2s"])),
        "r2_per_fold": oof["fold_r2s"],
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_std": float(np.std(oof["fold_maes"])),
        "mae_per_fold": oof["fold_maes"],
        "feature_importances": dict(zip(feature_cols, mean_importances.tolist())),
        "oof_predicted_rate": {
            str(cid): float(pred) for cid, pred in zip(oof["indices"], y_pred)
        },
    }


def _failure_rate_distribution(rates: pd.Series) -> dict:
    """Summary stats of the failure-rate distribution over the used clusters."""
    if rates.empty:
        return {}
    quantiles = rates.quantile([0.25, 0.5, 0.75, 0.9])
    return {
        "min": float(rates.min()),
        "p25": float(quantiles[0.25]),
        "median": float(quantiles[0.5]),
        "p75": float(quantiles[0.75]),
        "p90": float(quantiles[0.9]),
        "max": float(rates.max()),
        "mean": float(rates.mean()),
    }


@timed
def fit_failure_classifier(
    cluster_stats: dict,
    param_grid: dict,
    *,
    feature_cols: list[str] | None = None,
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    random_state: int = 42,
    min_test_support: int = 5,
    analysis_bus: LogDispatcher | None = None,
) -> dict:
    """Nested-CV Random Forest regressor predicting each cluster's failure rate
    from its separability features.

    Target is `failure_rate` (from `build_cluster_summary`). Clusters with no test
    samples or fewer than `min_test_support` are excluded (unreliable rate). Outer
    loop (StratifiedKFold on rate quantile bins, KFold fallback) gives unbiased OOF
    predictions; inner GridSearchCV tunes hyperparameters. Headline metric: Spearman ρ.
    """
    logger.info("Running failure classifier ...")
    df = pd.DataFrame.from_dict(cluster_stats, orient="index")

    # Noise pseudo-clusters have no genuine geometry, so they are excluded from the
    # meta-model; their test-support share is the coverage cost.
    is_noise = (
        df["is_noise_cluster"].fillna(0).astype(bool)
        if "is_noise_cluster" in df
        else pd.Series(False, index=df.index)
    )
    no_test = df["failure_rate"].isna()
    low_support = ~no_test & (df["n_test"].fillna(0) < min_test_support)
    n_excluded_no_test = int(no_test.sum())
    n_excluded_low_support = int((low_support & ~is_noise).sum())
    n_excluded_noise = int((is_noise & ~no_test).sum())
    noise_test_share = (
        float(df.loc[is_noise & ~no_test, "n_test"].fillna(0).sum())
        / float(df.loc[~no_test, "n_test"].fillna(0).sum())
        if df.loc[~no_test, "n_test"].fillna(0).sum()
        else 0.0
    )
    df = df[~no_test & ~low_support & ~is_noise]

    rates = df["failure_rate"].astype(float)
    n_test = df["n_test"].astype(float)
    global_error_rate = (
        float((rates * n_test).sum() / n_test.sum()) if n_test.sum() else 0.0
    )
    exclusions = {
        "n_clusters_total": int(no_test.size),
        "n_clusters_used": int(len(df)),
        "n_excluded_no_test": n_excluded_no_test,
        "n_excluded_low_support": n_excluded_low_support,
        "n_excluded_noise": n_excluded_noise,
        "noise_test_share": noise_test_share,
        "min_test_support": min_test_support,
        "global_error_rate": global_error_rate,
    }
    if n_excluded_no_test or n_excluded_low_support or n_excluded_noise:
        logger.info(
            "Excluded clusters — no test: %d, support < %d: %d, noise pseudo-clusters: %d "
            "(%.1f%% of test support); %d/%d used",
            n_excluded_no_test,
            min_test_support,
            n_excluded_low_support,
            n_excluded_noise,
            100.0 * noise_test_share,
            len(df),
            no_test.size,
        )

    if feature_cols is None:
        feature_cols = [
            c
            for c in df.select_dtypes("number").columns
            if c not in ("failure_rate", "n_test", "is_noise_cluster")
        ]
    X = df[feature_cols].copy()
    y = df["failure_rate"].astype(float)

    context_metrics = {
        "failure_rate_distribution": _failure_rate_distribution(rates),
    }
    n_used = len(df)
    if n_used < 2 or float(y.std()) < 1e-9:
        message = (
            f"Failure classifier skipped: {n_used} usable cluster(s), "
            f"failure-rate std={float(y.std()):.4g}. Need >=2 clusters with variance."
        )
        logger.warning("[STAGE-SKIP] %s", message)
        results = {
            "skipped": True,
            "reason": "degenerate_target",
            "message": message,
            **exclusions,
            **context_metrics,
        }
        if analysis_bus is not None:
            analysis_bus.publish(
                LogBundle.from_dict({"json/analysis/classifier_results": results})
            )
        return results

    # Stratify the outer folds on quantile bins of the rate so each fold stays
    # representative on small datasets; fall back to plain KFold when the rate
    # can't be binned into >=2 groups.
    strata = _quantile_strata(y, n_outer_splits)
    if strata is not None:
        outer_k = _max_safe_splits(int(strata.value_counts().min()), n_outer_splits)
    else:
        outer_k = 0
    if outer_k >= 2:
        outer_cv = StratifiedKFold(
            n_splits=outer_k, shuffle=True, random_state=random_state
        )
        split_labels = strata
    else:
        outer_k = _max_safe_splits(n_used, n_outer_splits)
        outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
        split_labels = None

    m_train_worst = n_used - math.ceil(n_used / outer_k)
    inner_k = _max_safe_splits(m_train_worst, n_inner_splits)
    if outer_k < n_outer_splits or inner_k < n_inner_splits:
        logger.warning(
            "[CV-ADAPT] Adapting CV (clusters=%d): outer %d→%d, inner %d→%d%s",
            n_used,
            n_outer_splits,
            outer_k,
            n_inner_splits,
            inner_k or 0,
            " (no GridSearchCV — using RF defaults)" if inner_k == 0 else "",
        )

    inner_cv = (
        KFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        if inner_k > 0
        else None
    )

    oof = _run_nested_cv(
        X, y, outer_cv, outer_k, split_labels, inner_cv, param_grid, random_state
    )

    results = {
        **exclusions,
        **context_metrics,
        **_aggregate_oof_results(oof, feature_cols),
    }
    # Selective prediction: reject high predicted-risk clusters → accuracy on the
    # retained set, from support-weighted OOF predictions.
    support = df.loc[oof["indices"], "n_test"].astype(float).to_numpy()
    cluster_class = df.loc[oof["indices"], "cluster_class"].to_numpy()
    results["risk_coverage"] = selective_prediction_metrics(
        oof["y_pred"], oof["y_true"], support
    )
    # Class-balanced view (macro-recall): exposes whether rejection sacrifices
    # minority classes, the blind spot of pooled accuracy.
    results["selective_macro_recall"] = selective_recall_metrics(
        oof["y_pred"], oof["y_true"], support, cluster_class
    )
    if analysis_bus is not None:
        analysis_bus.publish(
            LogBundle.from_dict({"json/analysis/classifier_results": results})
        )
    rc = results["risk_coverage"]
    logger.info(
        "Classifier results — Spearman: %.4f, R²: %.4f, MAE: %.4f; "
        "selective acc@%.0f%%: %.4f (random %.4f, oracle benefit recovered %.2f)",
        results["spearman"],
        results["r2"],
        results["mae"],
        100 * rc.get("coverage_target", 0.8),
        rc.get("acc_at_target_predictor", float("nan")),
        rc.get("global_accuracy", float("nan")),
        rc.get("oracle_benefit_recovered", float("nan")),
    )
    return results


def main():
    """Main entry point for failure-classifier training (per-classifier stage)."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )
    paths = paths_from_cfg(cfg)
    save_config(cfg, paths.configs / "config_composed.json")

    complexity_path = paths.shared / "complexity.json"
    class_complexity_path = paths.shared / "class_complexity.json"
    for p in (complexity_path, class_complexity_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing complexity artifact at {p}. "
                "Run `make complexity` first."
            )
    complexity = load_from_json(complexity_path)
    class_complexity = load_from_json(class_complexity_path)
    predictions = load_from_json(paths.outputs / "analysis/predictions/test.json")

    cluster_summary = build_cluster_summary(
        complexity,
        class_complexity,
        predictions,
    )

    bus = LogDispatcher()
    bus.subscribe(JSONSubscriber(paths.outputs))
    bus.publish(LogBundle.from_dict({"json/analysis/cluster_summary": cluster_summary}))
    logger.info("Cluster summary published.")

    fit_failure_classifier(
        cluster_summary,
        to_container(cfg.failure_classifier.param_grid),
        n_outer_splits=cfg.failure_classifier.n_outer_splits,
        n_inner_splits=cfg.failure_classifier.n_inner_splits,
        min_test_support=cfg.failure_classifier.min_test_support,
        random_state=cfg.seed,
        analysis_bus=bus,
    )

    flush_timing(paths.outputs / "timing.json")


if __name__ == "__main__":
    main()
