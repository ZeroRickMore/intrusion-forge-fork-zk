import numpy as np


def risk_coverage_curve(
    score: np.ndarray,
    failure_rate: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage vs accuracy as clusters are admitted in ascending `score` order.

    Selection is per cluster (the deployable unit). Both axes are support-weighted,
    so coverage is the fraction of test traffic retained:
        coverage = Σ support(admitted) / Σ support
        accuracy = 1 − Σ(failure_rate·support, admitted) / Σ(support, admitted)
    Returns empty arrays when there is nothing to rank.
    """
    score = np.asarray(score, dtype=float)
    failure_rate = np.asarray(failure_rate, dtype=float)
    support = np.asarray(support, dtype=float)
    if score.size == 0 or support.sum() <= 0:
        return np.array([]), np.array([])

    order = np.argsort(score, kind="stable")
    s = support[order]
    correct = (1.0 - failure_rate[order]) * s
    cum_support = np.cumsum(s)
    coverage = cum_support / cum_support[-1]
    accuracy = np.cumsum(correct) / cum_support
    return coverage, accuracy


def macro_recall_curve(
    score: np.ndarray,
    failure_rate: np.ndarray,
    support: np.ndarray,
    cluster_class: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage vs macro-averaged recall on the retained set (ascending `score`).

    Each cluster is intra-class, so its `1 − failure_rate` is its class's recall
    contribution. Recall is pooled per class over admitted clusters, then averaged
    over the classes still present — the class-balanced counterpart of
    `risk_coverage_curve`. Returns empty arrays when there is nothing to rank.
    """
    score = np.asarray(score, dtype=float)
    failure_rate = np.asarray(failure_rate, dtype=float)
    support = np.asarray(support, dtype=float)
    cluster_class = np.asarray(cluster_class)
    if score.size == 0 or support.sum() <= 0:
        return np.array([]), np.array([])

    order = np.argsort(score, kind="stable")
    s = support[order]
    correct = (1.0 - failure_rate[order]) * s
    cls = cluster_class[order]

    coverage = np.cumsum(s) / s.sum()
    classes = np.unique(cls)
    cum_correct = {c: np.cumsum(np.where(cls == c, correct, 0.0)) for c in classes}
    cum_support = {c: np.cumsum(np.where(cls == c, s, 0.0)) for c in classes}

    macro_recall = np.empty(coverage.size, dtype=float)
    for i in range(coverage.size):
        recalls = [
            cum_correct[c][i] / cum_support[c][i]
            for c in classes
            if cum_support[c][i] > 0
        ]
        macro_recall[i] = float(np.mean(recalls)) if recalls else np.nan
    return coverage, macro_recall


def _curve_summary(
    cov_predictor: np.ndarray,
    val_predictor: np.ndarray,
    cov_oracle: np.ndarray,
    val_oracle: np.ndarray,
    baseline: float,
    coverage_target: float,
) -> dict:
    """AURC over coverage ∈ [0, 1] (np.interp clamps the low-coverage tail to the
    best-cluster value) plus each curve's value at `coverage_target`. Shared by the
    accuracy and macro-recall summaries; `baseline` is the flat Random value."""
    grid = np.linspace(0.0, 1.0, 101)
    return {
        "aurc_predictor": float(np.trapezoid(np.interp(grid, cov_predictor, val_predictor), grid)),
        "aurc_oracle": float(np.trapezoid(np.interp(grid, cov_oracle, val_oracle), grid)),
        "aurc_random": baseline,
        "at_target_predictor": float(np.interp(coverage_target, cov_predictor, val_predictor)),
        "at_target_oracle": float(np.interp(coverage_target, cov_oracle, val_oracle)),
    }


def _recovered(value_predictor: float, value_oracle: float, baseline: float) -> float:
    """Fraction of the oracle's gain over Random that the predictor captures.

    NaN when the oracle leaves no positive headroom — which on the macro-recall
    axis legitimately happens, since the oracle ranks by error (not recall) and
    need not dominate there.
    """
    gain = value_oracle - baseline
    return float((value_predictor - baseline) / gain) if gain > 1e-9 else float("nan")


def selective_prediction_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    support: np.ndarray,
    *,
    coverage_target: float = 0.8,
) -> dict:
    """Scalar risk–coverage summary (pooled accuracy) for the failure regressor.

    Three rankings of the same (actual rate, support): predictor (by predicted
    rate, label-free), oracle (by true rate, best achievable), random (flat at
    global accuracy). `oracle_benefit_recovered` is the fraction of the oracle's
    gain at `coverage_target` the predictor captures. Returns {} without support.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    support = np.asarray(support, dtype=float)
    total = support.sum()
    if predicted.size == 0 or total <= 0:
        return {}

    global_accuracy = float(1.0 - (actual * support).sum() / total)
    cov_p, acc_p = risk_coverage_curve(predicted, actual, support)
    cov_o, acc_o = risk_coverage_curve(actual, actual, support)
    s = _curve_summary(cov_p, acc_p, cov_o, acc_o, global_accuracy, coverage_target)

    return {
        "coverage_target": coverage_target,
        "global_accuracy": global_accuracy,
        "aurc_predictor": s["aurc_predictor"],
        "aurc_oracle": s["aurc_oracle"],
        "aurc_random": global_accuracy,
        "acc_at_target_predictor": s["at_target_predictor"],
        "acc_at_target_oracle": s["at_target_oracle"],
        "lift_over_random": s["at_target_predictor"] - global_accuracy,
        "oracle_benefit_recovered": _recovered(
            s["at_target_predictor"], s["at_target_oracle"], global_accuracy
        ),
    }


def selective_recall_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    support: np.ndarray,
    cluster_class: np.ndarray,
    *,
    coverage_target: float = 0.8,
) -> dict:
    """Class-balanced counterpart of `selective_prediction_metrics`: macro-recall
    on the retained set instead of pooled accuracy.

    The Oracle still ranks by true *error* rate (accuracy-optimal), so on this axis
    it need not dominate — a dip below the Random baseline signals that error-greedy
    rejection costs class balance. Returns ``{}`` when there is no usable support.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    support = np.asarray(support, dtype=float)
    cluster_class = np.asarray(cluster_class)
    if predicted.size == 0 or support.sum() <= 0:
        return {}

    cov_p, rec_p = macro_recall_curve(predicted, actual, support, cluster_class)
    cov_o, rec_o = macro_recall_curve(actual, actual, support, cluster_class)
    global_macro_recall = float(rec_p[-1])  # full retention = whole-set macro recall
    s = _curve_summary(cov_p, rec_p, cov_o, rec_o, global_macro_recall, coverage_target)

    return {
        "coverage_target": coverage_target,
        "global_macro_recall": global_macro_recall,
        "aurc_predictor": s["aurc_predictor"],
        "aurc_oracle": s["aurc_oracle"],
        "aurc_random": global_macro_recall,
        "recall_at_target_predictor": s["at_target_predictor"],
        "recall_at_target_oracle": s["at_target_oracle"],
        "lift_over_random": s["at_target_predictor"] - global_macro_recall,
        "oracle_benefit_recovered": _recovered(
            s["at_target_predictor"], s["at_target_oracle"], global_macro_recall
        ),
    }
