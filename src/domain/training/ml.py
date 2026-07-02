from pathlib import Path

import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.core.utils import load_from_joblib, save_to_joblib
from src.engine.ml.preprocessing import build_pipeline


def _check_context(context: dict | None) -> tuple[list[str], list[str]]:
    if context is None:
        raise ValueError(
            "ML training requires `context` with `num_cols` and `cat_cols`."
        )
    return context["num_cols"], context["cat_cols"]


def fit_classifier(
    name: str,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    context: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Build an sklearn pipeline (preprocess + classifier) and fit on (X, y).

    `X` carries both numerical and categorical columns. `X_val`/`y_val` are
    accepted for interface parity with DL but ignored. Returns ``(pipeline, {})``.
    """
    num_cols, cat_cols = _check_context(context)
    pipeline = build_pipeline(name, params, num_cols, cat_cols)
    pipeline.fit(X, y)
    return pipeline, {}


def grid_search_classifier(
    name: str,
    params: dict,
    grid: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    scoring: str = "f1_macro",
    cv: int = 5,
    n_jobs: int = -1,
    max_samples: int | None = None,
    context: dict | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """Cross-validated grid search over the classifier step of the pipeline.

    Grid keys are remapped to ``clf__<key>`` to target the classifier inside the
    Pipeline. Transform steps are cached across combinations (each fold
    preprocesses once); when ``max_samples`` is set and exceeded, the search runs
    on a stratified subsample and the winner is refit on the full data.
    Returns ``(best_pipeline, summary)``.
    """
    num_cols, cat_cols = _check_context(context)
    clf_grid = {f"clf__{k}": v for k, v in grid.items()}

    subsampled = max_samples is not None and len(X) > max_samples
    if subsampled:
        _, X_search, _, y_search = train_test_split(
            X, y, test_size=max_samples, stratify=y, random_state=random_state
        )
    else:
        X_search, y_search = X, y

    with tempfile.TemporaryDirectory() as cache_dir:
        base = build_pipeline(name, params, num_cols, cat_cols)
        base.memory = cache_dir
        search = GridSearchCV(
            base,
            param_grid=clf_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=not subsampled,
        )
        search.fit(X_search, y_search)

    if subsampled:
        best_clf_params = {
            k.replace("clf__", "", 1): v for k, v in search.best_params_.items()
        }
        best_pipeline = build_pipeline(
            name, {**params, **best_clf_params}, num_cols, cat_cols
        )
        best_pipeline.fit(X, y)
    else:
        best_pipeline = search.best_estimator_

    cv_results = [
        {
            "params": {k.replace("clf__", "", 1): v for k, v in p.items()},
            "mean_test_score": float(s),
            "std_test_score": float(std),
        }
        for p, s, std in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        )
    ]
    summary = {
        "best_params": {
            k.replace("clf__", "", 1): v for k, v in search.best_params_.items()
        },
        "best_score": float(search.best_score_),
        "scoring": scoring,
        "cv": cv,
        "cv_results": cv_results,
    }
    return best_pipeline, summary


def predict_with_proba(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    context: dict | None = None,
    return_embedding: bool = False,
) -> tuple:
    """Return ``(predicted labels, class probability matrix)`` for the pipeline.

    ML pipelines have no learned embedding, so ``return_embedding=True`` returns
    ``z=None`` as a third element — kept symmetric with the DL path so callers can
    request an embedding uniformly.
    """
    y_pred, y_proba = pipeline.predict(X), pipeline.predict_proba(X)
    if return_embedding:
        return y_pred, y_proba, None
    return y_pred, y_proba


def save_model(
    pipeline: Pipeline,
    path: Path,
    *,
    name: str = "",
    params: dict | None = None,
    suffix: str = "",
) -> None:
    """Save the full sklearn Pipeline to ``path / f'model{suffix}.joblib'``."""
    save_to_joblib(pipeline, Path(path) / f"model{suffix}.joblib")


def load_model(
    path: Path, *, context: dict | None = None, suffix: str = ""
) -> Pipeline:
    """Load the sklearn Pipeline from ``path / f'model{suffix}.joblib'``."""
    return load_from_joblib(Path(path) / f"model{suffix}.joblib")
