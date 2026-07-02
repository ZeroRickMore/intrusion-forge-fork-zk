import logging
import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.core.config import load_config, save_config
from src.core.log import (
    JSONSubscriber,
    LogBundle,
    LogDispatcher,
    setup_logger,
)
from src.core.utils import flush_timing, skip_if_exists, timed, save_to_joblib, load_from_json, save_to_json


from src.domain.analysis.metadata import (
    compute_clusters_metadata,
    compute_df_metadata,
    get_df_info,
)
from src.core.io import load_df, save_df
from sklearn.preprocessing import RobustScaler

from src.domain.data.preprocessing import (
    LogTransformer,
    TopNHashEncoder,
    build_preprocessor,
    drop_nans,
    encode_labels,
    ml_split,
    query_filter,
    rare_category_filter,
    representative_split
)
from src.domain.analysis.complexity.shared import _l2_normalize
from src.domain.clustering import build_cluster_fn
from src.domain.clustering.base import (
    assign_clusters_within_class,
    assign_nearest_centroid,
    cluster_size_balance,
)

setup_logger(log_file="resources/logs.txt")
logger = logging.getLogger(__name__)


def _absorb_small_clusters(
    labels: np.ndarray, floor: int
) -> tuple[np.ndarray, int, int]:
    """Reassign clusters smaller than `floor` to noise (-1)."""
    ids, counts = np.unique(labels[labels != -1], return_counts=True)
    small = ids[counts < floor]
    if small.size == 0:
        return labels, 0, 0
    mask = np.isin(labels, small)
    return np.where(mask, -1, labels), int(small.size), int(mask.sum())


def _cluster_per_class(
    X_num: np.ndarray,
    y_class: np.ndarray,
    classes: list,
    *,
    X_cat: np.ndarray | None = None,
    algorithms: dict[str, dict],
    max_fit_samples: int,
    random_state: int,
    metric: str = "euclidean",
    min_cluster_floor: int = 50,
    max_clusters_total: int | None = None,
    grid_target_cluster_size: int | None = None,
    resolution_weight: float = 0.1,
    save_clustering_models : bool,
    clustering_models_base_path : Path,
    clustering_algorithm_name: str,
) -> tuple[np.ndarray, dict[int, np.ndarray], set[int], dict[str, dict]]:
    """Per-class clustering. Returns (labels, centroids, noise_cluster_ids, report).

    Cluster IDs are globally unique via offset. Residual -1 noise points (and
    clusters absorbed by `min_cluster_floor`) are reassigned to per-class
    pseudo-clusters; their IDs are collected in noise_cluster_ids.

    `max_clusters_total` caps the total number of genuine clusters so the complexity
    subsample floor never exceeds the cap (= max_complexity_samples // floor). It is
    split equally across classes; None leaves the count driven by the data-relative grid.
    """
    n = X_num.shape[0]
    max_clusters_per_class = (
        max(2, max_clusters_total // len(classes))
        if max_clusters_total is not None
        else None
    )
    labels = np.full(n, -1, dtype=np.int64)
    centroids: dict[int, np.ndarray] = {}
    offset = 0
    report: dict[str, dict] = {}

    if save_clustering_models:
        path_to_clustering_class_to_model_json = Path(clustering_models_base_path) / 'class_to_model.json'
        class_to_cluster_model = load_from_json(file_path=path_to_clustering_class_to_model_json) if os.path.exists(path_to_clustering_class_to_model_json) else {}
        class_to_cluster_model[clustering_algorithm_name] = {}

    for current_class in tqdm(classes, desc="Clustering classes"):
        mask = y_class == current_class
        if not mask.any():
            continue
        X_num_cls = X_num[mask]
        X_num_cls = _l2_normalize(X_num_cls) if metric == "cosine" else X_num_cls
        X_cat_cls = X_cat[mask] if X_cat is not None else None

        algo_reports: dict[str, dict] = {}
        cluster_fn = build_cluster_fn(
            algorithms=algorithms,
            max_fit_samples=max_fit_samples,
            random_state=random_state,
            reporter=algo_reports.__setitem__,
            metric=metric,
            max_clusters=max_clusters_per_class,
            grid_target_cluster_size=grid_target_cluster_size,
            resolution_weight=resolution_weight,
        )
        raw_labels, clustering_model = cluster_fn(X_num_cls, X_cat_cls)

        # Save the clustering models if required
        if save_clustering_models:
            # Single algorithm model storing (no ensemble)
            path_to_current_class_model_joblib = Path(clustering_models_base_path) / str(str(current_class) + '___' + clustering_algorithm_name + '.joblib')
            save_to_joblib(data=clustering_model, file_path=path_to_current_class_model_joblib)
            class_to_cluster_model[clustering_algorithm_name][str(current_class)] = str(path_to_current_class_model_joblib)

        raw_labels, n_floor_clusters, n_floor_points = _absorb_small_clusters(
            raw_labels, min_cluster_floor
        )

        n_cls = int(raw_labels.shape[0])
        n_noise_cls = int((raw_labels == -1).sum())
        report[str(current_class)] = {
            "n_samples": n_cls,
            "algorithms": algo_reports,
            "summary": {
                "n_clusters": int(np.unique(raw_labels[raw_labels != -1]).size),
                "n_noise": n_noise_cls,
                "noise_ratio": n_noise_cls / n_cls if n_cls > 0 else 0.0,
                "size_balance": cluster_size_balance(raw_labels),
                "floor_absorbed_clusters": n_floor_clusters,
                "floor_absorbed_points": n_floor_points,
            },
        }

        cluster_ids = np.unique(raw_labels[raw_labels != -1])
        labels[mask] = np.where(raw_labels == -1, -1, raw_labels + offset)
        # centroids on the raw (un-normalized) features; materialize once per class
        X_raw_cls = X_num[mask]
        for cid in cluster_ids:
            centroids[int(cid + offset)] = X_raw_cls[raw_labels == cid].mean(axis=0)
        del X_raw_cls
        if len(cluster_ids) > 0:
            offset += int(cluster_ids.max()) + 1

    # reassign noise points (-1) to per-class pseudo-clusters
    noise_cluster_ids: set[int] = set()
    noise_count = int((labels == -1).sum())
    if noise_count > 0:
        next_id = max(centroids.keys(), default=-1) + 1
        for noise_cls in sorted(np.unique(y_class)):
            noise_mask = (y_class == noise_cls) & (labels == -1)
            if noise_mask.any():
                labels[noise_mask] = next_id
                centroids[next_id] = X_num[noise_mask].mean(axis=0)
                noise_cluster_ids.add(next_id)
                next_id += 1

    # Save the json information about the classes to the models used for them
    if save_clustering_models:
        save_to_json(data=class_to_cluster_model, file_path=path_to_clustering_class_to_model_json)

    return labels, centroids, noise_cluster_ids, report

def _split_dataset_points(
        df,
        dataset_split_path : str,
        split_frac : float,
        random_state : int | None = None,
        label_col: str | None = None,
        force: bool = False
    ):
    """Splits the dataset in two, and saves the results into two csv files.  
    Returns the dataframe built on the split_frac, so if split_frac=0.7 the later prepare_data pipeline will be executed on the 0.7 dataset."""
    dataset_split_path = Path(dataset_split_path)
    os.makedirs(dataset_split_path, exist_ok=True)

    prepared_data_output_path  = dataset_split_path / f'trained_on.pkl'
    inference_data_output_path = dataset_split_path / f'inference_input.pkl'

    if not force and os.path.exists(prepared_data_output_path) and os.path.exists(inference_data_output_path):
        return load_df(prepared_data_output_path)

    prepared_data_df, inference_data_df = representative_split(df, split_frac, random_state, label_col)

    save_df(df=prepared_data_df, file_path=prepared_data_output_path)
    save_df(df=inference_data_df,file_path=inference_data_output_path)

    return prepared_data_df


@timed
def preprocess_df(
    df,
    num_cols,
    cat_cols,
    label_col,
    filter_query,
    min_cat_count,
    train_frac,
    val_frac,
    test_frac,
    random_state,
    top_n,
    hash_buckets,
    split_dataset,
    split_frac,
    force,
    dataset_split_path
):
    """Preprocess dataframe: filter, encode, scale, and split."""
    logger.info(
        "Preprocessing: %d rows, %d num_cols, %d cat_cols",
        len(df),
        len(num_cols),
        len(cat_cols),
    )
    df = drop_nans(df, num_cols + cat_cols + [label_col])
    df = query_filter(df, query=filter_query)
    df = rare_category_filter(df, [label_col], min_count=min_cat_count)

    # Handle topk_inference dataset split (must be done AFTER data filtering steps such as drop_nans, query_filter, rare_category_filter!)
    if split_dataset:
        df = _split_dataset_points(
            df=df,
            dataset_split_path=dataset_split_path,
            split_frac=split_frac,
            random_state=random_state,
            label_col=label_col,
            force=force
        )

    train_df, val_df, test_df = ml_split(
        df,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
        label_col=label_col,
    )
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    preprocessor = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_steps=[
            ("log_transformer", LogTransformer()),
            ("scaler", RobustScaler()),
        ],
        cat_steps=[
            ("top_n_encoder", TopNHashEncoder(top_n=top_n, hash_buckets=hash_buckets)),
        ],
    )
    logger.info("Preprocessor: %s", preprocessor)
    preprocessor.fit(train_df)
    train_df, val_df, test_df = (
        preprocessor.transform(split) for split in [train_df, val_df, test_df]
    )

    return train_df, val_df, test_df


def _cluster_splits(
    cfg,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
    dispatcher: LogDispatcher,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, set[int]]:
    """Cluster train per class, then attach the `cluster` column to all splits.

    Train is labelled by the per-class clusterer; val/test points are assigned
    inductively to the nearest train centroid within their own class. Returns the
    splits, centroids and pseudo-cluster ids; publishes the clustering report.
    """
    X_num = train_df[num_cols].to_numpy(dtype=np.float64)
    X_cat = train_df[cat_cols].to_numpy() if cat_cols else None
    y_class = train_df[label_col].to_numpy()
    all_classes = sorted(train_df[label_col].unique().tolist())

    logger.info("Running per-class clustering on train (n=%d)...", len(train_df))
    algorithms = OmegaConf.to_container(cfg.clustering.algorithms, resolve=True)
    # Cap genuine clusters so the complexity subsample floor never exceeds the cap
    # (floor·n_clusters ≤ max_complexity_samples). Ties cluster count to the complexity
    # budget; split equally across classes inside _cluster_per_class.
    max_clusters_total = (
        cfg.complexity.max_complexity_samples // cfg.complexity.min_subsample_per_cluster
    )
    labels, centroids, noise_cluster_ids, clustering_report = _cluster_per_class(
        X_num,
        y_class,
        all_classes,
        X_cat=X_cat,
        algorithms=algorithms,
        max_fit_samples=cfg.clustering.max_fit_samples,
        random_state=cfg.seed,
        metric=cfg.clustering.distance,
        min_cluster_floor=cfg.clustering.min_cluster_floor,
        max_clusters_total=max_clusters_total,
        grid_target_cluster_size=cfg.clustering.grid_target_cluster_size,
        resolution_weight=cfg.clustering.resolution_weight,
        save_clustering_models = cfg.prepare.topk_inference.save_clustering_models,
        clustering_models_base_path = cfg.path.clustering_models,
        clustering_algorithm_name=str(cfg.clustering.name)
    )
    dispatcher.publish(
        LogBundle.from_dict({"json/clustering_report": clustering_report})
    )

    # cluster → class map from train (each cluster is single-class by construction)
    cluster_to_class = {
        int(cid): y_class[labels == cid][0] for cid in np.unique(labels)
    }

    train_df = train_df.copy()
    train_df["cluster"] = labels
    assigned: dict[str, pd.DataFrame] = {}
    for name, split_df in (("val", val_df), ("test", test_df)):
        split_df = split_df.copy()
        if cfg.label_free_assignment:
            split_df["cluster"] = assign_nearest_centroid(
                split_df[num_cols].to_numpy(dtype=np.float64),
                centroids,
                metric=cfg.clustering.distance,
            )
        else:
            split_df["cluster"] = assign_clusters_within_class(
                split_df[num_cols].to_numpy(dtype=np.float64),
                split_df[label_col].to_numpy(),
                centroids,
                cluster_to_class,
                metric=cfg.clustering.distance,
            )
        assigned[name] = split_df
    val_df, test_df = assigned["val"], assigned["test"]

    noise_ids = sorted(noise_cluster_ids)
    noise_count = (
        sum(
            int(np.isin(df["cluster"], noise_ids).sum())
            for df in (train_df, val_df, test_df)
        )
        if noise_ids
        else 0
    )
    logger.info(
        "Clustering complete — %d clusters (noise reassigned: %d points into pseudo-clusters)",
        len(centroids),
        noise_count,
    )
    return train_df, val_df, test_df, centroids, noise_cluster_ids


def _publish_metadata(
    cfg,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str,
    label_mapping: dict,
    centroids: dict,
    noise_cluster_ids: set[int],
    dispatcher: LogDispatcher,
) -> dict:
    """Compute and publish dataset + cluster metadata; returns df_meta."""
    logger.info("Computing and saving metadata...")
    metadata = compute_df_metadata(
        {"train": train_df, "val": val_df, "test": test_df},
        label_col,
        num_cols,
        cat_cols,
        cfg.data.benign_tag,
        label_mapping=label_mapping,
    )
    dispatcher.publish(LogBundle.from_dict({"json/df_meta": metadata}))

    clusters_metadata = compute_clusters_metadata(
        train_df,
        val_df,
        test_df,
        label_col,
        cluster_col="cluster",
        centroids={str(k): v.tolist() for k, v in centroids.items()},
        noise_cluster_ids=sorted(noise_cluster_ids),
    )
    dispatcher.publish(LogBundle.from_dict({"json/clusters_meta": clusters_metadata}))
    logger.info("Cluster metadata saved.")
    return metadata


@timed
def prepare(cfg):
    """Prepare data given a configuration object."""
    num_cols = list(cfg.data.num_cols) if cfg.data.num_cols else []
    cat_cols = list(cfg.data.cat_cols) if cfg.data.cat_cols else []
    label_col = cfg.data.label_col

    raw_data_path = Path(cfg.path.raw_data)
    processed_data_path = Path(cfg.path.processed_data)
    data_logs_path = Path(cfg.path.shared)

    dispatcher = LogDispatcher()
    dispatcher.subscribe(JSONSubscriber(data_logs_path / "metadata"))

    logger.info("Loading and preprocessing data...")
    df = load_df(str(raw_data_path))

    # out = (
    #     df[label_col]
    #     .value_counts()
    #     .rename_axis("label")
    #     .reset_index(name="count")
    # )
    # out["percentage"] = out["count"] / out["count"].sum() * 100
    # print(out)

    logger.info("Raw data loaded: %d rows, %d columns", *df.shape)

    df_info = get_df_info(df, label_col=label_col, split_frac=cfg.prepare.topk_inference.split_frac)
    dispatcher.publish(LogBundle.from_dict({"json/df_info": df_info}))

    train_df, val_df, test_df = preprocess_df(
        df=df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_col=label_col,
        filter_query=cfg.data.filter_query,
        min_cat_count=cfg.data.min_cat_count,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        random_state=cfg.seed,
        top_n=cfg.data.top_n,
        hash_buckets=cfg.data.hash_buckets,
        split_dataset=cfg.prepare.topk_inference.split_dataset, # split_dataset bool
        split_frac=cfg.prepare.topk_inference.split_frac, # split_frac
        force=cfg.prepare.force, # force
        dataset_split_path=cfg.path.dataset_split, # dataset_split_path
    )

    train_df, val_df, test_df = (
        df.reset_index(drop=True) for df in [train_df, val_df, test_df]
    )

    train_df, val_df, test_df, centroids, noise_cluster_ids = _cluster_splits(
        cfg, train_df, val_df, test_df, num_cols, cat_cols, label_col, dispatcher
    )

    train_df, val_df, test_df, label_mapping = encode_labels(
        train_df, val_df, test_df, label_col, dst_label_col=f"encoded_{label_col}"
    )

    logger.info("Saving processed data...")
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        save_df(split_df, processed_data_path / f"{split_name}.{cfg.data.extension}")

    metadata = _publish_metadata(
        cfg,
        train_df,
        val_df,
        test_df,
        num_cols,
        cat_cols,
        label_col,
        label_mapping,
        centroids,
        noise_cluster_ids,
        dispatcher,
    )
    return train_df, val_df, test_df, metadata


def main():
    """Main entry point for data preparation."""
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    ext = cfg.data.extension
    processed = Path(cfg.path.processed_data)
    shared = Path(cfg.path.shared)
    markers = [processed / f"{s}.{ext}" for s in ("train", "val", "test")]
    markers.append(shared / "metadata/clusters_meta.json")
    if skip_if_exists(markers, cfg.prepare.force, "prepare"):
        return

    save_config(cfg, shared / "config_composed.json")
    prepare(cfg)
    flush_timing(shared / "timing.json")


if __name__ == "__main__":
    main()
