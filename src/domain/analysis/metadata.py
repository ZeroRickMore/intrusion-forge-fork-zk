import numpy as np
import pandas as pd


def get_df_info(df: pd.DataFrame, *, label_col: str | None = None, split_frac : float = 1) -> dict:
    """Return basic information about a DataFrame."""
    info = {
        "split_frac" : split_frac,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
        "memory_usage": int(df.memory_usage(deep=True).sum()),
        "feature_info": {
            col: {
                "dtype": str(df[col].dtype),
                "unique_count": int(df[col].nunique()),
            }
            for col in df.columns
        },
    }
    if label_col and label_col in df.columns:
        info["label_distribution"] = df[label_col].value_counts().to_dict()
    return info


def compute_df_metadata(
    splits: dict[str, pd.DataFrame],
    label_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    benign_tag: str,
    *,
    label_mapping: dict | None = None,
    weights_key: str | None = None,
) -> dict:
    """Compute metadata dictionary for one or more named DataFrames.

    Args:
        splits: Mapping of ``tag → DataFrame`` (e.g. ``{"train": …, "val": …}``).
        weights_key: Tag of the split used for class-weight computation.
            Defaults to ``"train"`` if present, otherwise the first key.
    """
    if not splits:
        raise ValueError("splits must contain at least one DataFrame.")

    if weights_key is None:
        weights_key = "train" if "train" in splits else next(iter(splits))
    ref_df = splits[weights_key]

    class_counts = ref_df[label_col].value_counts().sort_index()
    class_weights = len(ref_df) / (len(class_counts) * class_counts)
    class_weights = np.log1p(class_weights) / np.log1p(class_weights).max()

    return {
        "label_mapping": label_mapping or {},
        "dataset_sizes": {tag: len(df) for tag, df in splits.items()},
        "samples_per_class": {
            tag: df[label_col].value_counts().to_dict() for tag, df in splits.items()
        },
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "benign_tag": benign_tag,
        "num_classes": ref_df[label_col].nunique(),
        "class_weights": class_weights.tolist(),
    }


def compute_clusters_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    cluster_col: str,
    centroids: dict,
    *,
    noise_cluster_ids: list[int] | None = None,
) -> dict:
    """Aggregate cluster metadata across all splits.

    Returns {class_to_clusters, clusters_distribution, centroids,
    noise_cluster_ids}.
    """
    df_ = pd.concat([train_df, val_df, test_df], ignore_index=True)
    encoded_label_col = f"encoded_{label_col}"

    class_to_clusters = {
        str(cls): [
            str(v) for v in df_[df_[encoded_label_col] == cls][cluster_col].unique()
        ]
        for cls in df_[encoded_label_col].unique()
    }

    clusters_distribution = {
        str(k): v for k, v in df_[cluster_col].value_counts().to_dict().items()
    }

    return {
        "class_to_clusters": class_to_clusters,
        "clusters_distribution": clusters_distribution,
        "centroids": {str(k): v for k, v in centroids.items()},
        "noise_cluster_ids": sorted(noise_cluster_ids) if noise_cluster_ids else [],
    }
