import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json

from src.core.config import load_config
from src.core.utils import load_from_joblib, load_from_json
from src.core.io import load_df, save_df
from collections import Counter

def top_k_predict(
    x,
    weak_clf,
    ext_clf,
    cluster_models,
    label_mapping,
    complexity_features_per_class,
    k,
):
    """
    Predict using top-k candidate classes from the weak classifier
    and select the prediction with highest confidence from the
    complexity-extended classifier.
    """

    proba_weak = weak_clf.predict_proba(x)[0]
    top_k_class_numerics = np.argsort(proba_weak)[-k:]

    best_confidence = -np.inf
    best_prediction = None

    for class_numeric in top_k_class_numerics:
        class_label = label_mapping[str(class_numeric)]
        # Lazy-load model
        if isinstance(cluster_models[class_label], str):
            cluster_models[class_label] = load_from_joblib(cluster_models[class_label])

        # First extend it, then align it with the order of the features as they appear in the ext_clf
        x_ext = get_extended_sample(sample=x, complexity_extension=complexity_features_per_class[str(class_numeric)])[ext_clf.named_steps["clf"].feature_names_in_]

        proba_ext = ext_clf.predict_proba(x_ext)

        conf = np.max(proba_ext)
        pred = np.argmax(proba_ext)

        if conf > best_confidence:
            best_confidence = conf
            best_prediction = pred


    return best_prediction, best_confidence

def extend_features_batch(df, complexity_extension : dict, column_ordering):
    """
    complexity_extension: dict like {feature_name: value}

    follows the specified column ordering for insertion 
    """

    df_ext = df.copy()

    for column_name, column_values in complexity_extension.items():
        df_ext[column_name] = column_values

    return df_ext[column_ordering]

def run_topk_predict_on_inference_input_df(
    inference_df,
    weak_clf,
    ext_clf,
    cluster_models,
    label_mapping,
    complexity_features_per_class,
    k
):
    
    print(f"- Running topk-inference on {inference_df.shape[0]:,} samples with k={k}.")
    
    weak_proba = weak_clf.predict_proba(inference_df) # Find weak proba of all samples
    top_k_classes = np.argsort(weak_proba, axis=1)[:, -k:] # Find the top-k classes of each sample's weak_proba

    class_to_indices = defaultdict(list) # dict that maps, for each class (numeric class, not label), the index of the samples that are candidate for that class
    for i in range(inference_df.shape[0]):
        for cls in top_k_classes[i]:
            class_to_indices[cls].append(i)

    # Find the best pred and confidence by iterating over each sample per class, storing the last "best" in its index in an array of zeros
    best_predictions = np.empty(inference_df.shape[0], dtype=object) # Each has the class label and not the class index
    best_confidences = np.zeros(inference_df.shape[0])

    what_is_happening = {} # TODO REMOVE

    for cls, idxs in tqdm(class_to_indices.items(), total=len(class_to_indices), desc="Classes"):
        X = inference_df.iloc[idxs].copy()
        X_ext = extend_features_batch(X, complexity_features_per_class[str(cls)], ext_clf.named_steps["clf"].feature_names_in_)

        per_class_ext_proba = ext_clf.predict_proba(X_ext)

        what_is_happening[label_mapping[str(cls)]] = [str(_) for _ in per_class_ext_proba[:5]] # TODO REMOVE

        per_class_preds = np.argmax(per_class_ext_proba, axis=1)
        per_class_confs = np.max(per_class_ext_proba, axis=1)

        for sample_index_in_the_class, sample_index_in_df in enumerate(idxs):
            if per_class_confs[sample_index_in_the_class] > best_confidences[sample_index_in_df]: # If the current confidence is better than the old confidence
                # Update global dataframe infos of the sample
                best_predictions[sample_index_in_df] = label_mapping[str(per_class_preds[sample_index_in_the_class])]
                best_confidences[sample_index_in_df] = per_class_confs[sample_index_in_the_class]

    print(json.dumps(what_is_happening, indent=4)) # TODO REMOVE
    # input("Does this make sense?") # TODO REMOVE
    
    return best_predictions, best_confidences

def load_inference_input_df_and_strip_labels(cfg):

    print("- Loading inference input dataset...")

    split_frac_of_trained_models = load_from_json(Path(cfg.path.shared) / "metadata" / "df_info.json").get("split_frac", None)

    print(f"- Training split fraction: {split_frac_of_trained_models}")

    if split_frac_of_trained_models in [None, 100.0]:
        raise ValueError(
            f"- The models were trained on a bad split_frac: {split_frac_of_trained_models},\n"
            f"  please re-run the entire pipeline with a reasonable\n"
            f"  'prepare.topk_inference.split_frac' parameter.\n"
        )
    
    if split_frac_of_trained_models != cfg.prepare.topk_inference.split_frac:
        raise ValueError(f"- The models were trained on split_frac={split_frac_of_trained_models},\n"
                         f"  but your current configuration's split_frac is {cfg.prepare.topk_inference.split_frac}.\n"
                         f"\t- Changing config's split_frac to {split_frac_of_trained_models} will solve this error,\n"
                          "\tbut make sure this is intended before doing so, as running inference on a model trained\n"
                          "\ton a different split_frac may result in data leak and false predictions.")

    inference_input_path = (Path(cfg.path.dataset_split) / "inference_input.pkl")

    print(f"- Loading df from inference input path:\n\t{inference_input_path}")

    if not os.path.exists(inference_input_path):
        raise ValueError(
            f"- [{inference_input_path}] does not exist: cannot topk-infer with no input data.\n"
            f"\t- You should run prepare_data with \"prepare.topk_inference.split_dataset : True\"."
        )

    inference_df = load_df(file_path=inference_input_path)

    print(f"- Loaded inference dataset with shape {inference_df.shape}")

    labels = inference_df.pop(cfg.data.label_col) # Remove labels to avoid information leak on predictions

    print(f"- Removed label column '{cfg.data.label_col}' from inference features.")

    return inference_df, labels

def print_diagnostics(inference_df_full, confidences, predictions, labels):
    print("\n=== Inference Diagnostics ===")

    n_samples = len(inference_df_full)

    print(f"Samples processed: {n_samples:,}")

    print(
        f"Confidence stats:"
        f"\n\tMean: {confidences.mean():.4f}"
        f"\n\tMin : {confidences.min():.4f}"
        f"\n\tMax : {confidences.max():.4f}"
    )

    pred_counts = Counter(predictions)

    print("\nTop predicted classes:")
    for cls, count in pred_counts.most_common(10):
        pct = 100 * count / n_samples
        print(f"\tClass {cls}: {count} samples ({pct:5.2f}%)")

    # Accuracy (only if labels use same encoding as predictions)
    try:
        accuracy = np.mean(predictions == labels.to_numpy())
        print(f"\nPrediction accuracy: {accuracy:.4%}")
    except Exception as e:
        print(f"\nCould not compute accuracy: {e}")

    print("============================\n")


def main():
    """
    Main entry point for top-k inference over a stored dataset.
    """

    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:]
    )

    k = cfg.topk_inference.k

    weak_clf_path = Path(cfg.path.models) / 'model.joblib'
    ext_clf_path  = Path(cfg.path.models) / "model_extended.joblib"

    print("- Loading models...")
    print(f"\t- Weak classifier path:\n\t{weak_clf_path}")
    print(f"\t- Extended classifier path:\n\t{ext_clf_path}")

    weak_clf = load_from_joblib(weak_clf_path)
    ext_clf = load_from_joblib(ext_clf_path)

    print(f"- Loading clustering models mapping for clustering algorithm \"{cfg.clustering.name}\", and label mappings for the entire dataset.")

    cluster_models = load_from_json(file_path=Path(cfg.path.clustering_models) / "class_to_model.json")[str(cfg.clustering.name)]
    for class_ in cluster_models:
        cluster_models[class_] = load_from_joblib(cluster_models[class_])

    label_mapping = load_from_json(Path(cfg.path.shared) / "metadata" / "df_meta.json")["label_mapping"]
    complexity_features_per_class = load_from_json(Path(cfg.path.shared) / "class_complexity.json")

    inference_df_full, labels = load_inference_input_df_and_strip_labels(cfg)

    # Remove features the clf was not trained on
    expected_features_weak_clf = list(weak_clf.named_steps["clf"].feature_names_in_)
    # expected_features_ext_clf  = set(ext_clf.named_steps["clf"].feature_names_in_)
    # if expected_features_weak_clf != expected_features_ext_clf:
    #     raise ValueError(f"- The two datasets seem to have been trained on a different feature set!\n"
    #                      f"\t- Weak: {expected_features_weak_clf}\n"
    #                      f"\t- Ext : {expected_features_ext_clf}\n\n"
    #                      "- Cannot proceed...")
    missing_features = set(expected_features_weak_clf) - set(inference_df_full.columns)
    if missing_features:
        raise ValueError(f"- Inference dataset missing required features that the models were trained on:\n{missing_features}.\n\n- Cannot proceed...")
    inference_df = inference_df_full[expected_features_weak_clf]

    # Find predictions and confidences, and append to df as columns
    predictions, confidences = run_topk_predict_on_inference_input_df(
                    inference_df=inference_df,
                    weak_clf=weak_clf,
                    ext_clf=ext_clf,
                    cluster_models=cluster_models,
                    label_mapping=label_mapping,
                    complexity_features_per_class=complexity_features_per_class,
                    k=k
                )
    del inference_df

    inference_df_full["best_prediction"] = predictions
    inference_df_full["best_confidence"] = confidences

    print("- Re-attaching original labels...")

    inference_df_full["original_label"] = labels

    output_path = Path(cfg.path.topk_inference_out) / "results_with_original_labels.pkl"

    print(f"- Saving results to {output_path}")

    save_df(df=inference_df_full, file_path=output_path)

    print_diagnostics(inference_df_full=inference_df_full, confidences=confidences, predictions=predictions, labels=labels)

    print("- Topk-inference completed.")



if __name__ == "__main__":
    main()