import sys
import time
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

# sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config
from src.core.utils import load_from_joblib, load_from_json
from src.core.io import load_df, save_df
from collections import Counter


def top_k_predict(
    x,
    weak_clf,
    ext_clf,
    cluster_models,
    k,
):
    """
    Predict using top-k candidate classes from the weak classifier
    and select the prediction with highest confidence from the
    complexity-extended classifier.
    """

    proba_weak = weak_clf.predict_proba(x.reshape(1, -1))[0]
    top_k_classes = np.argsort(proba_weak)[-k:]

    best_confidence = -np.inf
    best_prediction = None

    for c in tqdm(top_k_classes, desc='Classes', position=1):
        # Lazy-load model
        if isinstance(cluster_models[c], str):
            cluster_models[c] = load_from_joblib(cluster_models[c])

        complexity = cluster_models[c].extract_complexity(x)

        x_ext = np.concatenate([np.asarray(x).ravel(), np.asarray(complexity).ravel(),])

        proba_ext = ext_clf.predict_proba(x_ext.reshape(1, -1))[0]

        conf = np.max(proba_ext)
        pred = np.argmax(proba_ext)


        if conf > best_confidence:
            best_confidence = conf
            best_prediction = pred


    return best_prediction, best_confidence


def run_topk_predict_on_inference_input_df(
    inference_df,
    weak_clf,
    ext_clf,
    cluster_models,
    k
):
    X = inference_df.to_numpy()

    print(f"- Running topk-inference on {len(X):,} samples with k={k}.")

    predictions = []
    confidences = []

    for x in tqdm(X, desc=f"Top-{k} inference", position=0):
        pred, conf = top_k_predict(
            x=x,
            weak_clf=weak_clf,
            ext_clf=ext_clf,
            cluster_models=cluster_models,
            k=k,
        )

        predictions.append(pred)
        confidences.append(conf)

    inference_df["prediction"] = predictions
    inference_df["confidence"] = confidences

    print(f"- Finished topk-inference.")

    # TODO REMOVE
    print("- Prediction distribution:")
    for cls, count in Counter(predictions).most_common():
        print(f"\tclass={cls:<5} || count={count}")

    return inference_df


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

    print(f"- Loading clustering models mapping for clustering algorithm \"{cfg.clustering.name}\"")

    cluster_models = load_from_json(file_path=Path(cfg.path.clustering_models) / "class_to_model.json")[str(cfg.clustering.name)]

    inference_df, labels = load_inference_input_df_and_strip_labels(cfg)

    inference_df = run_topk_predict_on_inference_input_df(
                    inference_df=inference_df,
                    weak_clf=weak_clf,
                    ext_clf=ext_clf,
                    cluster_models=cluster_models,
                    k=k
                )

    print("- Re-attaching original labels...")

    inference_df["original_label"] = labels

    output_path = Path(cfg.path.topk_inference_out) / "results_with_original_labels.pkl"

    print(f"- Saving results to {output_path}")

    save_df(df=inference_df, file_path=output_path)

    print("- Topk-inference completed.")



if __name__ == "__main__":
    main()