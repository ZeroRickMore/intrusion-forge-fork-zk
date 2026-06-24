# TODO REMOVE
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

# TODO REMOVE
from pathlib import Path
import os

from src.core.config import load_config
from src.core.utils import load_from_joblib, load_from_json
from src.core.io import load_df, save_df






def top_k_predict(x, weak_clf, ext_clf, cluster_models, k):
    proba_weak = weak_clf.predict_proba(x)          # shape: (n_classes,)
    top_k_classes = np.argsort(proba_weak)[-k:]        # k candidate labels

    best_confidence = -np.inf
    best_prediction = None

    for c in top_k_classes:
        # Load model on demand for the class
        if isinstance(cluster_models[c], str): # If it's a string, it's the first time the class was requested, thus we load the model now
            cluster_models[c] = load_from_joblib(cluster_models[c])

        complexity = cluster_models[c].extract_complexity(x)
        x_ext = np.concat(x, complexity)
        proba_ext = ext_clf.predict_proba(x_ext)    # shape: (n_classes,)

        conf = max(proba_ext)
        if conf > best_confidence:
            best_confidence = conf
            best_prediction = np.argmax(proba_ext)

    return best_prediction, best_confidence

def run_topk_predict_on_inference_input_df(
        inference_df,
        weak_clf,
        ext_clf,
        cluster_models,
        k
    ):
    X = inference_df.to_numpy()

    predictions = []
    confidences = []

    for x in X:
        pred, conf = top_k_predict(
            x,
            weak_clf,
            ext_clf,
            cluster_models,
            k
        )

        predictions.append(pred)
        confidences.append(conf)

    inference_df["prediction"] = predictions
    inference_df["confidence"] = confidences

    return inference_df

def load_inference_input_df_and_strip_labels(cfg):
    # Find the input csv based on the used split fraction
    split_frac_of_trained_models = load_from_json(Path(cfg.path.shared) / "metadata" / "df_info.json").get('split_frac', None)
    if split_frac_of_trained_models in [None, 100.0]:
        raise ValueError(f"The models were trained on a bad split_frac: {split_frac_of_trained_models}, please re-run the entire pipeline with the correct 'prepare.topk_inference.split_frac' parameter.")

    inference_input_path = Path(cfg.path.dataset_split) / str(split_frac_of_trained_models) / f'inference_input.pkl'
    if not os.path.exists(inference_input_path):
        raise ValueError(f"[{inference_input_path}] does not exist. Cannot topk-infer with no input data.\n\tDid you run prepare_data with the correct 'topk_inference' parameters?")
    
    inference_df = load_df(file_path=inference_input_path)

    # Strip the label column and return it separated, we do not want to use it during inference!
    labels = inference_df.pop(cfg.data.label_col)

    return inference_df, labels

# topk_inference.py data=nb15_v2 name=topk_inference classifier=random_forest
def main():
    """Main entry point for topk inference over a stored inference_input dataset."""

    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:]# TODO REMOVE:# + ['clustering=kmeans', 'data=nb15_v2', 'name=topk_inference_kmeans', 'classifier=random_forest'],
    )

    k = cfg.topk_inference.k

    weak_clf = load_from_joblib(Path(cfg.path.models) / 'model.joblib') # classifier trained on raw features only
    ext_clf  = load_from_joblib(Path(cfg.path.models) / 'model_extended.joblib') # classifier trained on complexity-extended features

    cluster_models = load_from_json(file_path=Path(cfg.path.clustering_models) / 'class_to_model.json')[str(cfg.clustering.name)] # I am only interested in the current clustering algorithm

    inference_df, labels = load_inference_input_df_and_strip_labels(cfg)

    inference_df = run_topk_predict_on_inference_input_df(
        inference_df=inference_df,
        weak_clf = weak_clf,
        ext_clf = ext_clf,
        cluster_models = cluster_models,
        k = k
    )

    # Place original labels back for future assessments and quality check
    inference_df['original_label'] = labels

    save_df(df=inference_df, file_path=Path(cfg.path.topk_inference_out) / 'results_with_original_labels.pkl')


if __name__ == '__main__':
    main()