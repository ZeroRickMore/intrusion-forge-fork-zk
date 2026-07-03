import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json
from scipy.spatial.distance import cdist

from src.core.config import load_config
from src.core.utils import load_from_joblib, load_from_json
from src.core.io import load_df, save_df
from collections import Counter

from src.domain.analysis.complexity.shared import (
    l2_normalize,
    hybrid_row_batch,
    hybrid_row_batch_euclidean
)

# ROW PER ROW VERSION. WAY TOO SLOW, BUT USED AS A STARTING POINT
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
    def get_extended_sample(sample, complexity_extension):
        new_sample = sample.copy()
        if not isinstance(complexity_extension, dict):
            raise TypeError(f"complexity_extension should be a dict, exactly what's inside of class_complexity.json per class.")
        
        for label in complexity_extension:
            new_sample[label] = complexity_extension[label]

        return new_sample

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

def extend_features_batch(df, complexity_extension_class : dict, complexity_extension_cluster_class : dict, column_ordering):
    """
    - df : the subset of samples that require the extension
    - complexity_extension_class: dict like {feature_name: value}, the extension columns with values for the class, equal for each sample  
    - complexity_extension_cluster_class: dict like {feature_name: value}, the extension columns with values for the cluster class of the class (subclass), equal for each sample
    - column_ordering: the list of the columns that appear in the classifier, which will be followed during the extension
    """
    # TODO qui devo fare la cosa che ci appiccico la roba specifica del cluster predictato.
    # quindi qui devo usare il modello di clustering per la predizione da cluster_models

    # occhio che quasi sicuramente cluster_model ti dice la sottoclasse di appartenenza, che se sei fortunato è una di quelle che sta dentro complexity.json
    # quindi la roba per l'estensione la prendi da li
    df_ext = df.copy()

    for column_name, column_values in complexity_extension_class.items():
        df_ext[f"{column_name}"] = column_values

    # TODO UNCOMMENT THIS AFTER INCLUDING CLUSTER EXTENSION IN TRANINING
    # for column_name, column_values in complexity_extension_cluster_class.items():
    #     df_ext[f"cluster_class_{column_name}"] = column_values

    return df_ext[column_ordering]

def find_top_h_nearest_clusters(
    h,
    df_of_class,
    num_cols,
    cat_cols,
    class_label_int,
    class_to_clusters,
    cluster_centroids,
    distance_metric="euclidean",
):

    centroids_to_check = {
        cluster_id: centroid
        for cluster_id, centroid in cluster_centroids.items()
        if cluster_id in class_to_clusters[str(class_label_int)]
    } # Dictionary {"278" : [feat1, feat2, ...]} with all the clusters of the specified class


    cluster_ids = np.array(list(centroids_to_check.keys())) # the cluster class ids

    centroid_matrix = np.array(
        [centroids_to_check[cid] for cid in cluster_ids],
        dtype=float
    ) # array of the sole centroid features



    # Find distances depending on metric
    if distance_metric == "cosine":
        X_num_norm = l2_normalize(df_of_class[num_cols].to_numpy(dtype=float))
        # X_cat = df_of_class[cat_cols].to_numpy() if cat_cols else None # TODO implement cat columns too somehow
        centroid_matrix = l2_normalize(centroid_matrix)
        euclid = cdist(X_num_norm, centroid_matrix, metric="euclidean")
        dists = np.clip(euclid**2 / 2, 0.0, 1.0)

    elif distance_metric == "euclidean":
        X_num = df_of_class[num_cols].to_numpy(dtype=float)
        feat_ranges = X_num.max(axis=0) - X_num.min(axis=0)
        dist = np.zeros((X_num.shape[0], centroid_matrix.shape[0]),dtype=np.float64,)
        for f in range(X_num.shape[1]):
            r = max(float(feat_ranges[f]), 1e-8)
            dist += (np.abs(X_num[:, f:f+1] - centroid_matrix[:, f]) / r)
        dists = dist / X_num.shape[1]

    else:
        raise ValueError(
            f"Unsupported metric '{distance_metric}'. "
            "Expected 'cosine' or 'euclidean'."
        )

    h = min(h, len(cluster_ids))

    part = np.argpartition(dists, h - 1, axis=1,)[:, :h]

    part_d = np.take_along_axis( dists, part, axis=1,)

    order = np.argsort(part_d,axis=1,)

    closest_distances = np.take_along_axis(part_d, order, axis=1,)
    closest_clusters = cluster_ids[np.take_along_axis(part, order, axis=1,)]
    
    return closest_clusters, closest_distances

def run_topk_predict_on_inference_input_df(
    inference_df,
    num_cols,
    cat_cols,
    weak_clf,
    ext_clf,
    class_to_clusters,
    cluster_centroids,
    label_mapping,
    complexity_features_per_class,
    complexity_features_per_cluster_class,
    k,
    h,
    distance_metric
):
    
    print(f"- Running topk-inference on {inference_df.shape[0]:,} samples with k={k}.")
    
    weak_proba = weak_clf.predict_proba(inference_df) # Find weak proba of all samples
    top_k_classes = np.argsort(weak_proba, axis=1)[:, -k:] # Find the top-k classes of each sample's weak_proba

    print(f"- Finding class_to_indices for iteration...")
    class_to_indices = defaultdict(list) # dict that maps, for each class (numeric class, not label), the index of the samples that are candidate for that class
    for i in range(inference_df.shape[0]):
        for cls in top_k_classes[i]:
            class_to_indices[cls].append(i)

    # Find the best pred and confidence by iterating over each sample per class, storing the last "best" in its index in an array of zeros
    best_predictions = np.empty(inference_df.shape[0], dtype=object) # Each has the class label and not the class index
    best_confidences = np.zeros(inference_df.shape[0])

    what_is_happening = {} # TODO REMOVE

    # cls is the class being tested as extension, idxs are the global indexes of the samples that have been candidated to class "cls" by the weak classifier
    for cls, idxs_of_df in tqdm(class_to_indices.items(), total=len(class_to_indices), desc="Classes", position=0):
        X_of_class = inference_df.iloc[idxs_of_df].copy() # Retrieve the samples to be tested for the cls extension
        # In X_of_class the dataframe is made of rows that respect the order of insertion in class_to_indices.
        # This means that the element of index 0 in X_of_class will have global index found in class_to_indices[cls][0]
        # global_index = class_to_indices[cls][0]

        # TEST VERSION WITH CLUSTER MODELS ------------
        # Find the top h cluster classes each sample in X belongs to, for class cls
        #cls_label = label_mapping[str(cls)]
        # Lazy load the cluster model, so that it's loaded only when necessary
        #if isinstance(cluster_models[cls_label], str):
        #    print(f"- Loading clustering model {cls_label} from joblib...")
        #    cluster_models[cls_label] = load_from_joblib(cluster_models[cls_label])
        #cluster_model = cluster_models[cls_label] # Load clustering model to infer the subclass each sample belongs to, to extend the subclass column for each sample
        # Find the top h clusters of belonging by sorting the distances to each centroid and retrieving the top h
        # Distances from each sample to each centroid
        #cluster_model_dist = cluster_model.transform(X_of_class)
        #top_h_clusters_classes = np.argsort(cluster_model_dist, axis=1)[:, :h] # The smaller distance, the better
        # print("CLUSTER MODEL FEATURES OF TRAINING")
        #print(sorted(list(cluster_model.feature_names_in_)))
        # print("COLUMNS OF DATASET")
        # print(sorted(list(X_of_class.columns)))
        #reversed_dict = {v:k for k,v in label_mapping.items()}
        # print("COLUMNS OF DATASET (int)")
        #print(sorted([reversed_dict[col_name] for col_name in list(X_of_class.columns)]))
        # ---------------------------

        # Indices of the h closest clusters for each sample
        print(f"Finding top h nearest clusters for points in class \"{label_mapping[str(cls)]}\"...")
        top_h_clusters_classes, _ = find_top_h_nearest_clusters(h=h,
                                                             distance_metric=distance_metric,
                                                             df_of_class=X_of_class, 
                                                             num_cols=num_cols,
                                                             cat_cols=cat_cols,
                                                             class_label_int=cls, 
                                                             class_to_clusters=class_to_clusters, 
                                                             cluster_centroids=cluster_centroids)

        cluster_class_to_indices = defaultdict(list) # dict that maps, for each cluster class (numeric class, not label), the index of the samples that, for the class cls, are candidate for that cluster class (subclass)
        for i in range(X_of_class.shape[0]):
            for cluster_class in top_h_clusters_classes[i]:
                cluster_class_to_indices[cluster_class].append(i)

        # Find best_conf and best_pred by iterating over each subclass of class of each sample of initial class
        for cluster_class_in_cls, idxs_of_X_of_class in tqdm(cluster_class_to_indices.items(), total=len(cluster_class_to_indices), desc=f"Subclasses of \"{label_mapping[str(cls)]}\"", position=1, leave=False):
            X_of_cluster_class = X_of_class.iloc[idxs_of_X_of_class].copy() # Retrieve the samples to be tested for the cls + cluster_class extension
            # In X_of_cluster_class the dataframe is made of rows that respect the order of insertion in cluster_class_to_indices
            # This means that the element of index 0 in X_of_cluster_class will have index in X_of_class found in cluster_class_to_indices[0]
            # To retrieve the global index, you have to follow the chain, which becomes
            # global_index = class_to_indices[cls][cluster_class_to_indices[cluster_class_in_cls][0]]

            # obtain the extended version of the samples for this class and cluster class (subclass)
            X_ext = extend_features_batch(
                df=X_of_cluster_class, # Only obtain the extension for the subset of samples to check
                complexity_extension_class=complexity_features_per_class[str(cls)], # Obtain the extension for the entire class, equal for every sample
                complexity_extension_cluster_class=complexity_features_per_cluster_class[str(cluster_class_in_cls)],
                column_ordering=ext_clf.named_steps["clf"].feature_names_in_, # Obtain the ordering of columns to follow for the extension
            )
 
            per_class_and_cluster_class_ext_proba = ext_clf.predict_proba(X_ext)

            what_is_happening[f"{label_mapping[str(cls)]} of cluster {cluster_class_in_cls}"] = [str(_) for _ in per_class_and_cluster_class_ext_proba[:5]] # TODO REMOVE

            per_class_and_cluster_class_preds = np.argmax(per_class_and_cluster_class_ext_proba, axis=1)
            per_class_and_cluster_class_confs = np.max(per_class_and_cluster_class_ext_proba, axis=1)

            for sample_index_in_the_cluster_class_of_class in tqdm(range(X_of_cluster_class.shape[0]), total=X_of_cluster_class.shape[0], desc=f"Subclass {cluster_class_in_cls} of \"{label_mapping[str(cls)]}\"", position=2, leave=False):
                sample_index_global = class_to_indices[cls][cluster_class_to_indices[cluster_class_in_cls][sample_index_in_the_cluster_class_of_class]] # Find the global index for comparisons
                
                # print(inference_df.iloc[sample_index_global])
                # print(X_of_cluster_class.iloc[sample_index_in_the_cluster_class_of_class])
                # The prints are used to check if the rows found are the exact same. Yes, they are. It means sample_index_global is correct
                
                if per_class_and_cluster_class_confs[sample_index_in_the_cluster_class_of_class] > best_confidences[sample_index_global]: # If the current confidence is better than the old confidence
                    if per_class_and_cluster_class_preds[sample_index_in_the_cluster_class_of_class] in top_k_classes[sample_index_global]:# If the current class is within the ones predicted as the top k classes of the weak classifier
                        # Update global dataframe infos of the sample
                        best_predictions[sample_index_global] = label_mapping[str(per_class_and_cluster_class_preds[sample_index_in_the_cluster_class_of_class])]
                        best_confidences[sample_index_global] = per_class_and_cluster_class_confs[sample_index_in_the_cluster_class_of_class]

    print(f"- Column meanings:\n{[label_mapping[str(_)] for _ in ext_clf.classes_]}\n")
    print(f"- what_is_happening:")
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

    #if cfg.clustering.only_numerical_columns:
    #    print("- Keeping only numerical columns in the dataframe...")
    #    inference_df = inference_df[cfg.data.num_cols]
    #else:
    #    print("- Keeping both numerical and categorical columns in the dataframe...")

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
    print("- Loading config...")
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:]
    )

    k = cfg.topk_inference.k
    h = cfg.topk_inference.h
    distance_metric = cfg.complexity.distance

    weak_clf_path = Path(cfg.path.models) / 'model.joblib'
    ext_clf_path  = Path(cfg.path.models) / "model_extended.joblib"

    print("- Loading classifiers...")
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
    complexity_features_per_cluster_class = load_from_json(Path(cfg.path.shared) / "complexity.json")

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

    clusters_meta = load_from_json(Path(cfg.path.shared) / "metadata" / "clusters_meta.json")
    class_to_clusters = clusters_meta["class_to_clusters"]
    cluster_centroids = clusters_meta["centroids"]

    # Find predictions and confidences, and append to df as columns
    predictions, confidences = run_topk_predict_on_inference_input_df(
                    inference_df=inference_df,
                    num_cols=cfg.data.num_cols,
                    cat_cols=cfg.data.cat_cols if not cfg.clustering.only_numerical_columns else [],
                    weak_clf=weak_clf,
                    ext_clf=ext_clf,
                    # cluster_models=cluster_models,
                    class_to_clusters=class_to_clusters,
                    cluster_centroids=cluster_centroids,
                    label_mapping=label_mapping,
                    complexity_features_per_class=complexity_features_per_class,
                    complexity_features_per_cluster_class=complexity_features_per_cluster_class,
                    k=k,
                    h=h,
                    distance_metric=distance_metric
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