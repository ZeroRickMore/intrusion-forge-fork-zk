import sys
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import random
from scipy.spatial.distance import cdist
from pprint import pformat

from src.core.config import load_config
from src.core.utils import load_from_joblib, load_from_json, save_to_json
from src.core.io import load_df, save_df
from collections import Counter

from src.domain.analysis.complexity.shared import (
    l2_normalize
)

GLOBAL_STATS = defaultdict(lambda : defaultdict(lambda : defaultdict()))

def extend_features_batch(df, 
                          complexity_extension_class : dict, 
                          complexity_extension_cluster_class : dict, 
                          column_ordering,
                          use_class_for_extension : bool,
                          use_cluster_for_extension : bool):
    """
    - df : the subset of samples that require the extension
    - complexity_extension_class: dict like {feature_name: value}, the extension columns with values for the class, equal for each sample  
    - complexity_extension_cluster_class: dict like {feature_name: value}, the extension columns with values for the cluster class of the class (subclass), equal for each sample
    - column_ordering: the list of the columns that appear in the classifier, which will be followed during the extension
    """
    df_ext = df.copy()

    if use_class_for_extension:
        for column_name, column_values in complexity_extension_class.items():
            df_ext[column_name] = column_values

    if use_cluster_for_extension:
        for column_name, column_values in complexity_extension_cluster_class.items():
            df_ext[column_name] = column_values

        # Handle noise clusters missing a lot of columns, that default to 0.0
        if complexity_extension_cluster_class['cluster_is_noise_cluster'] == 1.0:
            for column_name in column_ordering:
                if column_name.startswith("cluster_"):
                    df_ext[column_name] = 0.0

    return df_ext[column_ordering]

def find_top_h_nearest_clusters(
    h,
    df_of_class,
    num_cols,
    cat_cols,
    class_label_int,
    class_to_clusters,
    cluster_centroids,
    distance_metric,
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
    distance_metric,
    use_class_for_extension,
    use_cluster_for_extension
):
    global GLOBAL_STATS
    print(f"- Running topk-inference on {inference_df.shape[0]:,} samples with k={k}.")
    
    weak_proba = weak_clf.predict_proba(inference_df) # Find weak proba of all samples
    top_k_classes = np.argsort(weak_proba, axis=1)[:, -k:] # Find the top-k classes of each sample's weak_proba
 
    print(f"\n- Finding class_to_indices for iteration...")
    class_to_indices = defaultdict(list) # dict that maps, for each class (numeric class, not label), the index of the samples that are candidate for that class
    for i in range(inference_df.shape[0]):
        for cls in top_k_classes[i]:
            class_to_indices[cls].append(i)

    # Find the best pred and confidence by iterating over each sample per class, storing the last "best" in its index in an array of zeros
    best_predictions = np.empty(inference_df.shape[0], dtype=object) # Each has the class label and not the class index
    best_confidences = np.zeros(inference_df.shape[0])

    what_is_happening = {} # TODO REMOVE

    print(f"\n- Iterating over the classes...")
    # cls is the class being tested as extension, idxs are the global indexes of the samples that have been candidated to class "cls" by the weak classifier
    for cls, idxs_of_df in tqdm(class_to_indices.items(), total=len(class_to_indices), desc="Classes", position=0):
        X_of_class = inference_df.iloc[idxs_of_df].copy() # Retrieve the samples to be tested for the cls extension
        class_label = label_mapping[str(cls)]
        # In X_of_class the dataframe is made of rows that respect the order of insertion in class_to_indices.
        # This means that the element of index 0 in X_of_class will have global index found in class_to_indices[cls][0]
        # global_index = class_to_indices[cls][0]

        # Indices of the h closest clusters for each sample
        top_h_clusters_classes, distances = find_top_h_nearest_clusters(h=h,
                                                             distance_metric=distance_metric,
                                                             df_of_class=X_of_class,
                                                             num_cols=num_cols,
                                                             cat_cols=cat_cols,
                                                             class_label_int=cls,
                                                             class_to_clusters=class_to_clusters,
                                                             cluster_centroids=cluster_centroids)

        # Populate global stats ---------------------
        class_labels = [label_mapping[str(_)] for _ in list(ext_clf.classes_)] 
        # -------------------------------------------

        cluster_class_to_indices = defaultdict(list) # dict that maps, for each cluster class (numeric class, not label), the index of the samples that, for the class cls, are candidate for that cluster class (subclass)
        for i in range(X_of_class.shape[0]):
            for cluster_class in top_h_clusters_classes[i]:
                cluster_class_to_indices[cluster_class].append(i)

        # Find best_conf and best_pred by iterating over each subclass of class of each sample of initial class
        for cluster_class_in_cls, idxs_of_X_of_class in tqdm(cluster_class_to_indices.items(), total=len(cluster_class_to_indices), desc=f"Clusters of \"{class_label}\"", position=1, leave=False):
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
                use_class_for_extension=use_class_for_extension,
                use_cluster_for_extension=use_cluster_for_extension
            )

            per_class_and_cluster_class_ext_proba = ext_clf.predict_proba(X_ext)

            what_is_happening[f"{class_label} of cluster {cluster_class_in_cls}"] = [str(_) for _ in per_class_and_cluster_class_ext_proba[:5]] # TODO REMOVE

            per_class_and_cluster_class_preds = np.argmax(per_class_and_cluster_class_ext_proba, axis=1)
            per_class_and_cluster_class_confs = np.max(per_class_and_cluster_class_ext_proba, axis=1)

            for sample_index_in_the_cluster_class_of_class in tqdm(range(X_of_cluster_class.shape[0]), total=X_of_cluster_class.shape[0], desc=f"Cluster {cluster_class_in_cls} of \"{class_label}\"", position=2, leave=False):
                sample_index_of_class = cluster_class_to_indices[cluster_class_in_cls][sample_index_in_the_cluster_class_of_class]
                sample_index_global = class_to_indices[cls][sample_index_of_class] # Find the global index for comparisons
                
                # Populate global stats ---------------------
                sample_key = str(sample_index_global)
                class_key = f"Class {cls} ({label_mapping[str(cls)]})"
                cluster_key = f"Cluster Id {cluster_class_in_cls}"

                # Keep populating a pre existent sample in the dict
                if str(sample_index_global) in GLOBAL_STATS['Top k Inference Execution']['Per Sample']:
                    # Common part
                    if class_key not in GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class']:
                        GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'][class_key] = {
                            'Class Id' : str(cls),
                            'Class Label' : label_mapping[str(cls)]
                        } # Init this class
                        GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'][class_key]['Top h Clusters (Cluster Id)'] = top_h_clusters_classes[sample_index_of_class].tolist()
                        GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'][class_key]['Per Cluster'] = {}

                    if cluster_key in GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'][class_key]['Per Cluster']:
                        raise ValueError(f"This should never happen! Why is {cluster_key} already in the sample?? Here it is:\n-------------------\n{pformat(GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key])}")

                    GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'][class_key]['Per Cluster'][cluster_key] = {
                        'Cluster Id' : str(cluster_class_in_cls),
                        'Ext Proba (Value, Class Label)' : [(per_class_and_cluster_class_ext_proba[sample_index_in_the_cluster_class_of_class].tolist()[i], class_labels[i]) for i in range(len(class_labels))],
                        'Current Cluster Top Prediction (Class Id, Class Label)' : (str(per_class_and_cluster_class_preds[sample_index_in_the_cluster_class_of_class]), label_mapping[str(per_class_and_cluster_class_preds[sample_index_in_the_cluster_class_of_class])]),
                        'Current Cluster Top Confidence (Value)' : float(per_class_and_cluster_class_confs[sample_index_in_the_cluster_class_of_class]),
                    }        
                # ---------------------------------
                # Add new sample.
                # Only if there aren't way too many samples already... Also, make this a little random, so that you get some representation of all the classes in a way
                # Maximum amount of samples is 50, else the output file would be huge
                elif (random.randrange(0,100) > 90) and (len(GLOBAL_STATS['Top k Inference Execution']['Per Sample']) < 50): # Remove true for control over the sample insertion
                    GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key] = {} # Init sample  
                    GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Weak Proba (Value, Class Label)'] = [(weak_proba[sample_index_in_the_cluster_class_of_class].tolist()[i], class_labels[i]) for i in range(len(class_labels))]
                    GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Top k Classes (Class Id, Class Label)'] = [(str(class_id), label_mapping[str(class_id)]) for class_id in top_k_classes[sample_index_in_the_cluster_class_of_class].tolist()]
                    GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'] = {} # Init per class
                    
                    # Common part
                    GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'][class_key] = {
                        'Class Id' : str(cls),
                        'Class Label' : label_mapping[str(cls)]
                    } # Init this class
                   
                    GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'][class_key]['Top h Clusters (Cluster Id)'] = top_h_clusters_classes[sample_index_of_class].tolist()
                    GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'][class_key]['Per Cluster'] = {}
                    GLOBAL_STATS['Top k Inference Execution']['Per Sample'][sample_key]['Per Class'][class_key]['Per Cluster'][cluster_key] = {
                        'Cluster Id' : str(cluster_class_in_cls),
                        'Ext Proba (Value, Class Label)' : [(per_class_and_cluster_class_ext_proba[sample_index_in_the_cluster_class_of_class].tolist()[i], class_labels[i]) for i in range(len(class_labels))],
                        'Current Cluster Top Prediction (Class Id, Class Label)' : (str(per_class_and_cluster_class_preds[sample_index_in_the_cluster_class_of_class]), label_mapping[str(per_class_and_cluster_class_preds[sample_index_in_the_cluster_class_of_class])]),
                        'Current Cluster Top Confidence (Value)' : float(per_class_and_cluster_class_confs[sample_index_in_the_cluster_class_of_class]),
                    }
                # -------------------------------------------

                # print(inference_df.iloc[sample_index_global])
                # print(X_of_cluster_class.iloc[sample_index_in_the_cluster_class_of_class])
                # The prints are used to check if the rows found are the exact same. Yes, they are. It means sample_index_global is correct
                
                if per_class_and_cluster_class_confs[sample_index_in_the_cluster_class_of_class] > best_confidences[sample_index_global]: # If the current confidence is better than the old confidence
                    if per_class_and_cluster_class_preds[sample_index_in_the_cluster_class_of_class] in top_k_classes[sample_index_global]:# If the current class is within the ones predicted as the top k classes of the weak classifier
                        # Update global dataframe infos of the sample
                        best_predictions[sample_index_global] = label_mapping[str(per_class_and_cluster_class_preds[sample_index_in_the_cluster_class_of_class])]
                        best_confidences[sample_index_global] = per_class_and_cluster_class_confs[sample_index_in_the_cluster_class_of_class]
                    else:
                        pass # The proposed "best" class did not belong to the weak classifier, thus it's not a good candidate

    return best_predictions, best_confidences







def load_inference_input_df_and_strip_labels(cfg):
    global GLOBAL_STATS
    print("- Loading inference input dataset...")

    split_frac_of_trained_models = load_from_json(Path(cfg.path.shared) / "metadata" / "df_info.json").get("split_frac", None)

    GLOBAL_STATS['Config']['split_frac'] = split_frac_of_trained_models

    print(f"- Training split fraction: {split_frac_of_trained_models}")

    if split_frac_of_trained_models in [None, 100.0]:
        raise ValueError(
            f"- The models were trained on a bad split_frac: {split_frac_of_trained_models},\n"
            f"  please re-run the entire pipeline with a reasonable\n"
            f"  'prepare.topk_inference.split_frac' parameter.\n"
        )
    
    if False and split_frac_of_trained_models != cfg.prepare.topk_inference.split_frac:
        raise ValueError(f"- The models were trained on split_frac={split_frac_of_trained_models},\n"
                         f"  but your current configuration's split_frac is {cfg.prepare.topk_inference.split_frac}.\n"
                         f"\t- Changing config's split_frac to {split_frac_of_trained_models} will solve this error,\n"
                          "\tbut make sure this is intended before doing so, as running inference on a model trained\n"
                          "\ton a different split_frac may result in data leak and false predictions.")

    inference_input_path = Path(cfg.path.dataset_split) / "inference_input.pkl"
    # inference_input_path = Path(cfg.path.dataset_split) / "trained_on.pkl"

    GLOBAL_STATS['Config']['Paths']['Inference Dataframe'] = str(inference_input_path)

    print(f"- Loading df from inference input path:\n\t{inference_input_path}")

    if not os.path.exists(inference_input_path):
        raise ValueError(
            f"- [{inference_input_path}] does not exist: cannot topk-infer with no input data.\n"
            f"\t- You should run prepare_data with \"prepare.topk_inference.split_dataset : True\"."
        )

    #inference_df = load_df(file_path=inference_input_path)
    inference_df = load_df(Path(cfg.path.processed_data) / 'test.parquet')

    print(f"- Loaded inference dataset with shape {inference_df.shape}")

    labels = inference_df.pop(cfg.data.label_col) # Remove labels to avoid information leak on predictions
    encoded_labels = inference_df.pop(f"encoded_{cfg.data.label_col}") if f"encoded_{cfg.data.label_col}" in inference_df.columns else None # Remove encoded labels to avoid information leak on predictions

    GLOBAL_STATS['Config']['Label Column'] = cfg.data.label_col

    print(f"- Removed label columns '{cfg.data.label_col}' and 'encoded_{cfg.data.label_col}' (encoded if present) from inference features.")

    #if cfg.clustering.only_numerical_columns:
    #    print("- Keeping only numerical columns in the dataframe...")
    #    inference_df = inference_df[cfg.data.num_cols]
    #else:
    #    print("- Keeping both numerical and categorical columns in the dataframe...")

    return inference_df, labels, encoded_labels

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
    global GLOBAL_STATS

    print("- Loading config...")
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:]
    )

    k = cfg.topk_inference.k
    h = cfg.topk_inference.h
    distance_metric = cfg.complexity.distance
    use_class_for_extension = cfg.extend.use_class_features
    use_cluster_for_extension = cfg.extend.use_cluster_features

    if not use_cluster_for_extension:
        print("- WARNING: As you selected not to use cluster features to extend samples,\n  parameter 'h' will be set to 1, defaulting to a class-only extension.")
        h = 1
    
    GLOBAL_STATS['Config']['k classes'] = k
    GLOBAL_STATS['Config']['h clusters'] = h
 
    weak_clf_path = Path(cfg.path.models) / 'model.joblib'
    ext_clf_path  = Path(cfg.path.models) / "model_extended.joblib"

    GLOBAL_STATS['Config']['Paths']['Weak Classifier'] = str(weak_clf_path)
    GLOBAL_STATS['Config']['Paths']['Extended Classifier'] = str(ext_clf_path)

    print("- Loading classifiers...")
    print(f"\t- Weak classifier path:\n\t{weak_clf_path}")
    print(f"\t- Extended classifier path:\n\t{ext_clf_path}")

    weak_clf = load_from_joblib(weak_clf_path)
    ext_clf = load_from_joblib(ext_clf_path)

    print(f"- Loading clustering models mapping for clustering algorithm \"{cfg.clustering.name}\", and label mappings for the entire dataset.")

    # cluster_models = load_from_json(file_path=Path(cfg.path.clustering_models) / "class_to_model.json")[str(cfg.clustering.name)]
    # for class_ in cluster_models:
    #     cluster_models[class_] = load_from_joblib(cluster_models[class_])

    label_mapping = load_from_json(Path(cfg.path.shared) / "metadata" / "df_meta.json")["label_mapping"]
    complexity_features_per_class = load_from_json(Path(cfg.path.shared) / "class_complexity.json")
    complexity_features_per_cluster_class = load_from_json(Path(cfg.path.shared) / "complexity.json")

    # GLOBAL_STATS['config']['classes_infos']['label_mapping'] = label_mapping
    # GLOBAL_STATS['config']['complexity']['complexity_features_per_class'] = complexity_features_per_class
    # GLOBAL_STATS['config']['complexity']['complexity_features_per_cluster_class'] = complexity_features_per_cluster_class

    inference_df_full, labels, encoded_labels = load_inference_input_df_and_strip_labels(cfg)

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

    GLOBAL_STATS['Config']['Features']['Inference Dataframe'] = sorted(list(inference_df_full.columns))
    GLOBAL_STATS['Config']['Features']['Weak Classifier'] = sorted(expected_features_weak_clf)
    GLOBAL_STATS['Config']['Features']['Extended Classifier'] = sorted(list(ext_clf.named_steps["clf"].feature_names_in_))

    clusters_meta = load_from_json(Path(cfg.path.shared) / "metadata" / "clusters_meta.json")
    class_to_clusters = clusters_meta["class_to_clusters"]
    cluster_centroids = clusters_meta["centroids"]

    # GLOBAL_STATS['config']['classes_infos']['class_to_clusters'] = class_to_clusters
    # GLOBAL_STATS['config']['classes_infos']['cluster_centroids'] = cluster_centroids

    # Filter out Benign for more precise understanding on how the samples failed
    # mask = labels != str(cfg.data.benign_tag)
    # labels = labels[mask]
    # inference_df = inference_df[mask]

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
                    distance_metric=distance_metric,
                    use_class_for_extension=use_class_for_extension,
                    use_cluster_for_extension=use_cluster_for_extension,
                )
    del inference_df

    # Place the true label on each point seen during the inference in its dict for easy review
    reversed_label_mapping = {v : k for k,v in label_mapping.items()}
    print("- Updating global stats with final information...")
    for global_idx in tqdm(list(GLOBAL_STATS['Top k Inference Execution']['Per Sample'].keys())[:], 
                    total=len(GLOBAL_STATS['Top k Inference Execution']['Per Sample'].keys()), 
                    desc='Samples', 
                    position=0):
        
        # Keep exclusively the ones that failed
        if str(predictions[int(global_idx)]) == str(labels.iloc[int(global_idx)]):
            GLOBAL_STATS['Top k Inference Execution']['Per Sample'].pop(global_idx)
            continue

        # Quick reordering of keys for better visualization
        GLOBAL_STATS['Top k Inference Execution']['Per Sample'][global_idx] = {
            'True Class (Class Id, Class Label)' : (reversed_label_mapping[labels.iloc[int(global_idx)]], labels.iloc[int(global_idx)]),
            'Best Prediction (Class Id, Class Label)' : (reversed_label_mapping[predictions[int(global_idx)]], predictions[int(global_idx)]),
            'Best Confidence (Value)' : predictions[int(global_idx)],
            'Weak Proba (Value, Class Label)' : GLOBAL_STATS['Top k Inference Execution']['Per Sample'][global_idx]['Weak Proba (Value, Class Label)'],
            'Top k Classes (Class Id, Class Label)' : GLOBAL_STATS['Top k Inference Execution']['Per Sample'][global_idx]['Top k Classes (Class Id, Class Label)'],
            'Per Class' : GLOBAL_STATS['Top k Inference Execution']['Per Sample'][global_idx]['Per Class'],
        }
    
    global_stats_output_path = Path(cfg.path.topk_inference_out) / 'global_stats.json'
    print(f"- Saving GLOBAL_STATS to {global_stats_output_path}")
    save_to_json(data=GLOBAL_STATS, file_path=global_stats_output_path)

    output_path = Path(cfg.path.topk_inference_out) / "results_with_original_labels.pkl"
    print(f"- Saving results df to {output_path}")
    save_df(df=inference_df_full, file_path=output_path)

    print_diagnostics(inference_df_full=inference_df_full, confidences=confidences, predictions=predictions, labels=labels)

    print("- Topk-inference completed.")



if __name__ == "__main__":
    main()