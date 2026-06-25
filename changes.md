# Changes
A list of all the changes done to the original project to implement topk-inference in the pipeline.

## Config changes
-   Each clustering algorithm has parameter "name" in its config.
-   in /configs/default.yaml, 'prepare' has now the following form, with just the key 'topk_inference' being added and nothing else removed/modified, in the following form.
    ```yaml
    prepare:
    force: false                         # re-run preprocessing+clustering even if shared outputs exist
    topk_inference:                      # topk_inference settings for data preparation. On the first run, everything has to be true.
        save_clustering_models: True     # saves the best clustering model(s) into ${path.out_base_path}/topk_inference
        split_dataset: True              # Splits the dataset into 2 parts, one used for the usual prepare_data pipeline, the other will stay untouched and will be later used by topk_inference
        split_frac: 0.7                  # The fraction that will be used by the usual prepare_data pipeline. The remaining will be used as topk_inference input
    ```
- in /configs/path/default.yaml config, the following keys have been added, nothing else removed/modified
    ```yaml
    # output splits
    dataset_split: resources/raw_data/$(data.dir)/Dataset_Splits/${data.file_name}/$(prepare.topk_inference.split_frac)

    [...]

    # Per-classifier (resolved against ${classifier.name})
    clustering_models:  ${path.out_base_path}/${classifier.name}/clustering_models
    topk_inference_out: $(path.out_base_path)/${classifier.name}/topk_inference_results_$(prepare.topk_inference.split_frac)
    ```

## /MakeFile
- Added a line to run topk_inference.py via MakeFile:
    ```makefile
    ## topk_inference
    topk-inference:
        PYTHONPATH=. $(PYTHON) pipelines/topk_inference.py $(HYDRA)
    ```

## pipelines/prepare_data.py
-   df_info.json now contains a new `split_frac` key, representing the split fraction the dataset was prepared on.  
    This will be used by topk_inference.py for sanity check, so that there can be no misconfiguration that leads to the following scenario, for example:
    - Dataset prepared on split 0.7
    - Topk-inference ran on split 0.6 (by mistake)
    This scenario would cause strange behavior of leaked data during the training phase, thus avoided.
-   `cfg.prepare.topk_inference.split_dataset` will be used to:
    - split the dataframe, save the two halves into the following paths, and use only the first part of the split for prepare_data.py
        - {cfg.path.dataset_split} / f'trained_on.pkl'
        - {cfg.path.dataset_split} / f'inference_input.pkl'  

        The split method is called right after dataframe filtering in prepare_data.preprocess_df(), so that the inference_input dataframe will not contain the rows that would have been filtered in the entire dataframe.  
        The split is executed via this code in preprocessing.py, trying to ensure a representative split that does not benefit any class in both splits. 

        ```py
        def representative_split(
                df,
                split_frac,
                random_state: int | None = None,
                label_col: str | None = None,
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                """Split a DataFrame into train, validation, and test sets with optional stratification."""

                stratify = df[label_col] if label_col else None
                train_df, rest = train_test_split(
                    df, 
                    train_size=split_frac, 
                    random_state=random_state, 
                    stratify=stratify
                )

                return train_df, rest
        ```
-   `cfg.prepare.topk_inference.save_clustering_models` will be used to:
    - save the clustering models, each associated to its class. The results will be:
        - Multiple .joblib file containing each model per class, in format:  
            {cfg.path.clustering_models} / {current_class_name}___{cfg.clustering.name}.joblib
        - A class_to_model.json file containing the mapping from the clustering algorithm name to each class and its .joblib model file.  
        Here is an example of this file:  

        ```json
        {
            "kmeans": {
                "Benign": "resources/experiments/topk_inference_kmeans/unsw_nb15_v2_42/random_forest/clustering_models/Benign___kmeans.joblib",
                "DoS": "resources/experiments/topk_inference_kmeans/unsw_nb15_v2_42/random_forest/clustering_models/DoS___kmeans.joblib",
                "Exploits": "resources/experiments/topk_inference_kmeans/unsw_nb15_v2_42/random_forest/clustering_models/Exploits___kmeans.joblib",
                "Fuzzers": "resources/experiments/topk_inference_kmeans/unsw_nb15_v2_42/random_forest/clustering_models/Fuzzers___kmeans.joblib",
                "Generic": "resources/experiments/topk_inference_kmeans/unsw_nb15_v2_42/random_forest/clustering_models/Generic___kmeans.joblib",
                "Reconnaissance": "resources/experiments/topk_inference_kmeans/unsw_nb15_v2_42/random_forest/clustering_models/Reconnaissance___kmeans.joblib"
            }
        }
        ```

    To allow this saving behavior, each fit_fn in algorithm.py has been changed to return both the labels and the model (earlier, it was only the labels), and in a cascading way, each method call that interacted with the fit_fn will handle this extra parameter, such as grid_search.

    Ensemble algorithm model saving has been implemented, this is the reason for class_to_model.json having a clustering algorithm name key, but not supported in topk_inference.py yet. Let's say it is a template that can be implemented afterwards, but as of now, the models are stored correctly. Again, this has not been tested thoroughly, though.

## pipelines/topk_inference.py
Entire workflow.
- main() loads all the information used for the topk-inference, and executes sanity checks where necessary.
    -  **inference_df**: loaded from `{cfg.path.dataset_split} / {cfg.path.shared/"metadata"/"df_info.json"['split_frac']} / "inference_input.pkl"`.  
        - The implicit sanity check makes it so that the topk-inference will always happen on the inference_input.pkl split that was produced during prepare_data.py, as the folder of the split is taken from df_info.json that is only modified during prepare_data.py
        - The dataset contains all of the columns, but not all of them are used by the model for training.  
        For this reason, the training columns are inferred from the weak_clf itself, and removed from the dataframe.
        - The dataset contains the Label column, so main() handles the removal of this column not to interfere with the sample classification.  
        The labels are then re-inserted in the df after the classification, to allow for a quick inspection and quality check.
    - **weak_clf**: output of "make classify"
    - **ext_clf**: output of "make classify-extended"
    - **cluster_models**: output of "make prepare" with topk-inference flag set to True in config
    - **label_mapping**: found in shared/metadata/df_meta.json and used to infer the class label from the index returned by predict_proba(x)
    - **complexity_features_per_class**: found in shared/class_complexity.json and used to infer the extension features to add complexity to the samples. It is used to find x_ext which is sample x extended with the feature of the candidate class, once per each class in the top-k classes found with predict_proba(x).
    - **k**: cfg.topk_inference.k

- after loading the information, run_topk_predict_on_inference_input_df() is called, which:
    - calls top_k_predict() on each sample x of the inference_input_df
    - stores the result (best_prediction, best_confidence) inside of the columns "prediction" and "confidence" of the df
    - returns the df containing, per each sample, the result of the inference

- after inference, the original labels are re-attached to the df under the column "original_label". This allows for easier information gathering on how the inference performed against the original labels.
- finally, the dataframe is saved into file `{cfg.path.topk_inference_out} / results_with_original_labels.pkl`

# TODO
- Return the classifier of Spectral Algorithm
- Understand if topk_inference.k param is the exact same as complexity.top_k_clusters (for now, they are two separate config params)
- If class_complexity.json is a single one and not per clustering model, what is this pseudocode line: `complexity = cluster_models[c].extract_complexity(x)`? As of now, the cluster_models are literally never used, which is kind of strange, but if we consider that class_complexity.json has the complexity of each class, maybe it is implicit that it uses the clustering model for that class.
- Executing the pseudocode and the function sample-by-sample takes around 72 hours for 700k samples... We must use vectorial operations, is it correct? It's in the order of seconds then.
- topk-inference exclusively predicts as Benign. Here is a potential reason:
    - Benign class is EXTREMELY over-represented in the dataset. The following is the class distribution of the entire dataframe. Note that some class might be filtered as a "rare category", but this is not the focus point here. This is the print of "out" in prepare_data.py, which is currently commented in the code.
    ```yaml
    0          Benign  2295222   96.023345
    1        Exploits    31551    1.319974
    2         Fuzzers    22310    0.933365
    3         Generic    16560    0.692807
    4  Reconnaissance    12779    0.534625
    5             DoS     5794    0.242399
    6        Analysis     2299    0.096181
    7        Backdoor     2169    0.090743
    8       Shellcode     1427    0.059700
    9           Worms      164    0.006861
    ```
    The same exact percentage distribution is found in the inference_input dataset and trained_on dataset, so the split follows the proportions of the original dataset.  
    As such, Benign is so over-represented that it is supposed that the extended classifiers have a huge bias towards that class, which is the reason why the output of each candidate class, per each sample, ends up being in favor of Benign with a large gap.  
    The following is the print of "what_is_happening" in topk_inference_vectorial.py, containing the first 10 sample results of predict_proba(x_ext), per class.
    ```yaml
    {
        "Generic": [
            "[0.004 0.06  0.336 0.08  0.164 0.09  0.176 0.048 0.    0.042]",
            "[0.004 0.056 0.35  0.09  0.148 0.09  0.178 0.048 0.    0.036]",
            "[0.004 0.062 0.332 0.084 0.162 0.09  0.176 0.048 0.    0.042]",
            "[0.004 0.06  0.34  0.082 0.158 0.09  0.176 0.048 0.    0.042]",
            "[0.004 0.062 0.338 0.086 0.152 0.092 0.176 0.048 0.    0.042]"
        ],
        "DoS": [
            "[0.01  0.05  0.332 0.086 0.148 0.098 0.17  0.062 0.002 0.042]",
            "[0.01  0.052 0.334 0.084 0.144 0.096 0.17  0.062 0.002 0.046]",
            "[0.01  0.046 0.346 0.092 0.134 0.098 0.174 0.062 0.002 0.036]",
            "[0.01  0.052 0.328 0.09  0.146 0.098 0.17  0.062 0.002 0.042]",
            "[0.012 0.052 0.33  0.082 0.15  0.098 0.166 0.062 0.002 0.046]"
        ],
        "Backdoor": [
            "[0.008 0.064 0.336 0.076 0.146 0.108 0.146 0.076 0.002 0.038]",
            "[0.008 0.066 0.338 0.072 0.144 0.106 0.146 0.076 0.002 0.042]",
            "[0.008 0.06  0.35  0.084 0.132 0.108 0.148 0.076 0.002 0.032]",
            "[0.008 0.066 0.332 0.078 0.146 0.108 0.146 0.076 0.002 0.038]",
            "[0.01  0.066 0.334 0.07  0.148 0.108 0.142 0.078 0.002 0.042]"
        ],
        "Worms": [
            "[0.004 0.074 0.344 0.088 0.156 0.108 0.114 0.05  0.002 0.06 ]",
            "[0.004 0.076 0.346 0.082 0.156 0.106 0.114 0.05  0.002 0.064]",
            "[0.004 0.076 0.338 0.09  0.158 0.108 0.114 0.05  0.002 0.06 ]",
            "[0.006 0.076 0.342 0.08  0.162 0.108 0.11  0.05  0.002 0.064]",
            "[0.004 0.074 0.348 0.088 0.152 0.108 0.114 0.05  0.002 0.06 ]"
        ],
        "Benign": [
            "[0.004 0.036 0.368 0.074 0.202 0.086 0.13  0.044 0.002 0.054]",
            "[0.004 0.038 0.37  0.072 0.202 0.084 0.13  0.044 0.002 0.054]",
            "[0.004 0.04  0.382 0.072 0.198 0.086 0.132 0.044 0.002 0.04 ]",
            "[0.004 0.038 0.364 0.074 0.204 0.086 0.13  0.044 0.002 0.054]",
            "[0.006 0.038 0.366 0.07  0.208 0.086 0.126 0.044 0.002 0.054]"
        ],
        "Exploits": [
            "[0.006 0.04  0.334 0.076 0.162 0.102 0.166 0.058 0.    0.056]",
            "[0.006 0.034 0.348 0.076 0.16  0.106 0.168 0.058 0.    0.044]",
            "[0.008 0.04  0.33  0.074 0.168 0.104 0.162 0.058 0.    0.056]",
            "[0.006 0.034 0.344 0.076 0.16  0.106 0.17  0.058 0.    0.046]",
            "[0.006 0.034 0.344 0.076 0.16  0.106 0.17  0.058 0.    0.046]"
        ],
        "Analysis": [
            "[0.008 0.04  0.544 0.098 0.096 0.07  0.108 0.016 0.    0.02 ]",
            "[0.008 0.04  0.544 0.098 0.096 0.07  0.108 0.016 0.    0.02 ]",
            "[0.008 0.04  0.544 0.098 0.096 0.07  0.108 0.016 0.    0.02 ]",
            "[0.008 0.04  0.544 0.098 0.096 0.07  0.108 0.016 0.    0.02 ]",
            "[0.008 0.04  0.544 0.098 0.096 0.07  0.108 0.016 0.    0.02 ]"
        ],
        "Fuzzers": [
            "[0.008 0.062 0.334 0.078 0.142 0.096 0.166 0.062 0.002 0.05 ]",
            "[0.008 0.054 0.338 0.086 0.142 0.094 0.172 0.064 0.004 0.038]",
            "[0.008 0.062 0.33  0.076 0.148 0.096 0.166 0.062 0.002 0.05 ]",
            "[0.008 0.06  0.342 0.086 0.128 0.1   0.168 0.062 0.002 0.044]",
            "[0.008 0.062 0.33  0.076 0.148 0.096 0.166 0.062 0.002 0.05 ]"
        ],
        "Reconnaissance": [
            "[0.004 0.034 0.538 0.08  0.098 0.064 0.124 0.028 0.    0.03 ]",
            "[0.004 0.034 0.538 0.086 0.094 0.064 0.126 0.028 0.    0.026]",
            "[0.004 0.034 0.538 0.08  0.098 0.064 0.124 0.028 0.    0.03 ]",
            "[0.004 0.04  0.522 0.078 0.108 0.066 0.122 0.028 0.    0.032]",
            "[0.004 0.034 0.538 0.086 0.094 0.064 0.126 0.028 0.    0.026]"
        ],
        "Shellcode": [
            "[0.006 0.042 0.548 0.086 0.104 0.054 0.098 0.036 0.004 0.022]",
            "[0.006 0.042 0.548 0.086 0.104 0.054 0.098 0.036 0.004 0.022]",
            "[0.006 0.042 0.546 0.086 0.106 0.054 0.1   0.036 0.004 0.02 ]",
            "[0.006 0.042 0.548 0.086 0.104 0.054 0.098 0.036 0.004 0.022]",
            "[0.006 0.042 0.546 0.086 0.106 0.054 0.1   0.036 0.004 0.02 ]"
        ]
    }
    ```