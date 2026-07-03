make prepare DATA=nb15_v2 NAME=topk_inference_kmeans2 CLASSIFIER=random_forest CLUSTERING=kmeans

make complexity DATA=nb15_v2 NAME=topk_inference_kmeans2 CLASSIFIER=random_forest CLUSTERING=kmeans EXTEND=1

make classify DATA=nb15_v2 NAME=topk_inference_kmeans2 CLASSIFIER=random_forest CLUSTERING=kmeans

make classify-extended DATA=nb15_v2 NAME=topk_inference_kmeans2 CLASSIFIER=random_forest CLUSTERING=kmeans

# make topk-inference DATA=nb15_v2 NAME=topk_inference_kmeans2 CLASSIFIER=random_forest CLUSTERING=kmeans