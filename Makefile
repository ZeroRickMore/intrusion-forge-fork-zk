# ──────────────────────────────────────────────────────────────────────────────
# Intrusion Forge — Experiment Runner
#
# One parametric command for sweeps. Variables passed on the command line are
# FIXED; those omitted are ITERATED.
#
#   make run            NAME=my_exp                                # all datasets × ML+DL × all clustering algos
#   make run            NAME=my_exp DATA=letter_recognition        # 1 dataset × all compatible classifiers × all clustering
#   make run            NAME=my_exp CLASSIFIER=random_forest       # all datasets × 1 classifier × all clustering
#   make run            NAME=my_exp CLUSTERING=kmeans              # all datasets × all classifiers × 1 clustering
#   make run            NAME=my_exp DATA=cic_2018_v2 CLASSIFIER=tabular CLUSTERING=kmeans   # single (ds, clf, clustering)
#
# Single-stage targets (DATA + CLASSIFIER explicit):
#   make prepare           DATA=cic_2018_v2 NAME=my_exp
#   make classify          DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make complexity        DATA=cic_2018_v2 NAME=my_exp                     # shared, dataset-level
#   make failure-classify  DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make render            DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=random_forest
#   make extend            DATA=cic_2018_v2 NAME=my_exp CLASSIFIER=tabular  # complexity + classify-extended + render
#
# Flags:
#   FORCE=1               re-run shared stages (prepare, complexity), ignoring skip markers
#   EXTEND=1              in `run`, adds classify-extended (SHAP) to the flow for every (ds, clf)
#   LABELFREE=1           build the extended splits with label-free nearest-centroid assignment
#                         (injection honesty control; pair with EXTEND=1 for the full flow)
#   CLUSTERING=<name>     fix the clustering strategy (ensemble/kmeans/hdbscan/birch/spectral/kprototypes);
#                         omit it in `run` to sweep all of CLUSTERING_ALGOS into NAME_<algo>
#
# k-fold note: k-fold evaluation (kfold=true) is disabled automatically for LARGE_DATASETS
#   (nb15_v2, bot_iot_v2, cic_2018_v2, ton_iot_v2) because millions of rows make it impractical.
#   Override per-call: make classify DATA=cic_2018_v2 ... kfold=true
# ──────────────────────────────────────────────────────────────────────────────

# Use venv if present; falls back to the active conda (or system) python otherwise.
# Override explicitly: make <target> PYTHON=python
PYTHON    ?= $(if $(wildcard venv/bin/python),venv/bin/python,python)
STREAMLIT ?= $(if $(wildcard venv/bin/streamlit),venv/bin/streamlit,streamlit)
DATA       ?= cic_2018_v2
NAME       ?= exp_euc
SEED       ?= 42
CLASSIFIER ?= tabular
DISTANCE   ?= euclidean
CLUSTERING ?= kmeans
CLUSTERING_ALGOS ?= kmeans hdbscan spectral birch kprototypes
FORCE      ?=
EXTEND     ?=
export EXTEND
LABELFREE  ?=
export LABELFREE

# `run` distinguishes "passed on the command line" from "default" via $(origin).
DATA_GIVEN    := $(if $(filter command line,$(origin DATA)),1,)
CLF_GIVEN     := $(if $(filter command line,$(origin CLASSIFIER)),1,)
CLUSTER_GIVEN := $(if $(filter command line,$(origin CLUSTERING)),1,)

ML_CLASSIFIERS := \
    naive_bayes \
    logistic_regression \
    lda \
    knn \
    decision_tree \
    random_forest \
    hist_gradient_boosting \
    linear_svc \
    xgboost

DL_CLASSIFIERS_MIXED     := tabular
DL_CLASSIFIERS_NUMERICAL := numerical

DATASET_FORMATS := \
    statlog_landsat_satellite:numerical \
    thyroid_disease:numerical \
    letter_recognition:numerical \
    bank_marketing:mixed \
    covertype:numerical \
    nb15_v2:mixed \
    ton_iot_v2:mixed \
    cic_2018_v2:mixed \
    bot_iot_v2:mixed \
#    synthetic_test:mixed

# Datasets too large for k-fold evaluation (millions of rows → hours per classifier).
# kfold=false is injected automatically for these; override with kfold=true if needed.
LARGE_DATASETS := nb15_v2 bot_iot_v2 cic_2018_v2 ton_iot_v2

HYDRA       := data=$(DATA) name=$(NAME) seed=$(SEED) classifier=$(CLASSIFIER) \
               clustering=$(CLUSTERING) distance=$(DISTANCE)
FORCE_FLAG  := $(if $(FORCE),prepare.force=true complexity.force=true,)
EXTEND_FLAGS := $(if $(LABELFREE),extend.generate=true extend.labelfree=true,$(if $(EXTEND),extend.generate=true,))
KFOLD_FLAG  := $(if $(filter $(DATA),$(LARGE_DATASETS)),kfold=false,)

# Cost analysis. `cost-model` characterises the k-NN build cost for one dataset over both
# distances × COST_SEEDS: α (≈2, Θ(m²)) gets a confidence interval across seeds, while c
# contrasts cosine vs euclidean (~4×). The aggregator scans EXPERIMENTS_DIR.
COST_CLF        ?= random_forest
COST_SEEDS      ?= 42 0 1
EXPERIMENTS_DIR ?= resources/experiments

.PHONY: prepare classify classify-extended extend complexity failure-classify render run cost-model cost-summary cost-aggregate generate dashboard help

## prepare:            Step 1 — preprocess raw CSV → parquet splits           (DATA, NAME, SEED, FORCE)
prepare:
	PYTHONPATH=. $(PYTHON) pipelines/prepare_data.py $(HYDRA) $(FORCE_FLAG)

## classify:           Step 2 — train & evaluate one classifier (ML or DL)    (DATA, NAME, SEED, CLASSIFIER)
classify:
	PYTHONPATH=. $(PYTHON) pipelines/classify.py $(HYDRA) $(KFOLD_FLAG)

## classify-extended:  Step 2b — train classifier on complexity-extended features + SHAP  (DATA, NAME, SEED, CLASSIFIER)
classify-extended:
	PYTHONPATH=. $(PYTHON) pipelines/classify.py $(HYDRA) extend.generate=true

## extend:             complexity + classify-extended + render (assumes prepare + classify done)  (DATA, NAME, SEED, CLASSIFIER)
extend:
	PYTHONPATH=. $(PYTHON) pipelines/compute_complexity.py $(HYDRA) $(FORCE_FLAG) extend.generate=true
	PYTHONPATH=. $(PYTHON) pipelines/classify.py $(HYDRA) extend.generate=true
	PYTHONPATH=. $(PYTHON) pipelines/render_plots.py $(HYDRA)

## extend-lf:          label-free extended: nearest-centroid assignment, ignores class labels    (DATA, NAME, SEED, CLASSIFIER)
extend-lf:
	PYTHONPATH=. $(PYTHON) pipelines/compute_complexity.py $(HYDRA) $(FORCE_FLAG) extend.generate=true extend.labelfree=true
	PYTHONPATH=. $(PYTHON) pipelines/classify.py $(HYDRA) extend.generate=true
	PYTHONPATH=. $(PYTHON) pipelines/render_plots.py $(HYDRA)

## complexity:         Step 3a — cluster + class complexity (shared, idempotent)  (DATA, NAME, SEED, FORCE, LABELFREE)
complexity:
	PYTHONPATH=. $(PYTHON) pipelines/compute_complexity.py $(HYDRA) $(FORCE_FLAG) $(EXTEND_FLAGS)

## failure-classify:   Step 3b — RF to detect problematic clusters            (DATA, NAME, SEED, CLASSIFIER)
failure-classify: complexity
	PYTHONPATH=. $(PYTHON) pipelines/fit_failure_classifier.py $(HYDRA)

## render:             Step 4 — render plots from analysis artifacts          (DATA, NAME, SEED, CLASSIFIER)
render:
	PYTHONPATH=. $(PYTHON) pipelines/render_plots.py $(HYDRA)

## run:                Whole flow — fix passed vars, iterate the rest (DATA?, CLASSIFIER?, CLUSTERING?)  (NAME, SEED, DISTANCE, FORCE, EXTEND)
run:
	@data_given="$(DATA_GIVEN)"; \
	clf_given="$(CLF_GIVEN)"; \
	clu_given="$(CLUSTER_GIVEN)"; \
	requested_data="$(DATA)"; \
	requested_clf="$(CLASSIFIER)"; \
	if [ -n "$$clu_given" ]; then clu_list="$(CLUSTERING)"; else clu_list="$(CLUSTERING_ALGOS)"; fi; \
	if [ -n "$$data_given" ]; then \
		pairs=""; \
		for entry in $(DATASET_FORMATS); do \
			ds=$${entry%%:*}; \
			if [ "$$ds" = "$$requested_data" ]; then pairs="$$entry"; break; fi; \
		done; \
		if [ -z "$$pairs" ]; then \
			echo "ERROR: DATA='$$requested_data' not in DATASET_FORMATS."; exit 1; \
		fi; \
	else \
		pairs="$(DATASET_FORMATS)"; \
	fi; \
	for clu in $$clu_list; do \
		if [ -n "$$clu_given" ]; then name="$(NAME)"; else name="$(NAME)_$$clu"; fi; \
		echo ""; \
		echo "##############################################"; \
		echo " CLUSTERING = $$clu   →   name=$$name"; \
		echo "##############################################"; \
		for entry in $$pairs; do \
			ds=$${entry%%:*}; fmt=$${entry##*:}; \
			if [ -n "$$clf_given" ]; then \
				skip=""; \
				if [ "$$requested_clf" = "tabular" ]   && [ "$$fmt" != "mixed" ];     then skip=1; fi; \
				if [ "$$requested_clf" = "numerical" ] && [ "$$fmt" != "numerical" ]; then skip=1; fi; \
				if [ -n "$$skip" ]; then \
					echo "skip: $$requested_clf not compatible with $$ds ($$fmt)"; \
					continue; \
				fi; \
				clf_list="$$requested_clf"; \
			else \
				if [ "$$fmt" = "mixed" ]; then \
					clf_list="$(ML_CLASSIFIERS) $(DL_CLASSIFIERS_MIXED)"; \
				else \
					clf_list="$(ML_CLASSIFIERS) $(DL_CLASSIFIERS_NUMERICAL)"; \
				fi; \
			fi; \
			echo ""; \
			echo "══════════════════════════════════════════════"; \
			echo " Dataset: $$ds  |  format=$$fmt  |  name=$$name  seed=$(SEED)"; \
			echo "══════════════════════════════════════════════"; \
			$(MAKE) --no-print-directory prepare \
				DATA=$$ds NAME=$$name SEED=$(SEED) CLUSTERING=$$clu \
				DISTANCE=$(DISTANCE) $(FORCE_FLAG) || exit 1; \
			$(MAKE) --no-print-directory complexity \
				DATA=$$ds NAME=$$name SEED=$(SEED) CLUSTERING=$$clu \
				DISTANCE=$(DISTANCE) $(FORCE_FLAG) || exit 1; \
			for clf in $$clf_list; do \
				echo ""; \
				echo "── classifier: $$clf ─────────────────────────────"; \
				$(MAKE) --no-print-directory classify \
					DATA=$$ds NAME=$$name SEED=$(SEED) CLASSIFIER=$$clf \
					CLUSTERING=$$clu DISTANCE=$(DISTANCE) || exit 1; \
				$(MAKE) --no-print-directory failure-classify \
					DATA=$$ds NAME=$$name SEED=$(SEED) CLASSIFIER=$$clf \
					CLUSTERING=$$clu DISTANCE=$(DISTANCE) || exit 1; \
				if [ -n "$(EXTEND)" ]; then \
					$(MAKE) --no-print-directory classify-extended \
						DATA=$$ds NAME=$$name SEED=$(SEED) CLASSIFIER=$$clf \
						CLUSTERING=$$clu DISTANCE=$(DISTANCE) || exit 1; \
				fi; \
				$(MAKE) --no-print-directory render \
					DATA=$$ds NAME=$$name SEED=$(SEED) CLASSIFIER=$$clf \
					CLUSTERING=$$clu DISTANCE=$(DISTANCE) || exit 1; \
			done; \
		done; \
	done
	@echo ""
	@echo "Done."

## cost-model:         Deliverable A — k-NN build cost model T(m)=c·m^α over 2 distances × COST_SEEDS for one dataset  (DATA, NAME, COST_SEEDS, COST_CLF)
cost-model:
	@for dist in cosine euclidean; do \
	  for seed in $(COST_SEEDS); do \
	    nm=$(NAME)_$$dist; \
	    echo ""; echo "== cost-model: $(DATA) [$$dist] seed=$$seed -> name=$$nm =="; \
	    $(MAKE) --no-print-directory prepare \
	      DATA=$(DATA) NAME=$$nm SEED=$$seed CLUSTERING=kmeans DISTANCE=$$dist || exit 1; \
	    $(MAKE) --no-print-directory classify \
	      DATA=$(DATA) NAME=$$nm SEED=$$seed CLASSIFIER=$(COST_CLF) CLUSTERING=kmeans DISTANCE=$$dist || exit 1; \
	    PYTHONPATH=. $(PYTHON) pipelines/cost_sweep.py \
	      data=$(DATA) name=$$nm seed=$$seed classifier=$(COST_CLF) \
	      clustering=kmeans distance=$$dist || exit 1; \
	  done; \
	done
	@echo ""; echo "cost-model done -> <NAME>_<distance>/$(DATA)_<seed>/shared/cost_model.json (α across COST_SEEDS; c cosine vs euclidean)."

## cost-summary:       Roll up cost_model.json files: α mean±std per distance + c euclidean/cosine ratio  (EXPERIMENTS_DIR)
cost-summary:
	@PYTHONPATH=. $(PYTHON) pipelines/cost_sweep.py summary=$(EXPERIMENTS_DIR)
	@echo ""; echo "cost-summary done -> $(EXPERIMENTS_DIR)/cost_model_summary.json."

## cost-aggregate:     Deliverable B — cross-run cost↔quality table (ρ vs cluster count vs build time) from finished runs  (EXPERIMENTS_DIR)
cost-aggregate:
	@PYTHONPATH=. $(PYTHON) pipelines/cost_sweep.py aggregate=$(EXPERIMENTS_DIR)
	@echo ""; echo "cost-aggregate done -> $(EXPERIMENTS_DIR)/cost_quality_table.json."

## generate:           Generate synthetic test dataset                        (ROWS)
generate:
	$(PYTHON) generate_synthetic.py $(if $(ROWS),--rows $(ROWS),)

## dashboard:          Open the experiment dashboard in browser
dashboard:
	$(STREAMLIT) run dashboard.py

## help:               Show this help message
help:
	@echo "Usage: make <target> [DATA=<dataset>] [NAME=<name>] [SEED=<n>] [CLASSIFIER=<name>] [DISTANCE=<dist>]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Defaults:  DATA=$(DATA)  NAME=$(NAME)  SEED=$(SEED)  CLASSIFIER=$(CLASSIFIER)  DISTANCE=$(DISTANCE)  CLUSTERING=$(CLUSTERING)"
	@echo "Python:    $(PYTHON)  (override with PYTHON=)"
	@echo "ML classifiers:         $(ML_CLASSIFIERS)"
	@echo "DL classifiers (mixed): $(DL_CLASSIFIERS_MIXED)"
	@echo "DL classifiers (num):   $(DL_CLASSIFIERS_NUMERICAL)"
	@echo "Clustering strategies:  ensemble kmeans hdbscan birch spectral kprototypes"
	@echo ""
	@echo "Datasets (smallest → largest, kfold auto-disabled for large):"
	@echo "  small (kfold=true):   statlog_landsat_satellite  thyroid_disease  letter_recognition  bank_marketing  covertype"
	@echo "  large (kfold=false):  nb15_v2  ton_iot_v2  cic_2018_v2  bot_iot_v2"
	@echo ""
	@echo "Run examples (omitted vars iterate; passed vars are fixed):"
	@echo "  make run NAME=x                                      # all datasets × all classifiers × all clustering algos"
	@echo "  make run NAME=x DATA=letter_recognition              # 1 dataset, all compatible classifiers, all clustering"
	@echo "  make run NAME=x CLASSIFIER=random_forest             # all datasets, 1 classifier, all clustering"
	@echo "  make run NAME=x CLUSTERING=kmeans                    # all datasets × all classifiers, 1 clustering"
	@echo "  make run NAME=x DATA=cic_2018_v2 CLASSIFIER=tabular CLUSTERING=kmeans  # single"
	@echo "  (clustering swept → artifacts land under NAME_<algo>; clustering fixed → under NAME)"
