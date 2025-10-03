PY=python
DATA_DIR=data
PROC_DIR=$(DATA_DIR)/processed

download:
	$(PY) -m src.data_loading --download_movielens

preprocess:
	$(PY) -m src.preprocess --ratings_path $(DATA_DIR)/ml-100k/u.data --out_dir $(PROC_DIR) --min_interactions 5

baselines: popularity cf content

popularity:
	$(PY) -m src.popularity --train_path $(PROC_DIR)/train.csv

cf:
	$(PY) -m src.cf_baseline --train_path $(PROC_DIR)/train.csv --mode user
	$(PY) -m src.cf_baseline --train_path $(PROC_DIR)/train.csv --mode item

content:
	$(PY) -m src.content_based --items_path $(DATA_DIR)/ml-100k/u.item

neural:
	$(PY) -m src.neural_cf --train_path $(PROC_DIR)/train.csv --epochs 10

hybrid:
	$(PY) -m src.hybrid --w_cf 0.6 --w_content 0.4 --w_neural 0.0 --train_path $(PROC_DIR)/train.csv --test_path $(PROC_DIR)/test.csv --items_path $(DATA_DIR)/ml-100k/u.item

evaluate:
	$(PY) -m src.evaluation --train_path $(PROC_DIR)/train.csv --test_path $(PROC_DIR)/test.csv --k 10

alpha:
	$(PY) scripts/alpha_sweep.py --alphas 0.3 0.5 0.7 0.9 --k 10

tune:
	$(PY) -m src.optuna_tune --trials 25

coldstart:
	$(PY) -m src.cold_start --train_path $(PROC_DIR)/train.csv --test_path $(PROC_DIR)/test.csv --k 10

.PHONY: download preprocess baselines popularity cf content neural hybrid evaluate alpha tune coldstart
