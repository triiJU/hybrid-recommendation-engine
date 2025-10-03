# Hybrid Recommendation Engine

## Overview
A modular recommendation system that combines multiple paradigms to improve ranking quality and robustness:

- Popularity baseline
- Collaborative Filtering (user–user & item–item cosine)
- Content-Based Ranking (TF‑IDF over item metadata)
- Neural Collaborative Filtering (PyTorch embeddings + MLP)
- Hybrid Blending (alpha‑weighted combination)
- Hyperparameter & blending optimization (alpha sweep, Optuna)
- Cold‑start analysis (new / sparse users & items)
- Ranking Metrics: Precision@K, Recall@K, NDCG@K, Coverage, Diversity

Primary dataset: MovieLens 100K (explicit ratings).

## Why a Hybrid?
Each method has a weakness:
- Pure CF: Struggles with sparse users/items (cold start).
- Content-only: Ignores collaborative interaction structure.
- Neural CF: Better representation learning but still sparse-dependent.

Blending CF + Content + Neural embeddings improves:
- Relevance (higher NDCG / Precision@K)
- Coverage (more catalog exposure)
- Cold-start performance (content fallback)

## Features & Modules

| Capability | File / Script | Notes |
|------------|---------------|-------|
| Download dataset | `src/data_loading.py` | Fetch & unzip MovieLens 100K |
| Preprocess & split | `src/preprocess.py` | Filter low-interaction users, train/test |
| Popularity baseline | `src/popularity.py` | Global frequency + mean rating score |
| Collaborative Filtering | `src/cf_baseline.py` | User–user / item–item cosine similarity |
| Content Modeling (TF‑IDF) | `src/content_based.py` | Title-based sparse vectors |
| Neural CF | `src/neural_cf.py` | Embedding + MLP (PyTorch) |
| Hybrid blending | `src/hybrid.py` | Alpha‑weighted CF + Content score |
| Metrics | `src/metrics.py` | Precision@K, Recall@K, NDCG, Coverage, Diversity |
| Unified evaluation | `src/evaluation.py` | Runs & aggregates baselines |
| Alpha sweep | `scripts/alpha_sweep.py` | Tests multiple α values |
| Cold-start evaluation | `src/cold_start.py` | Sparse user segment performance |
| Hyperparameter tuning | `src/optuna_tune.py` | Mode + neighbor count search |
| Logging | `src/logging_config.py` | Structured logging to file/console |
| End-to-end pipeline | `scripts/run_pipeline.sh` | One‑shot reproducible run |
| Make targets | `Makefile` | Declarative workflow |

## Directory Structure
```
.
├── README.md
├── LICENSE
├── Makefile
├── requirements.txt
├── models/
├── experiments/
├── scripts/
│   ├── run_pipeline.sh
│   └── alpha_sweep.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_config.py
│   ├── utils.py
│   ├── data_loading.py
│   ├── preprocess.py
│   ├── popularity.py
│   ├── metrics.py
│   ├── cf_baseline.py
│   ├── content_based.py
│   ├── neural_cf.py
│   ├── hybrid.py
│   ├── evaluation.py
│   ├── cold_start.py
│   └── optuna_tune.py
└── tests/
    └── test_metrics.py
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start
```bash
make download
make preprocess
make baselines
make neural
make hybrid
make evaluate
make alpha
make coldstart
```

## Individual Commands (Manual)
```bash
python -m src.data_loading --download_movielens
python -m src.preprocess --ratings_path data/ml-100k/u.data --out_dir data/processed --min_interactions 5
python -m src.popularity --train_path data/processed/train.csv
python -m src.cf_baseline --train_path data/processed/train.csv --mode user
python -m src.neural_cf --train_path data/processed/train.csv --epochs 10
python -m src.hybrid --alpha 0.7 --train_path data/processed/train.csv --test_path data/processed/test.csv --items_path data/ml-100k/u.item
python -m src.evaluation --train_path data/processed/train.csv --test_path data/processed/test.csv --k 10
python -m src.cold_start --train_path data/processed/train.csv --test_path data/processed/test.csv --k 10
```

## Example (Placeholder) Metrics
Replace these with real results after your run.

| Model                 | P@10 | R@10 | NDCG@10 | Coverage | Diversity |
|-----------------------|------|------|---------|----------|-----------|
| Popularity            | 0.18 | 0.09 | 0.11    | 0.04     | 0.21      |
| User-CF               | 0.27 | 0.14 | 0.21    | 0.33     | 0.37      |
| Item-CF               | 0.26 | 0.13 | 0.20    | 0.29     | 0.35      |
| Content (TF‑IDF)      | 0.19 | 0.10 | 0.15    | 0.41     | 0.49      |
| Neural CF             | 0.29 | 0.16 | 0.23    | 0.36     | 0.39      |
| Hybrid (α=0.7)        | 0.31 | 0.17 | 0.24    | 0.44     | 0.46      |
| Hybrid + Neural Blend | 0.32 | 0.18 | 0.25    | 0.45     | 0.45      |

## Alpha Sweep
```bash
python scripts/alpha_sweep.py --alphas 0.3 0.5 0.7 0.9 --k 10
# Produces: experiments/alpha_sweep.json
```

## Cold-Start Snapshot (Example)
| Segment | P@10 (CF) | P@10 (Hybrid) |
|---------|-----------|---------------|
| New Users (≤4 ratings) | 0.09 | 0.19 |
| All Users (sample)     | 0.27 | 0.31 |

## Hyperparameter Tuning (Optuna)
```bash
python -m src.optuna_tune --trials 30 --k 10
# Optimizes mode (user/item) + neighbor count
```

## Testing
```bash
pytest -q
```

## Reproducibility & Artifacts
Generated after runs:
```
experiments/
  results.json          # Baseline metrics
  hybrid_result.json    # Latest hybrid metrics
  cold_start.json       # Cold-start segment metrics
  alpha_sweep.json      # Alpha tuning results
  run.log               # Log output (if logging enabled)
models/
  neural_cf.pt          # Saved neural CF model
```

## Scaling & Production Notes (Interview Talking Points)
- ANN (Faiss) for candidate retrieval (replaces brute-force cosine)
- Hybrid pipeline: candidate generation (popularity + CF + content) → neural re-ranker
- Embedding freshness: periodic offline refresh + on-demand user vector updates
- Cold-start mitigation: content-based fallback + side features (e.g., item metadata embeddings)
- Future: move to implicit feedback & pairwise ranking (BPR / sampled softmax)

## Future Enhancements
- Transformer-based text embeddings
- Visual embeddings (if extending to image-rich products)
- MLflow tracking for experiments
- Meta-learner stacking to blend sources
- A/B simulation harness

## License
MIT

## Dataset Citation
MovieLens datasets: © GroupLens Research Group.
