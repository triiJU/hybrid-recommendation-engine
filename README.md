# Hybrid Recommendation Engine

## Overview
A modular hybrid recommender system supporting:
- Popularity baseline
- Collaborative Filtering (user–user & item–item cosine)
- Content-Based Ranking (TF-IDF over item metadata)
- Neural Collaborative Filtering (PyTorch embeddings + MLP)
- Hybrid Blending (tri-weight: CF, content, neural)
- Alpha sweep & Optuna hyperparameter tuning
- Cold-start evaluation (new / sparse users & items)
- Ranking Metrics: Precision@K, Recall@K, NDCG@K, Coverage, Item Diversity
- Reproducible pipeline (Makefile + scripts)

## Why Hybrid?
Pure collaborative filtering struggles with cold-start and sparse data. Content-based models generalize to new items but lack deep personalization. Neural CF captures non-linear interactions. The hybrid blends these to raise NDCG and coverage while mitigating cold-start performance loss.

## Features
| Component | File | Description |
|-----------|------|-------------|
| Data download | `src/data_loading.py` | MovieLens 100K fetch & extract |
| Preprocessing | `src/preprocess.py` | Filtering low-interaction users, train/test split |
| Popularity baseline | `src/popularity.py` | Global item ranking |
| Collaborative Filtering | `src/cf_baseline.py` | User–User / Item–Item cosine similarity |
| Content-based | `src/content_based.py` | TF-IDF item embeddings & similarity |
| Neural CF | `src/neural_cf.py` | Embeddings + MLP (explicit ratings regression) |
| Hybrid blend | `src/hybrid.py` | Tri-weight combination of CF + content + neural |
| Metrics | `src/metrics.py` | Precision@K, Recall@K, NDCG@K, Coverage, Diversity |
| Evaluation | `src/evaluation.py` | Unified baseline evaluation |
| Alpha sweep | `scripts/alpha_sweep.py` | Evaluate multiple α values |
| Optuna tuning | `src/optuna_tune.py` | Optimize CF neighbor count / mode |
| Cold-start eval | `src/cold_start.py` | Segment sparse users/items |
| Logging | `src/logging_config.py` | Structured logging |
| Pipeline script | `scripts/run_pipeline.sh` | End-to-end automation |
| Make targets | `Makefile` | Reproducible commands |

## Metrics (Example / Placeholder)
| Model                 | P@10 | R@10 | NDCG@10 | Coverage | Diversity |
|-----------------------|------|------|---------|----------|-----------|
| Popularity            | 0.18 | 0.09 | 0.11    | 0.04     | 0.21      |
| User-CF               | 0.27 | 0.14 | 0.21    | 0.33     | 0.37      |
| Item-CF               | 0.26 | 0.13 | 0.20    | 0.29     | 0.35      |
| Content (TF-IDF)      | 0.19 | 0.10 | 0.15    | 0.41     | 0.49      |
| Neural CF             | 0.29 | 0.16 | 0.23    | 0.36     | 0.39      |
| Hybrid (CF=0.6, Content=0.4)        | 0.31 | 0.17 | 0.24    | 0.44     | 0.46      |
| Hybrid (CF=0.5, Content=0.3, Neural=0.2) | 0.32 | 0.18 | 0.25    | 0.45     | 0.45      |

(Replace with real outputs after running.)

## Cold-Start (Example)
| Segment | P@10 (User-CF) | P@10 (Content) | P@10 (Hybrid) |
|---------|----------------|----------------|---------------|
| New Users (≤4 ratings) | 0.09 | 0.15 | 0.19 |
| New Items (low exposure) | 0.06 | 0.14 | 0.17 |

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
└── src/
```

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

make download
make preprocess
make baselines
make neural
make hybrid
make evaluate
make alpha
make coldstart
```

## Example Hybrid Run

### Tri-Weight Blending
```bash
# CF + Content blend (no neural)
python -m src.hybrid --w_cf 0.6 --w_content 0.4 --w_neural 0.0 \
  --train_path data/processed/train.csv \
  --test_path data/processed/test.csv \
  --items_path data/ml-100k/u.item

# CF + Content + Neural blend (neural weight ignored if model not found)
python -m src.hybrid --w_cf 0.5 --w_content 0.3 --w_neural 0.2 \
  --train_path data/processed/train.csv \
  --test_path data/processed/test.csv \
  --items_path data/ml-100k/u.item \
  --neural_model_path models/neural_cf.pt
```

**Note:** If the neural model is not trained/available, the neural weight will be automatically set to 0 and weights will be renormalized.

## API Usage

The system provides a FastAPI-based REST API for serving recommendations:

```bash
# Start the API server
uvicorn api.app:app --reload

# Get recommendations with custom weights
curl "http://localhost:8000/recommend?user_id=1&k=10&w_cf=0.6&w_content=0.4&w_neural=0.0"

# Check API health
curl "http://localhost:8000/health"

# Get metadata
curl "http://localhost:8000/meta"
```

## Alpha Sweep
```bash
python scripts/alpha_sweep.py --alphas 0.3 0.5 0.7 0.9 --k 10
```

## Optuna Hyperparameter Tuning
```bash
python -m src.optuna_tune --trials 30
```

## Cold-Start Evaluation
```bash
python -m src.cold_start --train_path data/processed/train.csv \
  --test_path data/processed/test.csv --k 10
```

## Scalability (Interview Talking Points)
- Replace brute-force similarity with ANN (Faiss)
- Offline batch refresh + incremental updates
- Candidate generation → re-ranking pipeline
- Feature store for user/item embeddings

## Future Work
- Transformer text embeddings
- Implicit feedback (BPR / WARP)
- Meta-learning blend weights
- MLflow tracking
- A/B simulation harness

## License
MIT

## Citation
MovieLens datasets: © GroupLens Research Group.
