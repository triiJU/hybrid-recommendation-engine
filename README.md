# Hybrid Recommendation Engine

## Overview
A modular recommendation system combining:
- Popularity baseline
- Collaborative Filtering (user–user & item–item cosine)
- Content-Based ranking (TF‑IDF over item metadata)
- Neural Collaborative Filtering (PyTorch embeddings + MLP)
- Hybrid blending (alpha-weighted score fusion)
- Hyperparameter tuning (Optuna) and blending (alpha sweep)
- Cold-start performance analysis (sparse users/items)
- Ranking metrics: Precision@K, Recall@K, NDCG@K, Coverage, Diversity

Dataset: MovieLens 100K (explicit ratings)

## Motivation
Different paradigms capture different signal types. CF leverages interaction patterns, content models handle new items, neural embeddings model nonlinear relationships. Blending them increases relevance and robustness under sparsity.

## Key Features
| Module | Description |
|--------|-------------|
| Popularity | Global baseline using interaction count + mean rating |
| CF Baseline | User–user & item–item cosine similarity |
| Content-Based | TF‑IDF representation over item titles (extensible) |
| Neural CF | Embedding + MLP trained to regress ratings |
| Hybrid | Alpha-weighted combination of CF + content (optionally extend to neural) |
| Metrics | Precision@K, Recall@K, NDCG, coverage, diversity |
| Cold Start | Separate evaluation for low-interaction users/items |
| Tuning | Optuna search (CF neighbor count, mode) + alpha sweep script |

## Directory Layout
(As previously implemented.)

## Quick Start
(As previously implemented: make download → make preprocess → etc.)

## Example Metrics (Placeholder)
(Update with actual runs.)
| Model | P@10 | R@10 | NDCG@10 | Coverage | Diversity |
|-------|------|------|---------|----------|-----------|
| Popularity | 0.18 | 0.09 | 0.11 | 0.04 | 0.21 |
| User-CF | 0.27 | 0.14 | 0.21 | 0.33 | 0.37 |
| Item-CF | 0.26 | 0.13 | 0.20 | 0.29 | 0.35 |
| Content | 0.19 | 0.10 | 0.15 | 0.41 | 0.49 |
| Neural CF | 0.29 | 0.16 | 0.23 | 0.36 | 0.39 |
| Hybrid (α=0.7) | 0.31 | 0.17 | 0.24 | 0.44 | 0.46 |

## Cold-Start Example
| Segment | P@10 CF | P@10 Hybrid |
|---------|---------|-------------|
| New Users | 0.09 | 0.19 |
| All Users | 0.27 | 0.31 |

## Future Extensions
- Transformer text embeddings
- Implicit feedback & pairwise ranking (BPR)
- ANN vector retrieval (Faiss)
- MLflow logging
- Meta-learned blending

## License
MIT

## Dataset Citation
MovieLens datasets © GroupLens Research Group.
