import math, numpy as np
from itertools import combinations

def precision_recall_at_k(recommended, ground_truth, k):
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(ground_truth))
    precision = hits / k if k else 0
    recall = hits / len(ground_truth) if ground_truth else 0
    return precision, recall

def ndcg_at_k(recommended, ground_truth, k):
    rec_k = recommended[:k]
    dcg=0.0
    for i,item in enumerate(rec_k):
        if item in ground_truth:
            dcg += 1 / math.log2(i+2)
    ideal = min(len(ground_truth), k)
    idcg = sum(1 / math.log2(i+2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0

def coverage(all_recommendations, total_items):
    unique = set()
    for recs in all_recommendations:
        unique.update(recs)
    return len(unique)/total_items if total_items else 0.0

def diversity(item_embeddings, recommended_ids, id_to_index):
    if len(recommended_ids) < 2:
        return 0.0
    from numpy.linalg import norm
    import numpy as np
    def cosine(a,b): return float(a @ b / (norm(a)*norm(b) + 1e-8))
    vecs = [item_embeddings[id_to_index[i]] for i in recommended_ids if i in id_to_index]
    if len(vecs) < 2: return 0.0
    dists=[]
    for a,b in combinations(range(len(vecs)),2):
        dists.append(1 - cosine(vecs[a], vecs[b]))
    return float(np.mean(dists)) if dists else 0.0