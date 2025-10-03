import argparse, pandas as pd, numpy as np, json
from .cf_baseline import CollaborativeFiltering
from .metrics import precision_recall_at_k, ndcg_at_k, coverage, diversity
from .content_based import ContentModel, load_items_file
from .config import Config
from .popularity import popularity_rank

def evaluate_all(train, test, items_df, k):
    results={}
    # Popularity
    pop = popularity_rank(train, top_n=10000)
    users = test.user_id.unique()
    users_sample = np.random.choice(users, size=min(150, len(users)), replace=False)
    pop_recs=[]; pop_prec=[]; pop_ndcg=[]
    for u in users_sample:
        gt = test[test.user_id==u].item_id.tolist()
        rec_ids = pop.item_id.head(k).tolist()
        pop_recs.append(rec_ids)
        p,_ = precision_recall_at_k(rec_ids, gt, k)
        n = ndcg_at_k(rec_ids, gt, k)
        pop_prec.append(p); pop_ndcg.append(n)
    results["popularity"] = {
        "precision_at_k": float(np.mean(pop_prec)),
        "ndcg_at_k": float(np.mean(pop_ndcg))
    }

    # CF user
    cf_user = CollaborativeFiltering(mode="user"); cf_user.fit(train)
    cf_prec=[]; cf_ndcg=[]; cf_recs=[]
    for u in users_sample:
        gt = test[test.user_id==u].item_id.tolist()
        recs = cf_user.recommend(u, n=k)
        ids=[r[0] for r in recs]
        cf_recs.append(ids)
        p,_ = precision_recall_at_k(ids, gt, k)
        n = ndcg_at_k(ids, gt, k)
        cf_prec.append(p); cf_ndcg.append(n)
    results["user_cf"] = {
        "precision_at_k": float(np.mean(cf_prec)),
        "ndcg_at_k": float(np.mean(cf_ndcg)),
        "coverage": coverage(cf_recs, cf_user.utility.shape[1])
    }

    # Content (simple anchor-based similarity using last rated item)
    cm = ContentModel(); cm.fit(items_df, text_col="title")
    content_prec=[]; content_ndcg=[]
    for u in users_sample:
        hist = train[train.user_id==u].sort_values("timestamp")
        if hist.empty:
            continue
        anchor = hist.item_id.iloc[-1]
        sims = cm.similar_items(anchor, top_n=k+5)
        rec_ids = [iid for iid,_ in sims if iid != anchor][:k]
        gt = test[test.user_id==u].item_id.tolist()
        p,_ = precision_recall_at_k(rec_ids, gt, k)
        n = ndcg_at_k(rec_ids, gt, k)
        content_prec.append(p); content_ndcg.append(n)
    results["content"] = {
        "precision_at_k": float(np.mean(content_prec)),
        "ndcg_at_k": float(np.mean(content_ndcg))
    }

    # Diversity (user CF recommendations)
    tfidf_mat = cm.matrix.toarray()
    id_to_index = {row.item_id: idx for idx,row in cm.items_df.reset_index().iterrows()}
    div_scores=[]
    for rec in cf_recs:
        div_scores.append(diversity(tfidf_mat, rec, id_to_index))
    results["user_cf"]["diversity"] = float(np.mean(div_scores))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/train.csv")
    parser.add_argument("--test_path", default="data/processed/test.csv")
    parser.add_argument("--items_path", default="data/ml-100k/u.item")
    parser.add_argument("--k", type=int, default=Config.TOP_K)
    args = parser.parse_args()

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    items_df = load_items_file(args.items_path)

    res = evaluate_all(train, test, items_df, args.k)
    print(res)
    with open("experiments/results.json","w") as f:
        json.dump(res,f,indent=2)