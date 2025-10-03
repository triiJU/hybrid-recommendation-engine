import argparse, json, numpy as np, pandas as pd
from src.cf_baseline import CollaborativeFiltering
from src.content_based import ContentModel, load_items_file
from src.metrics import precision_recall_at_k, ndcg_at_k

def alpha_sweep(train_path, test_path, items_path, alphas, k):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    cf = CollaborativeFiltering(mode="user")
    cf.fit(train)
    items_df = load_items_file(items_path)
    cm = ContentModel()
    cm.fit(items_df, text_col="title")

    users = test.user_id.unique()
    users = np.random.choice(users, size=min(120, len(users)), replace=False)

    results=[]
    for alpha in alphas:
        precisions=[]; ndcgs=[]
        for u in users:
            gt = test[test.user_id==u].item_id.tolist()
            recs = cm.hybrid_with_cf(u, cf, alpha=alpha, top_n=k)
            rec_ids = [r[0] for r in recs]
            p,_ = precision_recall_at_k(rec_ids, gt, k)
            n = ndcg_at_k(rec_ids, gt, k)
            precisions.append(p); ndcgs.append(n)
        results.append({
            "alpha": alpha,
            "precision_at_k": float(np.mean(precisions)),
            "ndcg_at_k": float(np.mean(ndcgs))
        })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/train.csv")
    parser.add_argument("--test_path", default="data/processed/test.csv")
    parser.add_argument("--items_path", default="data/ml-100k/u.item")
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    res = alpha_sweep(args.train_path, args.test_path, args.items_path, args.alphas, args.k)
    print(res)
    with open("experiments/alpha_sweep.json","w") as f:
        json.dump(res,f,indent=2)