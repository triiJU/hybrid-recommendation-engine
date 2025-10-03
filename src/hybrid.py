import argparse, json, pandas as pd
from .cf_baseline import CollaborativeFiltering
from .content_based import ContentModel, load_items_file
from .config import Config
from .metrics import precision_recall_at_k, ndcg_at_k

def evaluate_hybrid(cf, cm, test_df, alpha, k):
    users = test_df.user_id.unique()
    import numpy as np
    users = np.random.choice(users, size=min(150, len(users)), replace=False)
    precisions=[]; ndcgs=[]
    for u in users:
        gt = test_df[test_df.user_id==u].item_id.tolist()
        recs = cm.hybrid_with_cf(u, cf, alpha=alpha, top_n=k)
        ids = [r[0] for r in recs]
        p,_ = precision_recall_at_k(ids, gt, k)
        n = ndcg_at_k(ids, gt, k)
        precisions.append(p); ndcgs.append(n)
    return float(sum(precisions)/len(precisions)), float(sum(ndcgs)/len(ndcgs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/train.csv")
    parser.add_argument("--test_path", default="data/processed/test.csv")
    parser.add_argument("--items_path", default="data/ml-100k/u.item")
    parser.add_argument("--alpha", type=float, default=Config.HYBRID_DEFAULT_ALPHA)
    parser.add_argument("--top_k", type=int, default=Config.TOP_K)
    args = parser.parse_args()

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    items_df = load_items_file(args.items_path)

    cf = CollaborativeFiltering(mode="user")
    cf.fit(train)
    cm = ContentModel()
    cm.fit(items_df, text_col="title")

    p, n = evaluate_hybrid(cf, cm, test, args.alpha, args.top_k)
    out = {"alpha": args.alpha, "precision_at_k": p, "ndcg_at_k": n}
    print(out)
    with open("experiments/hybrid_result.json","w") as f:
        json.dump(out,f,indent=2)