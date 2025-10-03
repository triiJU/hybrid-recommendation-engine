import argparse, pandas as pd, numpy as np, json
from .cf_baseline import CollaborativeFiltering
from .content_based import ContentModel, load_items_file
from .metrics import precision_recall_at_k
from .config import Config

def segment_users(train_df, threshold=4):
    counts = train_df.user_id.value_counts()
    new_users = counts[counts<=threshold].index
    return set(new_users)

def segment_items(train_df, threshold=10):
    counts = train_df.item_id.value_counts()
    new_items = counts[counts<=threshold].index
    return set(new_items)

def cold_start_eval(train, test, items_df, k):
    cf = CollaborativeFiltering(mode="user")
    cf.fit(train)
    cm = ContentModel(); cm.fit(items_df, text_col="title")

    new_users = segment_users(train)
    new_items = segment_items(train)

    users = test.user_id.unique()
    users = np.random.choice(users, size=min(150, len(users)), replace=False)

    def evaluate_segment(filter_fn):
        import numpy as np
        precisions_cf=[]; precisions_hybrid=[]
        for u in users:
            if not filter_fn(u): continue
            gt = test[test.user_id==u].item_id.tolist()
            cf_recs = cf.recommend(u, n=k)
            cf_ids = [r[0] for r in cf_recs]
            p_cf,_ = precision_recall_at_k(cf_ids, gt, k)
            hybrid = cm.hybrid_with_cf(u, cf, alpha=0.7, top_n=k)
            h_ids = [r[0] for r in hybrid]
            p_h,_ = precision_recall_at_k(h_ids, gt, k)
            if not np.isnan(p_cf): precisions_cf.append(p_cf)
            if not np.isnan(p_h): precisions_hybrid.append(p_h)
        return float(np.mean(precisions_cf) if precisions_cf else 0.0), \
               float(np.mean(precisions_hybrid) if precisions_hybrid else 0.0)

    new_user_cf, new_user_h = evaluate_segment(lambda u: u in new_users)
    random_user_cf, random_user_h = evaluate_segment(lambda u: True)

    return {
        "new_user_precision_cf": new_user_cf,
        "new_user_precision_hybrid": new_user_h,
        "all_user_precision_cf": random_user_cf,
        "all_user_precision_hybrid": random_user_h
    }

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

    res = cold_start_eval(train, test, items_df, args.k)
    print(res)
    with open("experiments/cold_start.json","w") as f:
        json.dump(res,f,indent=2)