import argparse, pandas as pd, json
from .utils import ensure_dir

def popularity_rank(train_df, top_n=1000):
    agg = (train_df.groupby("item_id")
           .agg(interactions=("rating","count"),
                mean_rating=("rating","mean"))
           .reset_index())
    agg["score"] = agg["interactions"] * 0.7 + agg["mean_rating"] * 0.3
    return agg.sort_values("score", ascending=False).head(top_n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/train.csv")
    parser.add_argument("--out", default="experiments/popularity.json")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path)
    ensure_dir("experiments")
    top = popularity_rank(df)
    top.to_json(args.out, orient="records", indent=2)
    print(f"[INFO] Saved popularity baseline to {args.out}")