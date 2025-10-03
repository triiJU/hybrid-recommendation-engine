import argparse, pandas as pd
from sklearn.model_selection import train_test_split
from .config import Config
from .utils import ensure_dir

def load_ratings(path="data/ml-100k/u.data"):
    df = pd.read_csv(path, sep="\t", names=["user_id","item_id","rating","timestamp"])
    return df

def filter_users(df, min_interactions):
    counts = df.user_id.value_counts()
    valid = counts[counts >= min_interactions].index
    return df[df.user_id.isin(valid)].copy()

def split(df):
    return train_test_split(df, test_size=Config.TEST_SIZE, random_state=Config.SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_path", default="data/ml-100k/u.data")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--min_interactions", type=int, default=Config.MIN_INTERACTIONS)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ratings = load_ratings(args.ratings_path)
    ratings = filter_users(ratings, args.min_interactions)
    train, test = split(ratings)
    train.to_csv(f"{args.out_dir}/train.csv", index=False)
    test.to_csv(f"{args.out_dir}/test.csv", index=False)
    print(f"[INFO] Train {train.shape} Test {test.shape}")