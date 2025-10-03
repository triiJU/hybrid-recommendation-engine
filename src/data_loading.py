import argparse, os, zipfile, requests
from .utils import ensure_dir

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_movielens(dest="data"):
    ensure_dir(dest)
    zip_path = os.path.join(dest, "ml-100k.zip")
    if not os.path.exists(zip_path):
        r = requests.get(MOVIELENS_100K_URL)
        r.raise_for_status()
        with open(zip_path,"wb") as f:
            f.write(r.content)
    with zipfile.ZipFile(zip_path,"r") as z:
        z.extractall(dest)
    print("[INFO] MovieLens 100K ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_movielens", action="store_true")
    args = parser.parse_args()
    if args.download_movielens:
        download_movielens()