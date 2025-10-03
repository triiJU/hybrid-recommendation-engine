import argparse, numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, mode="user"):
        assert mode in ("user","item")
        self.mode=mode

    def fit(self, df):
        users = sorted(df.user_id.unique())
        items = sorted(df.item_id.unique())
        self.user_map = {u:i for i,u in enumerate(users)}
        self.item_map = {it:i for i,it in enumerate(items)}
        self.rev_item_map = {i:it for it,i in self.item_map.items()}
        row = df.user_id.map(self.user_map)
        col = df.item_id.map(self.item_map)
        data = df.rating.values
        self.utility = csr_matrix((data,(row,col)), shape=(len(users),len(items)))
        self.sim = cosine_similarity(self.utility if self.mode=="user" else self.utility.T)

    def predict_single(self, user_id, item_id, k=30):
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0
        u = self.user_map[user_id]; it = self.item_map[item_id]
        if self.mode=="user":
            col = self.utility[:, it].toarray().ravel()
            rated_idx = np.where(col>0)[0]
            sims = self.sim[u, rated_idx]; ratings = col[rated_idx]
        else:
            row = self.utility[u,:].toarray().ravel()
            rated_idx = np.where(row>0)[0]
            sims = self.sim[it, rated_idx]; ratings = row[rated_idx]
        if len(ratings)==0: return 0.0
        top = np.argsort(sims)[-k:]
        sims_k = sims[top]; ratings_k = ratings[top]
        denom = sims_k.sum()
        if denom == 0: return float(ratings.mean())
        return float((sims_k @ ratings_k)/(denom + 1e-8))

    def recommend(self, user_id, n=10, k=30):
        if user_id not in self.user_map: return []
        uidx = self.user_map[user_id]
        user_row = self.utility[uidx,:].toarray().ravel()
        rated = set(np.where(user_row>0)[0])
        out=[]
        for it_idx in range(self.utility.shape[1]):
            if it_idx in rated: continue
            item_id = self.rev_item_map[it_idx]
            score = self.predict_single(user_id, item_id, k=k)
            out.append((item_id, score))
        return sorted(out, key=lambda x: x[1], reverse=True)[:n]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/train.csv")
    parser.add_argument("--mode", default="user")
    args = parser.parse_args()
    df = pd.read_csv(args.train_path)
    cf = CollaborativeFiltering(mode=args.mode)
    cf.fit(df)
    sample = df.user_id.sample(1, random_state=42).iloc[0]
    print(cf.recommend(sample, n=10))