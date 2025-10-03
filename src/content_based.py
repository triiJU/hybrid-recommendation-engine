import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_items_file(path):
    rows=[]
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            rows.append((int(parts[0]), parts[1]))
    return pd.DataFrame(rows, columns=["item_id","title"])

class ContentModel:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    def fit(self, items_df, text_col="title"):
        self.items_df = items_df.reset_index(drop=True)
        self.matrix = self.vectorizer.fit_transform(self.items_df[text_col].fillna(""))

    def similar_items(self, item_id, top_n=10):
        if item_id not in set(self.items_df.item_id):
            return []
        idx = self.items_df.index[self.items_df.item_id==item_id][0]
        vec = self.matrix[idx]
        sims = cosine_similarity(vec, self.matrix).ravel()
        order = sims.argsort()[::-1]
        out=[]
        for i in order:
            if i == idx: continue
            out.append((self.items_df.item_id.iloc[i], float(sims[i])))
            if len(out)>=top_n: break
        return out

    def hybrid_with_cf(self, user_id, cf_model, alpha=0.7, top_n=10):
        base = cf_model.recommend(user_id, n=300)
        recs=[]
        for item_id, cf_score in base:
            content_score = 0.0
            uidx = cf_model.user_map.get(user_id)
            if uidx is not None:
                row = cf_model.utility[uidx,:].toarray().ravel()
                rated_idx = (row>0).nonzero()[0]
                sims=[]
                for ridx in rated_idx[:20]:
                    liked_item_id = cf_model.rev_item_map[ridx]
                    sim_list = self.similar_items(liked_item_id, top_n=50)
                    match = [s for s in sim_list if s[0]==item_id]
                    if match: sims.append(match[0][1])
                if sims:
                    content_score = sum(sims)/len(sims)
            final = alpha*cf_score + (1-alpha)*content_score
            recs.append((item_id, final))
        return sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]

    def multi_anchor_recommend(self, anchor_item_ids, exclude=None, top_n=100):
        """
        Aggregate similarities for multiple anchor items.
        
        Args:
            anchor_item_ids: List of item IDs to use as anchors
            exclude: Set of item IDs to exclude from recommendations
            top_n: Number of recommendations to return
            
        Returns:
            List of (item_id, average_similarity_score) tuples
        """
        if exclude is None:
            exclude = set()
        
        agg = {}
        valid = 0
        
        for aid in anchor_item_ids:
            if aid not in set(self.items_df.item_id):
                continue
            valid += 1
            
            sim_list = self.similar_items(aid, top_n=top_n*2)
            for item_id, sim_score in sim_list:
                if item_id == aid or item_id in exclude:
                    continue
                agg[item_id] = agg.get(item_id, 0.0) + sim_score
        
        # Average by number of valid anchors
        denom = max(valid, 1)
        ranked = sorted(
            ((iid, sc / denom) for iid, sc in agg.items()),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return ranked

# Alias for API naming consistency
ContentRecommender = ContentModel