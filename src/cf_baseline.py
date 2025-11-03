from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
    def __init__(self, mode="user", top_k=50):
        assert mode in ("user", "item")
        self.mode = mode
        self.top_k = top_k

    def fit(self, df):
        users = sorted(df.user_id.unique())
        items = sorted(df.item_id.unique())
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {it: i for i, it in enumerate(items)}
        self.rev_item_map = {i: it for it, i in self.item_map.items()}

        row = df.user_id.map(self.user_map)
        col = df.item_id.map(self.item_map)
        data = df.rating.values
        self.utility = csr_matrix((data, (row, col)), shape=(len(users), len(items)))

        base = self.utility if self.mode == "user" else self.utility.T
        # Compute sparse cosine sim
        sim_matrix = cosine_similarity(base, dense_output=False)

        # Keep only top-k neighbors per row (sparsify)
        sim_matrix = sim_matrix.tolil()
        for i in range(sim_matrix.shape[0]):
            row_data = sim_matrix.data[i]
            row_idx = sim_matrix.rows[i]
            if len(row_data) > self.top_k:
                top_idx = np.argsort(row_data)[-self.top_k:]
                sim_matrix.data[i] = [row_data[j] for j in top_idx]
                sim_matrix.rows[i] = [row_idx[j] for j in top_idx]
        self.sim = sim_matrix.tocsr()
