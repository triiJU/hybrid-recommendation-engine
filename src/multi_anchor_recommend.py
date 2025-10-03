def multi_anchor_recommend(self, anchor_item_ids, exclude=None, top_n=100):
    if exclude is None:
        exclude = set()
    agg = {}
    valid = 0
    for aid in anchor_item_ids:
        if aid not in self.item_index:
            continue
        valid += 1
        row = self.item_sim_matrix[self.item_index[aid]]
        for j, val in enumerate(row):
            if val <= 0: continue
            item_id = self.index_item[j]
            if item_id == aid or item_id in exclude: continue
            agg[item_id] = agg.get(item_id, 0.0) + float(val)
    denom = max(valid, 1)
    ranked = sorted(
        ((iid, sc / denom) for iid, sc in agg.items()),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    return ranked
