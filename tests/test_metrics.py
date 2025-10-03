from src.metrics import precision_recall_at_k, ndcg_at_k

def test_precision_recall():
    rec = [1,2,3,4,5]
    gt = [3,5,7]
    p,r = precision_recall_at_k(rec, gt, 5)
    assert 0 <= p <= 1
    assert 0 <= r <= 1

def test_ndcg():
    rec = [1,2,3,4,5]
    gt = [3,5,7]
    val = ndcg_at_k(rec, gt, 5)
    assert 0 <= val <= 1
