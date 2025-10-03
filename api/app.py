import os, time, logging
from functools import lru_cache
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Import your existing modules (adjust names if different)
from src.cf_baseline import CollaborativeFiltering
from src.content_based import ContentRecommender
from src.popularity import PopularityModel
from src.neural_cf import NeuralCF  # OPTIONAL; handle absence
from src.utils import set_seed

LOGGER = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

# Paths / env
DATA_PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
TRAIN_PATH = os.getenv("TRAIN_PATH", f"{DATA_PROCESSED_DIR}/train.csv")
ITEMS_PATH = os.getenv("ITEMS_PATH", "data/ml-100k/u.item")
NEURAL_MODEL_PATH = os.getenv("NEURAL_MODEL_PATH", "models/neural_cf.pt")
SEED = int(os.getenv("SEED", "42"))

# Default blend weights (can override via query params)
DEFAULT_W_CF = float(os.getenv("W_CF", "0.6"))
DEFAULT_W_CONTENT = float(os.getenv("W_CONTENT", "0.4"))
DEFAULT_W_NEURAL = float(os.getenv("W_NEURAL", "0.0"))

FALLBACK_K = 50  # internal candidate pool size

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[dict]
    used_neural: bool
    weights: dict
    latency_sec: float

class BatchRequest(BaseModel):
    user_ids: List[int]
    k: int = 10

class BatchResponse(BaseModel):
    results: List[RecommendResponse]

class Ctx:
    cf: Optional[CollaborativeFiltering] = None
    content: Optional[ContentRecommender] = None
    neural: Optional[NeuralCF] = None
    pop: Optional[PopularityModel] = None
    user_hist = {}
    item_titles = {}
    meta = {}

CTX = Ctx()
app = FastAPI(title="Hybrid Recommender API", version="1.0.0")

def _exists(p: str) -> bool:
    return p and os.path.exists(p)

def _load_titles(path: str):
    if not _exists(path):
        LOGGER.warning("Item metadata %s not found", path)
        return {}
    import pandas as pd
    df = pd.read_csv(path, sep="|", header=None, encoding="latin-1")
    df = df[[0,1]]
    df.columns = ["item_id","title"]
    return dict(zip(df.item_id, df.title))

def _normalize(w_cf, w_content, w_neural, neural_available):
    if not neural_available and w_neural > 0:
        LOGGER.info("Neural weight %.2f ignored (model absent).", w_neural)
        w_neural = 0.0
    total = w_cf + w_content + w_neural
    if total <= 0:
        LOGGER.warning("All weights <= 0; fallback CF=1.")
        return 1.0, 0.0, 0.0
    return w_cf/total, w_content/total, w_neural/total

def bootstrap():
    import pandas as pd
    set_seed(SEED)
    if not _exists(TRAIN_PATH):
        LOGGER.error("TRAIN_PATH missing: %s (run preprocess first)", TRAIN_PATH)
        return
    train = pd.read_csv(TRAIN_PATH)
    CTX.user_hist = train.groupby("user_id")["item_id"].apply(list).to_dict()

    # CF
    cf = CollaborativeFiltering(mode="user")
    cf.fit(train)
    CTX.cf = cf

    # Popularity
    pop = PopularityModel()
    pop.fit(train)
    CTX.pop = pop

    # Content
    try:
        content = ContentRecommender()
        content.fit(train, items_path=ITEMS_PATH)
        CTX.content = content
    except Exception as e:
        LOGGER.exception("Content model failed: %s", e)

    # Neural (optional)
    if _exists(NEURAL_MODEL_PATH):
        try:
            neural = NeuralCF.load_from_file(NEURAL_MODEL_PATH)  # implement this if not present
            CTX.neural = neural
        except Exception as e:
            LOGGER.warning("Neural load failed: %s", e)

    CTX.item_titles = _load_titles(ITEMS_PATH)
    CTX.meta = {
        "seed": SEED,
        "train_rows": len(train),
        "neural_loaded": CTX.neural is not None,
        "timestamp_loaded": time.time()
    }
    LOGGER.info("Bootstrap complete. Rows=%d", len(train))

@lru_cache(maxsize=4096)
def _serve(user_id: int, k: int, w_cf: float, w_content: float, w_neural: float) -> RecommendResponse:
    start = time.time()
    if CTX.cf is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
    hist = CTX.user_hist.get(user_id, [])
    used_neural = CTX.neural is not None
    w_cf, w_content, w_neural = _normalize(w_cf, w_content, w_neural, used_neural)

    scores = {}

    # CF
    for iid, s in CTX.cf.recommend(user_id, n=FALLBACK_K):
        scores[iid] = scores.get(iid, 0.0) + w_cf * s

    # Content multi-anchor
    if CTX.content and hist:
        anchors = hist[-3:]
        try:
            for iid, s in CTX.content.multi_anchor_recommend(anchors, exclude=set(hist), top_n=FALLBACK_K):
                scores[iid] = scores.get(iid, 0.0) + w_content * s
        except Exception as e:
            LOGGER.warning("Content recommend failed: %s", e)

    # Neural
    if used_neural:
        candidates = list(scores.keys())
        if not candidates and CTX.pop:
            candidates = [i for i,_ in CTX.pop.top_n(FALLBACK_K)]
        for iid in candidates:
            try:
                nscore = CTX.neural.predict_single(user_id, iid)
            except Exception:
                nscore = 0.0
            scores[iid] = scores.get(iid, 0.0) + w_neural * nscore

    # Fallback
    if not scores and CTX.pop:
        for iid, s in CTX.pop.top_n(k+len(hist)):
            if iid in hist: continue
            scores[iid] = s

    # Remove seen
    for iid in hist:
        scores.pop(iid, None)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    recs = [
        {"item_id": int(iid), "score": float(sc), "title": CTX.item_titles.get(iid)}
        for iid, sc in ranked
    ]
    return RecommendResponse(
        user_id=user_id,
        recommendations=recs,
        used_neural=used_neural,
        weights={"w_cf": w_cf, "w_content": w_content, "w_neural": w_neural},
        latency_sec=round(time.time()-start, 4)
    )

@app.on_event("startup")
def _on_startup():
    bootstrap()

@app.get("/health")
def health():
    return {"status":"ok", "cf_loaded": CTX.cf is not None}

@app.get("/meta")
def meta(): return CTX.meta

@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=100),
    w_cf: float = DEFAULT_W_CF,
    w_content: float = DEFAULT_W_CONTENT,
    w_neural: float = DEFAULT_W_NEURAL
):
    return _serve(user_id, k, w_cf, w_content, w_neural)

@app.post("/batch_recommend", response_model=BatchResponse)
def batch(body: BatchRequest):
    out = []
    for uid in body.user_ids:
        try:
            out.append(_serve(uid, body.k, DEFAULT_W_CF, DEFAULT_W_CONTENT, DEFAULT_W_NEURAL))
        except HTTPException:
            continue
    return BatchResponse(results=out)

@app.post("/refresh")
def refresh():
    _serve.cache_clear()
    bootstrap()
    return {"status":"refreshed", "timestamp": time.time()}
