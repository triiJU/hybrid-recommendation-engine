import argparse, json, pandas as pd, random, numpy as np, os, logging
from .cf_baseline import CollaborativeFiltering
from .content_based import ContentModel, load_items_file
from .config import Config
from .metrics import precision_recall_at_k, ndcg_at_k
from .logging_config import get_logger

logger = get_logger("hybrid")

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def evaluate_hybrid_triweight(cf, cm, neural, test_df, w_cf, w_content, w_neural, k, sample_users, seed):
    """
    Evaluate hybrid model with tri-weight blending.
    
    Args:
        cf: Collaborative filtering model
        cm: Content-based model
        neural: Neural CF model (can be None)
        test_df: Test dataframe
        w_cf, w_content, w_neural: Blend weights
        k: Top-k for evaluation
        sample_users: Number of users to sample
        seed: Random seed
        
    Returns:
        Tuple of (precision@k, ndcg@k, user_count_evaluated)
    """
    users = test_df.user_id.unique()
    np.random.seed(seed)
    n_sample = min(sample_users, len(users))
    users = np.random.choice(users, size=n_sample, replace=False)
    
    precisions = []
    ndcgs = []
    
    for u in users:
        gt = test_df[test_df.user_id == u].item_id.tolist()
        
        # Get CF recommendations (candidate pool)
        cf_recs = cf.recommend(u, n=300)
        
        # Get user's liked items for content scoring
        uidx = cf.user_map.get(u)
        liked_items = []
        if uidx is not None:
            row = cf.utility[uidx, :].toarray().ravel()
            rated_idx = (row > 0).nonzero()[0]
            liked_items = [cf.rev_item_map[ridx] for ridx in rated_idx[:20]]
        
        # Score each candidate
        scores = {}
        for item_id, cf_score in cf_recs:
            # CF score
            final_score = w_cf * cf_score
            
            # Content score: average similarity to liked items
            if w_content > 0 and liked_items:
                sims = []
                for liked_id in liked_items:
                    sim_list = cm.similar_items(liked_id, top_n=50)
                    match = [s for s in sim_list if s[0] == item_id]
                    if match:
                        sims.append(match[0][1])
                if sims:
                    content_score = sum(sims) / len(sims)
                    final_score += w_content * content_score
            
            # Neural score
            if w_neural > 0 and neural is not None:
                neural_score = neural.predict_single(u, item_id)
                final_score += w_neural * neural_score
            
            scores[item_id] = final_score
        
        # Rank and evaluate
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        ids = [r[0] for r in ranked]
        
        p, _ = precision_recall_at_k(ids, gt, k)
        n = ndcg_at_k(ids, gt, k)
        precisions.append(p)
        ndcgs.append(n)
    
    return (
        float(sum(precisions) / len(precisions)) if precisions else 0.0,
        float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        len(users)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/train.csv")
    parser.add_argument("--test_path", default="data/processed/test.csv")
    parser.add_argument("--items_path", default="data/ml-100k/u.item")
    parser.add_argument("--w_cf", type=float, default=0.6)
    parser.add_argument("--w_content", type=float, default=0.4)
    parser.add_argument("--w_neural", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=Config.TOP_K)
    parser.add_argument("--sample_users", type=int, default=150)
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument("--neural_model_path", default="models/neural_cf.pt")
    args = parser.parse_args()

    # Set seeds for reproducibility
    set_seed(args.seed)
    
    logger.info("Loading data...")
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    items_df = load_items_file(args.items_path)

    logger.info("Training CF model...")
    cf = CollaborativeFiltering(mode="user")
    cf.fit(train)
    
    logger.info("Training content model...")
    cm = ContentModel()
    cm.fit(items_df, text_col="title")

    # Try to load neural model
    neural = None
    neural_loaded = False
    if os.path.exists(args.neural_model_path):
        try:
            from .neural_cf import NeuralCF
            neural = NeuralCF.load_from_file(args.neural_model_path)
            neural_loaded = True
            logger.info("Neural model loaded from %s", args.neural_model_path)
        except Exception as e:
            logger.warning("Could not load neural model: %s", e)
    else:
        logger.info("Neural model not found at %s", args.neural_model_path)
    
    # Normalize weights
    w_cf, w_content, w_neural = args.w_cf, args.w_content, args.w_neural
    if not neural_loaded and w_neural > 0:
        logger.warning("Neural weight %.2f ignored (model not loaded)", w_neural)
        w_neural = 0.0
    
    total_weight = w_cf + w_content + w_neural
    if total_weight <= 0:
        logger.error("All weights are <= 0, using default CF=1.0")
        w_cf, w_content, w_neural = 1.0, 0.0, 0.0
        total_weight = 1.0
    
    # Normalize
    w_cf /= total_weight
    w_content /= total_weight
    w_neural /= total_weight
    
    logger.info("Evaluating with weights: CF=%.3f, Content=%.3f, Neural=%.3f", 
                w_cf, w_content, w_neural)

    p, n, user_count = evaluate_hybrid_triweight(
        cf, cm, neural, test, w_cf, w_content, w_neural, 
        args.top_k, args.sample_users, args.seed
    )
    
    # Prepare results
    result = {
        "weights": {
            "w_cf": w_cf,
            "w_content": w_content,
            "w_neural": w_neural
        },
        "precision_at_k": p,
        "ndcg_at_k": n,
        "sample_users": args.sample_users,
        "user_count_evaluated": user_count,
        "neural_loaded": neural_loaded,
        "seed": args.seed
    }
    
    logger.info("Results: Precision@%d=%.4f, NDCG@%d=%.4f", args.top_k, p, args.top_k, n)
    
    # Ensure experiments directory exists
    os.makedirs("experiments", exist_ok=True)
    
    # Write results
    with open("experiments/hybrid_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("Results saved to experiments/hybrid_result.json")