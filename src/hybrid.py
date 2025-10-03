import argparse, json, pandas as pd, numpy as np, os
from .cf_baseline import CollaborativeFiltering
from .content_based import ContentModel, load_items_file
from .config import Config
from .metrics import precision_recall_at_k, ndcg_at_k
from .logging_config import get_logger
from .utils import set_seed

logger = get_logger("hybrid")

def normalize_weights(w_cf, w_content, w_neural, neural_available):
    """
    Normalize weights to sum to 1.0. If neural not available, zero w_neural.
    If all weights are zero or negative, fallback to CF=1.0
    """
    if not neural_available:
        w_neural = 0.0
    
    total = w_cf + w_content + w_neural
    
    if total <= 0:
        logger.warning("All weights zero or negative, falling back to CF=1.0")
        return 1.0, 0.0, 0.0
    
    return w_cf/total, w_content/total, w_neural/total


def evaluate_hybrid_triweight(cf, cm, neural_model, test_df, w_cf, w_content, w_neural, k, sample_users):
    """
    Evaluate hybrid recommendations using tri-weight blending.
    """
    users = test_df.user_id.unique()
    users = np.random.choice(users, size=min(sample_users, len(users)), replace=False)
    
    precisions = []
    ndcgs = []
    
    for u in users:
        gt = test_df[test_df.user_id==u].item_id.tolist()
        
        # Get CF candidates
        cf_candidates = cf.recommend(u, n=300)
        
        # Get user's liked items for content scoring
        uidx = cf.user_map.get(u)
        liked_items = []
        if uidx is not None:
            row = cf.utility[uidx,:].toarray().ravel()
            rated_idx = (row>0).nonzero()[0]
            liked_items = [cf.rev_item_map[ridx] for ridx in rated_idx[:20]]
        
        # Compute tri-weight scores for each candidate
        final_scores = []
        for item_id, cf_score in cf_candidates:
            # CF score (already have it)
            score_cf = cf_score
            
            # Content score: average similarity to liked items
            content_score = 0.0
            if liked_items:
                sims = []
                for liked_id in liked_items:
                    sim_list = cm.similar_items(liked_id, top_n=50)
                    match = [s for s in sim_list if s[0]==item_id]
                    if match:
                        sims.append(match[0][1])
                if sims:
                    content_score = sum(sims) / len(sims)
            
            # Neural score
            neural_score = 0.0
            if neural_model is not None:
                neural_score = neural_model.predict_single(u, item_id)
            
            # Final blended score
            final = w_cf * score_cf + w_content * content_score + w_neural * neural_score
            final_scores.append((item_id, final))
        
        # Sort and take top_k
        final_scores.sort(key=lambda x: x[1], reverse=True)
        ids = [item_id for item_id, _ in final_scores[:k]]
        
        # Compute metrics
        p, _ = precision_recall_at_k(ids, gt, k)
        n = ndcg_at_k(ids, gt, k)
        precisions.append(p)
        ndcgs.append(n)
    
    return float(sum(precisions)/len(precisions)), float(sum(ndcgs)/len(ndcgs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/train.csv")
    parser.add_argument("--test_path", default="data/processed/test.csv")
    parser.add_argument("--items_path", default="data/ml-100k/u.item")
    parser.add_argument("--w_cf", type=float, default=0.6, help="Weight for CF component")
    parser.add_argument("--w_content", type=float, default=0.4, help="Weight for content component")
    parser.add_argument("--w_neural", type=float, default=0.0, help="Weight for neural component")
    parser.add_argument("--top_k", type=int, default=Config.TOP_K)
    parser.add_argument("--sample_users", type=int, default=150, help="Number of users to sample for evaluation")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed")
    parser.add_argument("--neural_model_path", default="models/neural_cf.pt", help="Path to neural model")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info("Starting hybrid evaluation with tri-weight blending")
    logger.info(f"Weights: CF={args.w_cf}, Content={args.w_content}, Neural={args.w_neural}")
    
    # Load data
    logger.info(f"Loading training data from {args.train_path}")
    train = pd.read_csv(args.train_path)
    logger.info(f"Loading test data from {args.test_path}")
    test = pd.read_csv(args.test_path)
    logger.info(f"Loading items data from {args.items_path}")
    items_df = load_items_file(args.items_path)
    
    # Build CF model
    logger.info("Fitting Collaborative Filtering model")
    cf = CollaborativeFiltering(mode="user")
    cf.fit(train)
    
    # Build content model
    logger.info("Fitting Content model")
    cm = ContentModel()
    cm.fit(items_df, text_col="title")
    
    # Try to load neural model
    neural_model = None
    neural_loaded = False
    if os.path.exists(args.neural_model_path):
        try:
            logger.info(f"Loading neural model from {args.neural_model_path}")
            from .neural_cf import NeuralCF
            neural_model = NeuralCF.load_from_file(args.neural_model_path)
            neural_loaded = True
            logger.info("Neural model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load neural model: {e}")
            neural_loaded = False
    else:
        logger.info(f"Neural model not found at {args.neural_model_path}, proceeding without it")
    
    # Normalize weights
    w_cf, w_content, w_neural = normalize_weights(args.w_cf, args.w_content, args.w_neural, neural_loaded)
    logger.info(f"Normalized weights: CF={w_cf:.3f}, Content={w_content:.3f}, Neural={w_neural:.3f}")
    
    # Evaluate
    logger.info(f"Evaluating on {args.sample_users} sampled users with top_k={args.top_k}")
    p, n = evaluate_hybrid_triweight(cf, cm, neural_model, test, w_cf, w_content, w_neural, args.top_k, args.sample_users)
    
    # Output results
    out = {
        "weights": {
            "cf": w_cf,
            "content": w_content,
            "neural": w_neural
        },
        "precision_at_k": p,
        "ndcg_at_k": n,
        "sample_users": args.sample_users,
        "neural_loaded": neural_loaded
    }
    
    logger.info(f"Results: Precision@{args.top_k}={p:.4f}, NDCG@{args.top_k}={n:.4f}")
    
    # Ensure experiments directory exists
    os.makedirs("experiments", exist_ok=True)
    
    # Write results
    output_path = "experiments/hybrid_result.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results written to {output_path}")