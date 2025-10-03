import argparse, optuna, pandas as pd, numpy as np
from .cf_baseline import CollaborativeFiltering
from .metrics import precision_recall_at_k
from .config import Config

def objective(trial, train, test, k):
    mode = trial.suggest_categorical("mode", ["user","item"])
    neighbors = trial.suggest_int("k_neighbors", 5, 60)
    cf = CollaborativeFiltering(mode=mode)
    cf.fit(train)
    users = test.user_id.unique()
    users = np.random.choice(users, size=min(80,len(users)), replace=False)
    prec=[]
    for u in users:
        gt = test[test.user_id==u].item_id.tolist()
        recs=[]
        for item_id in train.item_id.unique()[:400]:
            score = cf.predict_single(u, item_id, k=neighbors)
            recs.append((item_id, score))
        recs = sorted(recs, key=lambda x:x[1], reverse=True)[:k]
        ids = [r[0] for r in recs]
        p,_ = precision_recall_at_k(ids, gt, k)
        prec.append(p)
    score = float(np.mean(prec))
    trial.set_user_attr("precision_at_k", score)
    return -score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/train.csv")
    parser.add_argument("--test_path", default="data/processed/test.csv")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, test, args.k), n_trials=args.trials)
    print("Best params:", study.best_params)
    print("Best Precision@K:", -study.best_value)