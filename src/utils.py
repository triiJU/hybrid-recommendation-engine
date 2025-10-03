import os, random, numpy as np
from .config import Config

def set_seed(seed=Config.SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        if torch.cuda.is_available():
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)