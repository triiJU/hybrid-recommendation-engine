import logging, os
from .config import Config

def get_logger(name="recommender"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    os.makedirs(Config.EXPERIMENT_DIR, exist_ok=True)
    fh = logging.FileHandler(f"{Config.EXPERIMENT_DIR}/run.log")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger