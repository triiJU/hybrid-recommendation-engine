class Config:
    SEED = 42
    TEST_SIZE = 0.2
    MIN_INTERACTIONS = 5
    TOP_K = 10
    USE_TORCH = True
    DEVICE = "cuda"
    EMBEDDING_DIM = 32
    EPOCHS = 10
    BATCH_SIZE = 256
    LR = 1e-3
    SAVE_DIR = "models"
    EXPERIMENT_DIR = "experiments"
    HYBRID_DEFAULT_ALPHA = 0.7