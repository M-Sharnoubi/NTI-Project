"""
Configuration, label mappings, and reproducibility/device setup
for the Arabic customer-service multi-task MARBERT project.
"""
from pathlib import Path

import random

import numpy as np

import torch


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = Path(__file__).resolve().parent / "dataset.csv"

MODEL_NAME = "UBC-NLP/MARBERT"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "marbert_multitask_checkpoint"

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# None = choose automatically from the 99th percentile.
# You can set an integer manually, for example 128.
MAX_LENGTH = None

# Safety ceiling for automatic selection.
MAX_LENGTH_CAP = 256

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 2

DROPOUT = 0.20

# Multi-task loss weights
LAMBDA_INTENT = 1.0
LAMBDA_SENTIMENT = 1.0
LAMBDA_PRIORITY = 1.0

# Keep False initially. Enable only after inspecting class distributions.
USE_CLASS_WEIGHTS = False

# Automatically use CUDA AMP if CUDA is available.
USE_MIXED_PRECISION = True

RETURN_CONFIDENCE = True

# Near-duplicate leakage checking
ENABLE_NEAR_DUPLICATE_CHECK = True
NEAR_DUPLICATE_THRESHOLD = 94.0
NEAR_DUPLICATE_TOP_K = 8
MIN_TEXT_LENGTH_FOR_NEAR_DUP = 12


# ============================================================
# 2. EXPLICIT LABEL MAPPINGS
# ============================================================

INTENT_LABELS = [
    "track_order",
    "cancel_order",
    "policy_inquiry",
    "complaint",
    "general_inquiry",
]

SENTIMENT_LABELS = [
    "positive",
    "negative",
    "neutral",
]

PRIORITY_LABELS = [
    "low",
    "medium",
    "high",
]

intent2id = {
    "track_order": 0,
    "cancel_order": 1,
    "policy_inquiry": 2,
    "complaint": 3,
    "general_inquiry": 4,
}

id2intent = {
    0: "track_order",
    1: "cancel_order",
    2: "policy_inquiry",
    3: "complaint",
    4: "general_inquiry",
}

sentiment2id = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
}

id2sentiment = {
    0: "positive",
    1: "negative",
    2: "neutral",
}

priority2id = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

id2priority = {
    0: "low",
    1: "medium",
    2: "high",
}


# ============================================================
# 3. REPRODUCIBILITY / DEVICE
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # More reproducible training.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = bool(USE_MIXED_PRECISION and DEVICE.type == "cuda")

print("=" * 70)
print("Arabic Customer Service - Multi-Task MARBERT")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixed Precision: {USE_AMP}")
else:
    print("Mixed Precision: disabled on CPU")
