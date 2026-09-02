"""
Arabic customer-service multi-task MARBERT package.

Modules:
    config         - configuration, label mappings, seed/device setup
    preprocessing  - text cleaning, dataset validation, near-duplicate
                     detection, leakage-aware train/val/test split
    dataset        - PyTorch Dataset, collator, token-length analysis
    model          - MultiTaskMARBERT architecture, class weights, losses
    metrics        - macro-F1, per-task reports, error analysis
    checkpoint     - saving/loading trained checkpoints
    train          - training loop and evaluation loop
    inference      - entity extraction and single-message inference
    main           - end-to-end training/evaluation entry point
"""

from .checkpoint import load_trained_model, save_checkpoint
from .inference import analyze_customer_message, extract_entities

__all__ = [
    "load_trained_model",
    "save_checkpoint",
    "analyze_customer_message",
    "extract_entities",
]
