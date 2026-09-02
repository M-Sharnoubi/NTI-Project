"""
The shared-encoder multi-task MARBERT model, plus class-weight
computation and the per-task loss functions.
"""

from typing import Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoModel

from .config import (
    DROPOUT,
    DEVICE,
    USE_CLASS_WEIGHTS,
    intent2id,
    sentiment2id,
    priority2id,
)


# ============================================================
# 12. MULTI-TASK MARBERT
# ============================================================

class MultiTaskMARBERT(
    nn.Module
):

    def __init__(
        self,
        model_name: str,
        num_intents: int,
        num_sentiments: int,
        num_priorities: int,
        dropout: float = DROPOUT,
    ):

        super().__init__()

        # One shared MARBERT encoder.
        self.encoder = (
            AutoModel.from_pretrained(
                model_name
            )
        )

        hidden_size = (
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.intent_classifier = (
            nn.Linear(
                hidden_size,
                num_intents,
            )
        )

        self.sentiment_classifier = (
            nn.Linear(
                hidden_size,
                num_sentiments,
            )
        )

        self.priority_classifier = (
            nn.Linear(
                hidden_size,
                num_priorities,
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # BERT-style first-token representation.
        shared_representation = (
            outputs
            .last_hidden_state[:, 0, :]
        )

        shared_representation = (
            self.dropout(
                shared_representation
            )
        )

        intent_logits = (
            self.intent_classifier(
                shared_representation
            )
        )

        sentiment_logits = (
            self.sentiment_classifier(
                shared_representation
            )
        )

        priority_logits = (
            self.priority_classifier(
                shared_representation
            )
        )

        # Return raw logits.
        # No softmax during training.
        return {
            "intent_logits": intent_logits,
            "sentiment_logits": sentiment_logits,
            "priority_logits": priority_logits,
        }


# ============================================================
# 13. CLASS WEIGHT SUPPORT
# ============================================================

def compute_task_class_weights(
    series: pd.Series,
    mapping: Dict[str, int],
    task_name: str,
) -> torch.Tensor:

    labels = (
        series
        .map(mapping)
        .to_numpy()
    )

    all_classes = np.arange(
        len(mapping)
    )

    counts = np.bincount(
        labels,
        minlength=len(mapping),
    )

    total = max(
        len(labels),
        1,
    )

    num_classes = len(mapping)

    weights = np.ones(
        len(mapping),
        dtype=np.float32,
    )

    for class_id in all_classes:

        if counts[class_id] > 0:

            weights[class_id] = (
                total
                /
                (
                    num_classes
                    *
                    counts[class_id]
                )
            )

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    inverse_mapping = {
        value: key
        for key, value in mapping.items()
    }

    print(
        f"\nClass weights for {task_name}:"
    )

    for class_id, weight in enumerate(
        weights_tensor.tolist()
    ):

        label = inverse_mapping[
            class_id
        ]

        print(
            f"  {label:20s}: "
            f"{weight:.4f}"
        )

    return weights_tensor


# ============================================================
# 14. LOSS FUNCTIONS
# ============================================================

def make_losses(
    train_df: pd.DataFrame,
):

    intent_weight = None
    sentiment_weight = None
    priority_weight = None

    if USE_CLASS_WEIGHTS:

        intent_weight = (
            compute_task_class_weights(
                train_df["intent"],
                intent2id,
                "intent",
            )
            .to(DEVICE)
        )

        sentiment_weight = (
            compute_task_class_weights(
                train_df["sentiment"],
                sentiment2id,
                "sentiment",
            )
            .to(DEVICE)
        )

        priority_weight = (
            compute_task_class_weights(
                train_df["priority"],
                priority2id,
                "priority",
            )
            .to(DEVICE)
        )

    else:

        print(
            "\nClass weighting: DISABLED"
        )

        print(
            "Set USE_CLASS_WEIGHTS=True "
            "if the inspected training "
            "distribution is significantly imbalanced."
        )

    intent_loss = (
        nn.CrossEntropyLoss(
            weight=intent_weight
        )
    )

    sentiment_loss = (
        nn.CrossEntropyLoss(
            weight=sentiment_weight
        )
    )

    priority_loss = (
        nn.CrossEntropyLoss(
            weight=priority_weight
        )
    )

    return (
        intent_loss,
        sentiment_loss,
        priority_loss,
    )
