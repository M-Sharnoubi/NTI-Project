"""
Saving and loading trained MultiTaskMARBERT checkpoints
(model weights, tokenizer, and config.json).
"""

import os
import json

import torch
from transformers import AutoTokenizer

from .config import (
    MODEL_NAME,
    DROPOUT,
    intent2id,
    id2intent,
    sentiment2id,
    id2sentiment,
    priority2id,
    id2priority,
    LAMBDA_INTENT,
    LAMBDA_SENTIMENT,
    LAMBDA_PRIORITY,
    DEVICE,
    CHECKPOINT_DIR,
)
from .model import MultiTaskMARBERT


# ============================================================
# 18. CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    model: MultiTaskMARBERT,
    tokenizer,
    checkpoint_dir: str,
    max_length: int,
) -> None:

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    weights_path = os.path.join(
        checkpoint_dir,
        "model_state_dict.pt",
    )

    config_path = os.path.join(
        checkpoint_dir,
        "config.json",
    )

    # Model weights
    torch.save(
        model.state_dict(),
        weights_path,
    )

    # Tokenizer
    tokenizer.save_pretrained(
        checkpoint_dir
    )

    # Everything needed for later loading
    config = {
        "model_name":
            MODEL_NAME,

        "max_length":
            int(max_length),

        "dropout":
            float(DROPOUT),

        "intent2id":
            intent2id,

        "id2intent":
            {
                str(k): v
                for k, v in id2intent.items()
            },

        "sentiment2id":
            sentiment2id,

        "id2sentiment":
            {
                str(k): v
                for k, v in id2sentiment.items()
            },

        "priority2id":
            priority2id,

        "id2priority":
            {
                str(k): v
                for k, v in id2priority.items()
            },

        "lambda_intent":
            float(LAMBDA_INTENT),

        "lambda_sentiment":
            float(LAMBDA_SENTIMENT),

        "lambda_priority":
            float(LAMBDA_PRIORITY),
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saving best checkpoint to: "
        f"{checkpoint_dir}"
    )


# ============================================================
# 19. CHECKPOINT LOAD
# ============================================================

def load_trained_model(
    checkpoint_dir: str = CHECKPOINT_DIR,
):

    if not os.path.isdir(
        checkpoint_dir
    ):
        raise FileNotFoundError(
            f"Checkpoint directory not found: "
            f"{checkpoint_dir}"
        )

    config_path = os.path.join(
        checkpoint_dir,
        "config.json",
    )

    weights_path = os.path.join(
        checkpoint_dir,
        "model_state_dict.pt",
    )

    if not os.path.isfile(
        config_path
    ):
        raise FileNotFoundError(
            f"Missing config: "
            f"{config_path}"
        )

    if not os.path.isfile(
        weights_path
    ):
        raise FileNotFoundError(
            f"Missing weights: "
            f"{weights_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            checkpoint_dir
        )
    )

    model = MultiTaskMARBERT(
        model_name=config[
            "model_name"
        ],

        num_intents=len(
            config["intent2id"]
        ),

        num_sentiments=len(
            config["sentiment2id"]
        ),

        num_priorities=len(
            config["priority2id"]
        ),

        dropout=float(
            config.get(
                "dropout",
                DROPOUT,
            )
        ),
    )

    # Newer PyTorch supports weights_only.
    # Fallback keeps compatibility with older versions.
    try:

        state_dict = torch.load(
            weights_path,
            map_location=DEVICE,
            weights_only=True,
        )

    except TypeError:

        state_dict = torch.load(
            weights_path,
            map_location=DEVICE,
        )

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded trained model from: "
        f"{checkpoint_dir}"
    )

    return (
        model,
        tokenizer,
        config,
    )
