"""
Batch-to-device helper, the validation/evaluation loop, and the
full multi-task training loop (with early stopping + checkpointing).
"""

from contextlib import nullcontext
from typing import Dict

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    DEVICE,
    USE_AMP,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP_NORM,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LAMBDA_INTENT,
    LAMBDA_SENTIMENT,
    LAMBDA_PRIORITY,
    CHECKPOINT_DIR,
)
from .model import MultiTaskMARBERT, make_losses
from .metrics import calculate_average_macro_f1
from .checkpoint import save_checkpoint, load_trained_model


# ============================================================
# 15. DEVICE HELPER
# ============================================================

def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:

    return {
        key: value.to(
            DEVICE,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


# ============================================================
# 17. VALIDATION / EVALUATION
# ============================================================

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    intent_loss_fn: nn.Module,
    sentiment_loss_fn: nn.Module,
    priority_loss_fn: nn.Module,
) -> Dict:

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_intent_true = []
    all_intent_pred = []

    all_sentiment_true = []
    all_sentiment_pred = []

    all_priority_true = []
    all_priority_pred = []

    with torch.no_grad():

        for batch in data_loader:

            batch = move_batch_to_device(
                batch
            )

            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
                if USE_AMP
                else nullcontext()
            )

            with autocast_context:

                outputs = model(
                    input_ids=batch[
                        "input_ids"
                    ],

                    attention_mask=batch[
                        "attention_mask"
                    ],
                )

                intent_loss = (
                    intent_loss_fn(
                        outputs[
                            "intent_logits"
                        ],
                        batch[
                            "intent_label"
                        ],
                    )
                )

                sentiment_loss = (
                    sentiment_loss_fn(
                        outputs[
                            "sentiment_logits"
                        ],
                        batch[
                            "sentiment_label"
                        ],
                    )
                )

                priority_loss = (
                    priority_loss_fn(
                        outputs[
                            "priority_logits"
                        ],
                        batch[
                            "priority_label"
                        ],
                    )
                )

                loss = (
                    LAMBDA_INTENT
                    * intent_loss
                    +
                    LAMBDA_SENTIMENT
                    * sentiment_loss
                    +
                    LAMBDA_PRIORITY
                    * priority_loss
                )

            batch_size = (
                batch[
                    "input_ids"
                ].size(0)
            )

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_samples += batch_size

            # Intent
            all_intent_true.extend(
                batch[
                    "intent_label"
                ]
                .cpu()
                .numpy()
                .tolist()
            )

            all_intent_pred.extend(
                outputs[
                    "intent_logits"
                ]
                .argmax(dim=1)
                .cpu()
                .numpy()
                .tolist()
            )

            # Sentiment
            all_sentiment_true.extend(
                batch[
                    "sentiment_label"
                ]
                .cpu()
                .numpy()
                .tolist()
            )

            all_sentiment_pred.extend(
                outputs[
                    "sentiment_logits"
                ]
                .argmax(dim=1)
                .cpu()
                .numpy()
                .tolist()
            )

            # Priority
            all_priority_true.extend(
                batch[
                    "priority_label"
                ]
                .cpu()
                .numpy()
                .tolist()
            )

            all_priority_pred.extend(
                outputs[
                    "priority_logits"
                ]
                .argmax(dim=1)
                .cpu()
                .numpy()
                .tolist()
            )

    average_loss = (
        total_loss
        /
        max(total_samples, 1)
    )

    (
        average_macro_f1,
        f1s,
    ) = calculate_average_macro_f1(
        all_intent_true,
        all_intent_pred,
        all_sentiment_true,
        all_sentiment_pred,
        all_priority_true,
        all_priority_pred,
    )

    return {
        "loss": float(
            average_loss
        ),

        "average_macro_f1": float(
            average_macro_f1
        ),

        **f1s,

        "intent_true":
            all_intent_true,

        "intent_pred":
            all_intent_pred,

        "sentiment_true":
            all_sentiment_true,

        "sentiment_pred":
            all_sentiment_pred,

        "priority_true":
            all_priority_true,

        "priority_pred":
            all_priority_pred,
    }


# ============================================================
# 24. TRAINING LOOP
# ============================================================

def train_model(
    model: MultiTaskMARBERT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_df: pd.DataFrame,
    tokenizer,
    max_length: int,
) -> MultiTaskMARBERT:

    (
        intent_loss_fn,
        sentiment_loss_fn,
        priority_loss_fn,
    ) = make_losses(
        train_df
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    best_val_metric = (
        -float("inf")
    )

    epochs_without_improvement = 0

    print(
        "\n" + "=" * 70
    )

    print(
        "Training..."
    )

    print(
        "=" * 70
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        model.train()

        running_total_loss = 0.0
        running_intent_loss = 0.0
        running_sentiment_loss = 0.0
        running_priority_loss = 0.0

        sample_count = 0

        for batch in train_loader:

            batch = move_batch_to_device(
                batch
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
                if USE_AMP
                else nullcontext()
            ):

                outputs = model(
                    input_ids=batch[
                        "input_ids"
                    ],

                    attention_mask=batch[
                        "attention_mask"
                    ],
                )

                intent_loss = (
                    intent_loss_fn(
                        outputs[
                            "intent_logits"
                        ],
                        batch[
                            "intent_label"
                        ],
                    )
                )

                sentiment_loss = (
                    sentiment_loss_fn(
                        outputs[
                            "sentiment_logits"
                        ],
                        batch[
                            "sentiment_label"
                        ],
                    )
                )

                priority_loss = (
                    priority_loss_fn(
                        outputs[
                            "priority_logits"
                        ],
                        batch[
                            "priority_label"
                        ],
                    )
                )

                total_loss = (
                    LAMBDA_INTENT
                    * intent_loss

                    +
                    LAMBDA_SENTIMENT
                    * sentiment_loss

                    +
                    LAMBDA_PRIORITY
                    * priority_loss
                )

            scaler.scale(
                total_loss
            ).backward()

            if (
                GRADIENT_CLIP_NORM
                is not None
            ):

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRADIENT_CLIP_NORM,
                )

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_size = (
                batch[
                    "input_ids"
                ].size(0)
            )

            sample_count += (
                batch_size
            )

            running_total_loss += (
                float(
                    total_loss.item()
                )
                * batch_size
            )

            running_intent_loss += (
                float(
                    intent_loss.item()
                )
                * batch_size
            )

            running_sentiment_loss += (
                float(
                    sentiment_loss.item()
                )
                * batch_size
            )

            running_priority_loss += (
                float(
                    priority_loss.item()
                )
                * batch_size
            )

        train_total = (
            running_total_loss
            /
            max(
                sample_count,
                1,
            )
        )

        train_intent = (
            running_intent_loss
            /
            max(
                sample_count,
                1,
            )
        )

        train_sentiment = (
            running_sentiment_loss
            /
            max(
                sample_count,
                1,
            )
        )

        train_priority = (
            running_priority_loss
            /
            max(
                sample_count,
                1,
            )
        )

        val_results = evaluate_model(
            model,
            val_loader,
            intent_loss_fn,
            sentiment_loss_fn,
            priority_loss_fn,
        )

        print(
            f"\nEpoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_total:.4f}"
        )

        print(
            f"  Intent Loss: "
            f"{train_intent:.4f}"
        )

        print(
            f"  Sentiment Loss: "
            f"{train_sentiment:.4f}"
        )

        print(
            f"  Priority Loss: "
            f"{train_priority:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_results['loss']:.4f}"
        )

        print(
            f"Intent Macro F1: "
            f"{val_results['intent_macro_f1']:.4f}"
        )

        print(
            "Sentiment Macro F1: "
            f"{val_results['sentiment_macro_f1']:.4f}"
        )

        print(
            "Priority Macro F1: "
            f"{val_results['priority_macro_f1']:.4f}"
        )

        print(
            "Average Macro F1: "
            f"{val_results['average_macro_f1']:.4f}"
        )

        # ----------------------------------------------------
        # Best model selection based ONLY on validation
        # ----------------------------------------------------

        if (
            val_results[
                "average_macro_f1"
            ]
            >
            best_val_metric
        ):

            best_val_metric = (
                val_results[
                    "average_macro_f1"
                ]
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                tokenizer,
                CHECKPOINT_DIR,
                max_length,
            )

            print(
                "New best validation "
                "Average Macro F1: "
                f"{best_val_metric:.4f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                "No improvement. "
                f"Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

            if (
                epochs_without_improvement
                >=
                EARLY_STOPPING_PATIENCE
            ):

                print(
                    "Early stopping triggered."
                )

                break

    # --------------------------------------------------------
    # Load BEST validation checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading best validation checkpoint..."
    )

    (
        best_model,
        _,
        _,
    ) = load_trained_model(
        CHECKPOINT_DIR
    )

    return best_model
