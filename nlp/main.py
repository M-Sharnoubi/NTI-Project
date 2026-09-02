"""
End-to-end entry point: load dataset, clean/split it, train the
multi-task MARBERT model, evaluate on the held-out test set, run
error analysis, and show example inference.
"""

import os
import json

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import (
    DATASET_PATH,
    SEED,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    MODEL_NAME,
    DROPOUT,
    CHECKPOINT_DIR,
    DEVICE,
    intent2id,
    sentiment2id,
    priority2id,
    INTENT_LABELS,
    SENTIMENT_LABELS,
    PRIORITY_LABELS,
    id2intent,
    id2sentiment,
    id2priority,
    ENABLE_NEAR_DUPLICATE_CHECK,
    set_seed,
)
from .preprocessing import (
    validate_dataset,
    print_class_distribution,
    build_near_duplicate_groups,
    split_with_group_leakage_control,
)
from .dataset import CustomerDataset, MultiTaskCollator, analyze_token_lengths
from .model import MultiTaskMARBERT, make_losses
from .train import train_model, evaluate_model
from .metrics import print_task_results, build_error_dataframe, print_error_examples
from .inference import analyze_customer_message


# ============================================================
# 25. MAIN
# ============================================================

def main():

    set_seed(SEED)

    # --------------------------------------------------------
    # Dataset path
    # --------------------------------------------------------

    if not os.path.isfile(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            f"\nDataset not found: "
            f"{DATASET_PATH}\n\n"
            "Put dataset.csv beside this script "
            "or modify DATASET_PATH at the top."
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH
    )

    # --------------------------------------------------------
    # Validation / cleaning
    # --------------------------------------------------------

    df = validate_dataset(
        df
    )

    if len(df) < 20:

        raise ValueError(
            "Dataset is too small after cleaning "
            "for a reliable 80/10/10 split."
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading MARBERT tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # Near-duplicate leakage check
    # --------------------------------------------------------

    if ENABLE_NEAR_DUPLICATE_CHECK:

        (
            group_ids,
            _,
        ) = build_near_duplicate_groups(
            df
        )

    else:

        print(
            "\nNear-duplicate detection disabled."
        )

        group_ids = list(
            range(
                len(df)
            )
        )

    # --------------------------------------------------------
    # Train / Validation / Test
    # --------------------------------------------------------

    print(
        "\nCreating leakage-aware "
        "train/validation/test split..."
    )

    (
        train_df,
        val_df,
        test_df,
    ) = split_with_group_leakage_control(
        df,
        group_ids,
    )

    print(
        "\nSplit sizes:"
    )

    print(
        f"  Train: "
        f"{len(train_df)} "
        f"({len(train_df)/len(df):.2%})"
    )

    print(
        f"  Validation: "
        f"{len(val_df)} "
        f"({len(val_df)/len(df):.2%})"
    )

    print(
        f"  Test: "
        f"{len(test_df)} "
        f"({len(test_df)/len(df):.2%})"
    )

    print_class_distribution(
        train_df,
        "Train",
    )

    print_class_distribution(
        val_df,
        "Validation",
    )

    print_class_distribution(
        test_df,
        "Test",
    )

    # --------------------------------------------------------
    # Token length analysis
    # --------------------------------------------------------

    (
        token_stats,
        max_length,
    ) = analyze_token_lengths(
        df["text"].tolist(),
        tokenizer,
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = CustomerDataset(
        train_df,
        tokenizer,
        max_length,
        intent2id,
        sentiment2id,
        priority2id,
    )

    val_dataset = CustomerDataset(
        val_df,
        tokenizer,
        max_length,
        intent2id,
        sentiment2id,
        priority2id,
    )

    test_dataset = CustomerDataset(
        test_df,
        tokenizer,
        max_length,
        intent2id,
        sentiment2id,
        priority2id,
    )

    # --------------------------------------------------------
    # Dynamic padding
    # --------------------------------------------------------

    collator = (
        MultiTaskCollator(
            tokenizer
        )
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading MARBERT encoder..."
    )

    print(
        "Building one shared encoder "
        "with three classification heads..."
    )

    model = MultiTaskMARBERT(
        model_name=MODEL_NAME,
        num_intents=len(
            intent2id
        ),
        num_sentiments=len(
            sentiment2id
        ),
        num_priorities=len(
            priority2id
        ),
        dropout=DROPOUT,
    )

    model.to(DEVICE)

    total_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_df=train_df,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    #
    # Test is touched here only.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Final Test Results"
    )

    print(
        "=" * 70
    )

    (
        intent_loss_fn,
        sentiment_loss_fn,
        priority_loss_fn,
    ) = make_losses(
        train_df
    )

    final_results = evaluate_model(
        model,
        test_loader,
        intent_loss_fn,
        sentiment_loss_fn,
        priority_loss_fn,
    )

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    print_task_results(
        "Intent",
        final_results[
            "intent_true"
        ],
        final_results[
            "intent_pred"
        ],
        INTENT_LABELS,
    )

    # --------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------

    print_task_results(
        "Sentiment",
        final_results[
            "sentiment_true"
        ],
        final_results[
            "sentiment_pred"
        ],
        SENTIMENT_LABELS,
    )

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    print_task_results(
        "Priority",
        final_results[
            "priority_true"
        ],
        final_results[
            "priority_pred"
        ],
        PRIORITY_LABELS,
    )

    # --------------------------------------------------------
    # Overall average Macro F1
    # --------------------------------------------------------

    average_macro_f1 = (
        final_results[
            "average_macro_f1"
        ]
    )

    # --------------------------------------------------------
    # Exact Match Accuracy
    #
    # Correct only when all three heads
    # are simultaneously correct.
    # --------------------------------------------------------

    exact_match = np.mean(
        [
            (
                intent_true
                ==
                intent_pred
            )
            and
            (
                sentiment_true
                ==
                sentiment_pred
            )
            and
            (
                priority_true
                ==
                priority_pred
            )

            for (
                intent_true,
                intent_pred,
                sentiment_true,
                sentiment_pred,
                priority_true,
                priority_pred,
            )
            in zip(
                final_results[
                    "intent_true"
                ],
                final_results[
                    "intent_pred"
                ],
                final_results[
                    "sentiment_true"
                ],
                final_results[
                    "sentiment_pred"
                ],
                final_results[
                    "priority_true"
                ],
                final_results[
                    "priority_pred"
                ],
            )
        ]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Overall Multi-Task Evaluation"
    )

    print(
        "=" * 70
    )

    print(
        f"Average Macro F1: "
        f"{average_macro_f1:.4f}"
    )

    print(
        f"Exact Match Accuracy: "
        f"{exact_match:.4f}"
    )

    # --------------------------------------------------------
    # Error Analysis
    #
    # Uses predictions already generated during
    # the one final test evaluation.
    # --------------------------------------------------------

    error_df = build_error_dataframe(
        test_df,
        final_results,
    )

    print_error_examples(
        error_df,
        "intent_wrong",
        "Intent Errors",
        n=5,
    )

    print_error_examples(
        error_df,
        "sentiment_wrong",
        "Sentiment Errors",
        n=5,
    )

    print_error_examples(
        error_df,
        "priority_wrong",
        "Priority Errors",
        n=5,
    )

    error_df[
        "multiple_heads_wrong"
    ] = (
        error_df[
            [
                "intent_wrong",
                "sentiment_wrong",
                "priority_wrong",
            ]
        ]
        .sum(
            axis=1
        )
        >= 2
    )

    print_error_examples(
        error_df,
        "multiple_heads_wrong",
        "Samples Where Multiple Heads Are Wrong",
        n=5,
    )

    # --------------------------------------------------------
    # Save error analysis
    # --------------------------------------------------------

    error_path = os.path.join(
        CHECKPOINT_DIR,
        "test_error_analysis.csv",
    )

    error_df.to_csv(
        error_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nError analysis saved to: "
        f"{error_path}"
    )

    # --------------------------------------------------------
    # Example inference
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Example inference"
    )

    print(
        "=" * 70
    )

    sample_1 = (
        "الاوردر #1001 لسه موصلش وأنا متضايق"
    )

    sample_2 = (
        "السماعة وصلت مكسورة وأنا محتاج أبدلها"
    )

    # Create inference config matching saved format.
    inference_config = {
        "max_length":
            max_length,

        "id2intent":
            {
                str(k): v
                for k, v in id2intent.items()
            },

        "id2sentiment":
            {
                str(k): v
                for k, v in id2sentiment.items()
            },

        "id2priority":
            {
                str(k): v
                for k, v in id2priority.items()
            },
    }

    # First example
    result_1 = (
        analyze_customer_message(
            sample_1,
            model=model,
            tokenizer=tokenizer,
            config=inference_config,
        )
    )

    print(
        "\nInput 1:"
    )

    print(
        sample_1
    )

    print(
        json.dumps(
            result_1,
            ensure_ascii=False,
            indent=2,
        )
    )

    # Second example
    result_2 = (
        analyze_customer_message(
            sample_2,
            model=model,
            tokenizer=tokenizer,
            config=inference_config,
        )
    )

    print(
        "\nInput 2:"
    )

    print(
        sample_2
    )

    print(
        json.dumps(
            result_2,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nTraining and evaluation "
        "completed successfully."
    )


# ============================================================
# 26. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
