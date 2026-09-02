"""
Macro-F1 aggregation, per-task classification reports, and
test-set error analysis.
"""

from typing import Dict, List

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from .config import id2intent, id2sentiment, id2priority


# ============================================================
# 16. MULTI-TASK MACRO F1
# ============================================================

def calculate_average_macro_f1(
    y_true_intent,
    y_pred_intent,
    y_true_sentiment,
    y_pred_sentiment,
    y_true_priority,
    y_pred_priority,
):

    intent_f1 = (
        precision_recall_fscore_support(
            y_true_intent,
            y_pred_intent,
            average="macro",
            zero_division=0,
        )[2]
    )

    sentiment_f1 = (
        precision_recall_fscore_support(
            y_true_sentiment,
            y_pred_sentiment,
            average="macro",
            zero_division=0,
        )[2]
    )

    priority_f1 = (
        precision_recall_fscore_support(
            y_true_priority,
            y_pred_priority,
            average="macro",
            zero_division=0,
        )[2]
    )

    average_macro_f1 = (
        intent_f1
        +
        sentiment_f1
        +
        priority_f1
    ) / 3.0

    return (
        float(average_macro_f1),
        {
            "intent_macro_f1": float(
                intent_f1
            ),
            "sentiment_macro_f1": float(
                sentiment_f1
            ),
            "priority_macro_f1": float(
                priority_f1
            ),
        },
    )


# ============================================================
# 20. TASK METRICS / REPORT
# ============================================================

def print_task_results(
    task_name: str,
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str],
) -> None:

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        task_name
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy           : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{precision:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{f1:.4f}"
    )

    print(
        f"Weighted Precision : "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{weighted_f1:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(
                range(
                    len(target_names)
                )
            ),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "Confusion Matrix:"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(
                len(target_names)
            )
        ),
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"true_{x}"
            for x in target_names
        ],
        columns=[
            f"pred_{x}"
            for x in target_names
        ],
    )

    print(
        cm_df
    )


# ============================================================
# 21. ERROR ANALYSIS
# ============================================================

def build_error_dataframe(
    test_df: pd.DataFrame,
    evaluation: Dict,
) -> pd.DataFrame:

    error_df = pd.DataFrame(
        {
            "text":
                test_df[
                    "text"
                ].tolist(),

            "true_intent":
                [
                    id2intent[x]
                    for x
                    in evaluation[
                        "intent_true"
                    ]
                ],

            "pred_intent":
                [
                    id2intent[x]
                    for x
                    in evaluation[
                        "intent_pred"
                    ]
                ],

            "true_sentiment":
                [
                    id2sentiment[x]
                    for x
                    in evaluation[
                        "sentiment_true"
                    ]
                ],

            "pred_sentiment":
                [
                    id2sentiment[x]
                    for x
                    in evaluation[
                        "sentiment_pred"
                    ]
                ],

            "true_priority":
                [
                    id2priority[x]
                    for x
                    in evaluation[
                        "priority_true"
                    ]
                ],

            "pred_priority":
                [
                    id2priority[x]
                    for x
                    in evaluation[
                        "priority_pred"
                    ]
                ],
        }
    )

    error_df[
        "intent_wrong"
    ] = (
        error_df[
            "true_intent"
        ]
        !=
        error_df[
            "pred_intent"
        ]
    )

    error_df[
        "sentiment_wrong"
    ] = (
        error_df[
            "true_sentiment"
        ]
        !=
        error_df[
            "pred_sentiment"
        ]
    )

    error_df[
        "priority_wrong"
    ] = (
        error_df[
            "true_priority"
        ]
        !=
        error_df[
            "pred_priority"
        ]
    )

    return error_df


def print_error_examples(
    error_df: pd.DataFrame,
    filter_column: str,
    title: str,
    n: int = 5,
) -> None:

    subset = (
        error_df[
            error_df[
                filter_column
            ]
        ]
        .head(n)
    )

    print(
        "\n" + "-" * 70
    )

    print(
        title
    )

    print(
        "-" * 70
    )

    if subset.empty:

        print(
            "No examples."
        )

        return

    visible_columns = [
        "text",
        "true_intent",
        "pred_intent",
        "true_sentiment",
        "pred_sentiment",
        "true_priority",
        "pred_priority",
    ]

    print(
        subset[
            visible_columns
        ].to_string(
            index=False
        )
    )
