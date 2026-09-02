"""
Token-length analysis, the PyTorch Dataset wrapper, and the
dynamic-padding collator used for the multi-task MARBERT model.
"""

import math
from typing import Dict, List, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset

from .config import MAX_LENGTH, MAX_LENGTH_CAP


# ============================================================
# 9. TOKEN LENGTH ANALYSIS
# ============================================================

def analyze_token_lengths(
    texts: List[str],
    tokenizer,
) -> Tuple[
    Dict[str, float],
    int,
]:

    print(
        "\nAnalyzing MARBERT token lengths..."
    )

    lengths = []

    for text in texts:

        ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        lengths.append(
            len(ids)
        )

    token_lengths = np.asarray(
        lengths,
        dtype=np.int32,
    )

    stats = {
        "mean": float(
            np.mean(token_lengths)
        ),
        "median": float(
            np.median(token_lengths)
        ),
        "p90": float(
            np.percentile(
                token_lengths,
                90,
            )
        ),
        "p95": float(
            np.percentile(
                token_lengths,
                95,
            )
        ),
        "p99": float(
            np.percentile(
                token_lengths,
                99,
            )
        ),
        "max": int(
            np.max(token_lengths)
        ),
    }

    print(
        f"  Mean     : {stats['mean']:.2f}"
    )

    print(
        f"  Median   : {stats['median']:.2f}"
    )

    print(
        f"  P90      : {stats['p90']:.2f}"
    )

    print(
        f"  P95      : {stats['p95']:.2f}"
    )

    print(
        f"  P99      : {stats['p99']:.2f}"
    )

    print(
        f"  Maximum  : {stats['max']}"
    )

    model_max_length = tokenizer.model_max_length

    # Hugging Face may expose a very large sentinel.
    if (
        model_max_length is None
        or model_max_length > 100000
    ):
        model_max_length = 512

    if MAX_LENGTH is not None:

        chosen_max_length = int(
            MAX_LENGTH
        )

        print(
            f"\nMAX_LENGTH configured manually: "
            f"{chosen_max_length}"
        )

    else:

        # Start from p99.
        candidate = int(
            math.ceil(
                stats["p99"]
            )
        )

        candidate = max(
            candidate,
            16,
        )

        # Round upward to a small practical multiple.
        candidate = int(
            math.ceil(
                candidate / 8.0
            )
            * 8
        )

        candidate = min(
            candidate,
            int(MAX_LENGTH_CAP),
            int(model_max_length),
        )

        chosen_max_length = candidate

        print(
            "\nMAX_LENGTH selected automatically:"
        )

        print(
            f"  p99-based value: "
            f"{chosen_max_length}"
        )

        print(
            f"  model max length: "
            f"{model_max_length}"
        )

        print(
            f"  project cap: "
            f"{MAX_LENGTH_CAP}"
        )

    if (
        chosen_max_length
        < stats["p99"]
    ):

        print(
            "\nWarning: selected MAX_LENGTH "
            "is below p99."
        )

        print(
            "Some long examples will be truncated."
        )

    return (
        stats,
        chosen_max_length,
    )


# ============================================================
# 10. PYTORCH DATASET
# ============================================================

class CustomerDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length: int,
        intent_mapping: Dict[str, int],
        sentiment_mapping: Dict[str, int],
        priority_mapping: Dict[str, int],
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.intent_mapping = (
            intent_mapping
        )

        self.sentiment_mapping = (
            sentiment_mapping
        )

        self.priority_mapping = (
            priority_mapping
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.Tensor]:

        row = self.df.iloc[idx]

        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )

        return {
            "input_ids": torch.tensor(
                encoded["input_ids"],
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                encoded["attention_mask"],
                dtype=torch.long,
            ),

            "intent_label": torch.tensor(
                self.intent_mapping[
                    row["intent"]
                ],
                dtype=torch.long,
            ),

            "sentiment_label": torch.tensor(
                self.sentiment_mapping[
                    row["sentiment"]
                ],
                dtype=torch.long,
            ),

            "priority_label": torch.tensor(
                self.priority_mapping[
                    row["priority"]
                ],
                dtype=torch.long,
            ),
        }


# ============================================================
# 11. DYNAMIC PADDING COLLATOR
# ============================================================

class MultiTaskCollator:

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(
        self,
        features: List[
            Dict[str, torch.Tensor]
        ],
    ) -> Dict[str, torch.Tensor]:

        labels = {
            "intent_label": torch.stack(
                [
                    feature[
                        "intent_label"
                    ]
                    for feature in features
                ]
            ),

            "sentiment_label": torch.stack(
                [
                    feature[
                        "sentiment_label"
                    ]
                    for feature in features
                ]
            ),

            "priority_label": torch.stack(
                [
                    feature[
                        "priority_label"
                    ]
                    for feature in features
                ]
            ),
        }

        token_features = [
            {
                "input_ids": feature[
                    "input_ids"
                ],

                "attention_mask": feature[
                    "attention_mask"
                ],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            token_features,
            padding=True,
            return_tensors="pt",
        )

        batch.update(labels)

        return batch
