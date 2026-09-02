# Imports

from config import *
import os
import re
import json
import math
import random
import unicodedata
from contextlib import nullcontext
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from transformers import AutoTokenizer, AutoModel


"""# preprocessing"""

def preprocess_text(text: str) -> str:
    """
    Light semantics-preserving preprocessing.

    We intentionally do NOT:
      - stemming
      - stop-word removal
      - root extraction
      - aggressive Arabic normalization
      - aggressive punctuation removal
      - removal of #order_id
    """

    if text is None:
        return ""

    text = str(text)

    # Canonical Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Remove invisible / zero-width characters
    text = re.sub(
        r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]",
        "",
        text,
    )

    # Normalize non-breaking space
    text = text.replace("\u00A0", " ")

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ============================================================
# 5. DATASET VALIDATION / CLEANING
# ============================================================

REQUIRED_COLUMNS = [
    "text",
    "intent",
    "sentiment",
    "priority",
]


def print_class_distribution(
    df: pd.DataFrame,
    split_name: str,
) -> None:
    print(f"\n--- {split_name} class distributions ---")

    label_config = {
        "intent": INTENT_LABELS,
        "sentiment": SENTIMENT_LABELS,
        "priority": PRIORITY_LABELS,
    }

    for column in ["intent", "sentiment", "priority"]:
        counts = (
            df[column]
            .value_counts()
            .reindex(
                label_config[column],
                fill_value=0,
            )
        )

        percentages = (
            counts / max(len(df), 1) * 100
        ).round(2)

        print(f"\n{column}:")

        for label in label_config[column]:
            print(
                f"  {label:20s} "
                f"{int(counts[label]):5d} "
                f"({percentages[label]:6.2f}%)"
            )


def validate_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\nLoading dataset...")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    if list(df.columns) != REQUIRED_COLUMNS:
        print(
            "\nWarning: column order differs from expected order. "
            "The script will use columns by name."
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nChecking missing values...")
    print(df[REQUIRED_COLUMNS].isna().sum())

    original_rows = len(df)

    # --------------------------------------------------------
    # Normalize labels safely
    # --------------------------------------------------------

    for column in [
        "intent",
        "sentiment",
        "priority",
    ]:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    # --------------------------------------------------------
    # Text cleaning
    # --------------------------------------------------------

    df["text"] = df["text"].astype("string")

    before_text_cleanup = len(df)

    # Remove rows where text is actually missing
    df = df[df["text"].notna()].copy()

    df["text"] = df["text"].map(preprocess_text)

    empty_text_mask = df["text"].eq("")

    empty_text_count = int(
        empty_text_mask.sum()
    )

    if empty_text_count > 0:
        df = df.loc[
            ~empty_text_mask
        ].copy()

    removed_text_rows = (
        before_text_cleanup - len(df)
    )

    print(
        f"\nRows before cleaning: "
        f"{original_rows}"
    )

    print(
        "Rows removed due to "
        f"missing/empty text: {removed_text_rows}"
    )

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    invalid_intent = sorted(
        set(df["intent"])
        - set(INTENT_LABELS)
    )

    invalid_sentiment = sorted(
        set(df["sentiment"])
        - set(SENTIMENT_LABELS)
    )

    invalid_priority = sorted(
        set(df["priority"])
        - set(PRIORITY_LABELS)
    )

    if invalid_intent:
        print(
            f"\nInvalid intent labels: "
            f"{invalid_intent}"
        )

    if invalid_sentiment:
        print(
            f"\nInvalid sentiment labels: "
            f"{invalid_sentiment}"
        )

    if invalid_priority:
        print(
            f"\nInvalid priority labels: "
            f"{invalid_priority}"
        )

    if (
        invalid_intent
        or invalid_sentiment
        or invalid_priority
    ):
        raise ValueError(
            "\nDataset contains unsupported labels.\n"
            "Fix the dataset instead of silently "
            "changing semantic labels."
        )

    # --------------------------------------------------------
    # Exact duplicate analysis
    # --------------------------------------------------------

    print(
        "\nChecking exact duplicate texts..."
    )

    duplicate_text_mask = (
        df["text"].duplicated(
            keep=False
        )
    )

    duplicate_rows = int(
        duplicate_text_mask.sum()
    )

    print(
        "Rows participating in exact "
        f"duplicates: {duplicate_rows}"
    )

    # --------------------------------------------------------
    # Conflicting duplicates
    # --------------------------------------------------------

    conflicting_texts = []

    for text, group in (
        df.loc[
            duplicate_text_mask
        ]
        .groupby(
            "text",
            sort=False,
        )
    ):

        if len(group) <= 1:
            continue

        if any(
            group[column].nunique(
                dropna=False
            ) > 1
            for column in [
                "intent",
                "sentiment",
                "priority",
            ]
        ):
            conflicting_texts.append(text)

    print(
        "Conflicting exact duplicate texts: "
        f"{len(conflicting_texts)}"
    )

    if conflicting_texts:

        print(
            "\nAction: removing ALL rows "
            "for conflicting exact-duplicate "
            "texts because selecting the "
            "correct label would require guessing."
        )

        df = df[
            ~df["text"].isin(
                conflicting_texts
            )
        ].copy()

    # --------------------------------------------------------
    # Same-label exact duplicates
    # --------------------------------------------------------

    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=[
            "text",
            "intent",
            "sentiment",
            "priority",
        ],
        keep="first",
    ).copy()

    duplicate_rows_removed = (
        before_dedup - len(df)
    )

    print(
        "Identical-label duplicate rows removed: "
        f"{duplicate_rows_removed}"
    )

    # --------------------------------------------------------
    # Text statistics
    # --------------------------------------------------------

    print_class_distribution(
        df,
        "Full cleaned dataset",
    )

    char_lengths = (
        df["text"]
        .str.len()
    )

    word_lengths = (
        df["text"]
        .str.split()
        .str.len()
    )

    print(
        "\nText-length statistics:"
    )

    print(
        f"  Characters -> "
        f"mean={char_lengths.mean():.2f}, "
        f"median={char_lengths.median():.2f}, "
        f"max={char_lengths.max()}"
    )

    print(
        f"  Words      -> "
        f"mean={word_lengths.mean():.2f}, "
        f"median={word_lengths.median():.2f}, "
        f"max={word_lengths.max()}"
    )

    print(
        f"\nRows after cleaning: "
        f"{len(df)}"
    )

    print(
        f"Total rows removed by cleaning: "
        f"{original_rows - len(df)}"
    )

    return df.reset_index(
        drop=True
    )

# ============================================================
# 6. NEAR-DUPLICATE DETECTION
# ============================================================

def build_near_duplicate_groups(
    df: pd.DataFrame,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
    top_k: int = NEAR_DUPLICATE_TOP_K,
) -> Tuple[List[int], List[str]]:

    n = len(df)

    if n == 0:
        return [], []

    texts = df["text"].tolist()

    eligible_indices = {
        i
        for i, text in enumerate(texts)
        if len(text)
        >= MIN_TEXT_LENGTH_FOR_NEAR_DUP
    }

    parent = list(
        range(n)
    )

    rank = [0] * n

    def find(
        x: int,
    ) -> int:

        while parent[x] != x:

            parent[x] = (
                parent[parent[x]]
            )

            x = parent[x]

        return x

    def union(
        a: int,
        b: int,
    ) -> None:

        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            return

        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b

        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a

        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    print(
        "\nNear-duplicate check:"
    )

    print(
        f"  Threshold: {threshold}"
    )

    print(
        f"  Top-K candidates: {top_k}"
    )

    print(
        f"  Eligible texts: "
        f"{len(eligible_indices)}"
    )

    near_duplicate_pairs = []

    # RapidFuzz makes this practical for a dataset
    # around a few thousand examples.
    for i in eligible_indices:

        matches = process.extract(
            texts[i],
            texts,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
            limit=top_k + 1,
        )

        for _, score, j in matches:

            if i == j:
                continue

            if j not in eligible_indices:
                continue

            shorter = min(
                len(texts[i]),
                len(texts[j]),
            )

            longer = max(
                len(texts[i]),
                len(texts[j]),
            )

            if longer == 0:
                continue

            length_ratio = (
                shorter / longer
            )

            # Avoid grouping very different
            # length strings that happen to
            # share a substring.
            if length_ratio < 0.70:
                continue

            if score >= threshold:

                union(i, j)

                if i < j:
                    near_duplicate_pairs.append(
                        (
                            i,
                            j,
                            float(score),
                        )
                    )

    roots = [
        find(i)
        for i in range(n)
    ]

    root_to_group = {}
    group_ids = []

    next_group = 0

    for root in roots:

        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1

        group_ids.append(
            root_to_group[root]
        )

    print(
        f"High-similarity pairs detected: "
        f"{len(near_duplicate_pairs)}"
    )

    if near_duplicate_pairs:

        print(
            "\nSample near-duplicate examples:"
        )

        for (
            i,
            j,
            score,
        ) in near_duplicate_pairs[:5]:

            print(
                f"  [{score:.1f}] "
                f"{texts[i]}"
            )

            print(
                f"       ↳ {texts[j]}"
            )

    group_sizes = Counter(
        group_ids
    )

    multi_member_groups = sum(
        1
        for size in group_sizes.values()
        if size > 1
    )

    print(
        f"Near-duplicate groups: "
        f"{multi_member_groups}"
    )

    print(
        "Total groups used for leakage-aware "
        f"split: {len(group_sizes)}"
    )


    # --------------------------------------------------------
    # Label conflicts inside near-duplicate groups
    # --------------------------------------------------------

    conflicting_groups = 0

    for group_id in sorted(
        set(group_ids)
    ):

        indices = [
            i
            for i, g in enumerate(group_ids)
            if g == group_id
        ]

        if len(indices) <= 1:
            continue

        group = df.iloc[
            indices
        ]

        has_conflict = any(
            group[column].nunique(
                dropna=False
            ) > 1
            for column in [
                "intent",
                "sentiment",
                "priority",
            ]
        )

        if has_conflict:
            conflicting_groups += 1

    print(
        "Near-duplicate groups with conflicting "
        f"labels: {conflicting_groups}"
    )

    if conflicting_groups:
        print(
            "\nWarning: some highly similar "
            "examples have different labels.\n"
            "They will stay in the same split "
            "to reduce leakage, but their labels "
            "should be manually reviewed."
        )

    return (
        group_ids,
        texts,
    )

# ============================================================
# 7. COMBINED LABEL
# ============================================================

def make_combined_label(
    df: pd.DataFrame,
) -> pd.Series:

    return (
        df["intent"].astype(str)
        + "||"
        + df["sentiment"].astype(str)
        + "||"
        + df["priority"].astype(str)
    )

# ============================================================
# 8. LEAKAGE-AWARE TRAIN / VAL / TEST SPLIT
# ============================================================

def split_with_group_leakage_control(
    df: pd.DataFrame,
    group_ids: List[int],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    group_ids = np.asarray(
        group_ids
    )

    group_to_indices = {}

    for index, group_id in enumerate(
        group_ids
    ):

        group_to_indices.setdefault(
            int(group_id),
            [],
        ).append(index)

    groups = sorted(
        group_to_indices.keys()
    )

    combined = make_combined_label(df)

    group_records = []

    for group_id in groups:

        indices = group_to_indices[
            group_id
        ]

        labels = combined.iloc[
            indices
        ].tolist()

        counts = Counter(labels)

        # Used only for stratification.
        # Original labels are never changed.
        strat_label = (
            counts.most_common(1)[0][0]
        )

        group_records.append(
            {
                "group_id": group_id,
                "strat_label": strat_label,
            }
        )

    group_df = pd.DataFrame(
        group_records
    )

    # --------------------------------------------------------
    # TEST = 10%
    # --------------------------------------------------------

    test_group_count = max(
        1,
        round(
            len(groups)
            * TEST_RATIO
        ),
    )

    can_stratify_test = (
        group_df["strat_label"]
        .value_counts()
        .min()
        >= 2
        and
        test_group_count
        >=
        group_df["strat_label"].nunique()
    )

    if can_stratify_test:

        train_val_groups, test_groups = (
            train_test_split(
                group_df,
                test_size=test_group_count,
                random_state=SEED,
                stratify=group_df[
                    "strat_label"
                ],
            )
        )

    else:

        print(
            "\nWarning: combined-label "
            "stratification for test split "
            "is not possible."
        )

        print(
            "Using reproducible group-only split."
        )

        train_val_groups, test_groups = (
            train_test_split(
                group_df,
                test_size=test_group_count,
                random_state=SEED,
                shuffle=True,
            )
        )

    # --------------------------------------------------------
    # VALIDATION = 10% overall
    # From remaining 90%, validation is 1/9.
    # --------------------------------------------------------

    val_fraction_within_train_val = (
        VAL_RATIO
        /
        (TRAIN_RATIO + VAL_RATIO)
    )

    val_group_count = max(
        1,
        round(
            len(train_val_groups)
            * val_fraction_within_train_val
        ),
    )

    can_stratify_val = (
        train_val_groups["strat_label"]
        .value_counts()
        .min()
        >= 2
        and
        val_group_count
        >=
        train_val_groups[
            "strat_label"
        ].nunique()
    )

    if can_stratify_val:

        train_groups, val_groups = (
            train_test_split(
                train_val_groups,
                test_size=val_group_count,
                random_state=SEED,
                stratify=train_val_groups[
                    "strat_label"
                ],
            )
        )

    else:

        print(
            "\nWarning: combined-label "
            "stratification for validation "
            "is not possible."
        )

        print(
            "Using reproducible group-only split."
        )

        train_groups, val_groups = (
            train_test_split(
                train_val_groups,
                test_size=val_group_count,
                random_state=SEED,
                shuffle=True,
            )
        )

    train_group_set = set(
        train_groups["group_id"].tolist()
    )

    val_group_set = set(
        val_groups["group_id"].tolist()
    )

    test_group_set = set(
        test_groups["group_id"].tolist()
    )

    train_indices = [
        index
        for index, group_id in enumerate(
            group_ids
        )
        if int(group_id)
        in train_group_set
    ]

    val_indices = [
        index
        for index, group_id in enumerate(
            group_ids
        )
        if int(group_id)
        in val_group_set
    ]

    test_indices = [
        index
        for index, group_id in enumerate(
            group_ids
        )
        if int(group_id)
        in test_group_set
    ]

    train_df = (
        df.iloc[
            train_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        df.iloc[
            val_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df.iloc[
            test_indices
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        train_df,
        val_df,
        test_df,
    )


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
        dataframe: pd.DataFrame,
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


# ============================================================
# 22. REGEX ENTITY EXTRACTION
# ============================================================

# Examples:
# #1001
# #1002
# #1003
ORDER_ID_PATTERN = re.compile(
    r"(?<!\w)#\d+"
)


def extract_entities(
    text: str,
) -> Dict[str, Optional[str]]:

    match = (
        ORDER_ID_PATTERN.search(
            text or ""
        )
    )

    return {
        "order_id":
            match.group(0)
            if match
            else None
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


