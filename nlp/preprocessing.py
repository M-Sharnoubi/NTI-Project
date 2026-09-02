"""
Text preprocessing, dataset validation/cleaning, near-duplicate
detection, and leakage-aware train/val/test splitting.
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from sklearn.model_selection import train_test_split

from .config import (
    INTENT_LABELS,
    SENTIMENT_LABELS,
    PRIORITY_LABELS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
    NEAR_DUPLICATE_THRESHOLD,
    NEAR_DUPLICATE_TOP_K,
    MIN_TEXT_LENGTH_FOR_NEAR_DUP,
)


# ============================================================
# 4. LIGHT ARABIC PREPROCESSING
# ============================================================

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
