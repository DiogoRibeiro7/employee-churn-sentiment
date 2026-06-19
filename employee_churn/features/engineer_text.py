"""Lightweight text-statistics feature engineering.

These features are cheap, dependency-free signals derived purely from the shape
of the text (length, punctuation, negation, lexical diversity). They complement
the model-based sentiment/emotion features and frequently add predictive lift on
short free-text feedback.
"""

from __future__ import annotations

import re
from typing import Dict

import pandas as pd

_WORD_RE = re.compile(r"[A-Za-z']+")
_NEGATIONS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nor",
        "cannot",
        "cant",
        "wont",
        "dont",
        "doesnt",
        "didnt",
        "isnt",
        "arent",
        "wasnt",
        "werent",
    }
)


def text_statistics(text: str) -> Dict[str, float]:
    """Compute shape-based statistics for a single piece of text.

    Args:
        text: Input string (non-strings are coerced via ``str``).

    Returns:
        Mapping with character count, word count, average word length,
        exclamation and question counts, uppercase-word ratio, negation count,
        and lexical diversity (unique words / total words).
    """
    raw = "" if text is None else str(text)
    words = _WORD_RE.findall(raw)
    word_count = len(words)
    char_count = len(raw)

    if word_count == 0:
        return {
            "char_count": float(char_count),
            "word_count": 0.0,
            "avg_word_length": 0.0,
            "exclamation_count": float(raw.count("!")),
            "question_count": float(raw.count("?")),
            "uppercase_ratio": 0.0,
            "negation_count": 0.0,
            "lexical_diversity": 0.0,
        }

    lower_words = [w.lower() for w in words]
    uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    negations = sum(1 for w in lower_words if w.replace("'", "") in _NEGATIONS)

    return {
        "char_count": float(char_count),
        "word_count": float(word_count),
        "avg_word_length": sum(len(w) for w in words) / word_count,
        "exclamation_count": float(raw.count("!")),
        "question_count": float(raw.count("?")),
        "uppercase_ratio": uppercase_words / word_count,
        "negation_count": float(negations),
        "lexical_diversity": len(set(lower_words)) / word_count,
    }


# Stable ordering of the statistic keys so generated columns are deterministic.
TEXT_STAT_KEYS = (
    "char_count",
    "word_count",
    "avg_word_length",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
    "negation_count",
    "lexical_diversity",
)


def add_text_statistics(
    df: pd.DataFrame, text_column: str, prefix: str = "text_"
) -> pd.DataFrame:
    """Expand a DataFrame with shape-based text-statistics columns.

    Args:
        df: DataFrame containing a text column.
        text_column: Name of the column with text data.
        prefix: Prefix applied to each generated feature column.

    Returns:
        DataFrame with one added column per entry in :data:`TEXT_STAT_KEYS`.
    """
    expanded = df.copy()
    stats = expanded[text_column].map(text_statistics)
    for key in TEXT_STAT_KEYS:
        expanded[f"{prefix}{key}"] = stats.map(lambda s, k=key: s[k]).astype(float)
    return expanded
