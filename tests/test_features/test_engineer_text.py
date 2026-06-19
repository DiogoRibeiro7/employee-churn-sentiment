"""Tests for text-statistics feature engineering."""

import pandas as pd

from employee_churn.features.engineer_text import (
    TEXT_STAT_KEYS,
    add_text_statistics,
    text_statistics,
)


def test_text_statistics_basic() -> None:
    stats = text_statistics("I am NOT happy!!")
    assert stats["word_count"] == 4
    assert stats["exclamation_count"] == 2
    assert stats["negation_count"] == 1
    assert stats["uppercase_ratio"] == 0.25  # "NOT" of 4 words
    assert 0.0 < stats["lexical_diversity"] <= 1.0


def test_text_statistics_empty() -> None:
    stats = text_statistics("")
    assert stats["word_count"] == 0
    assert stats["avg_word_length"] == 0.0
    assert stats["lexical_diversity"] == 0.0


def test_add_text_statistics_columns() -> None:
    df = pd.DataFrame({"feedback": ["great work", "bad and sad?"]})
    out = add_text_statistics(df, "feedback")
    for key in TEXT_STAT_KEYS:
        assert f"text_{key}" in out.columns
    assert out.loc[1, "text_question_count"] == 1.0
    # Original column preserved, input not mutated.
    assert "feedback" in out.columns
    assert "text_word_count" not in df.columns
