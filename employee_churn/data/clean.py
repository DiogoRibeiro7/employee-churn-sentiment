"""Data cleaning utilities."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def clean_hr_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize HR data fields.

    The function normalizes column names to snake case and drops duplicate
    rows.

    Args:
        df: Raw HR DataFrame.

    Returns:
        Cleaned DataFrame with standardized columns.
    """
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower().replace(" ", "_") for col in cleaned.columns]
    cleaned = cleaned.drop_duplicates()
    return cleaned


def anonymize_text(df: pd.DataFrame, text_columns: Iterable[str]) -> pd.DataFrame:
    """Anonymize personally identifiable information in text columns.

    This uses a naive pattern that replaces capitalized words with the token
    ``"<NAME>"``. It is intended as a placeholder for more robust anonymization.

    Args:
        df: DataFrame containing text columns to anonymize.
        text_columns: Column names that should be anonymized.

    Returns:
        DataFrame with anonymized text columns.
    """
    anonymized = df.copy()
    pattern = re.compile(r"\b[A-Z][a-z]+\b")
    for column in text_columns:
        anonymized[column] = anonymized[column].str.replace(
            pattern, "<NAME>", regex=True
        )
    return anonymized
