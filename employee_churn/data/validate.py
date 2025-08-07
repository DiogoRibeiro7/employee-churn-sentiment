"""Data validation utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    """Validate that required columns are present in the DataFrame.

    Args:
        df: DataFrame to validate.
        required_columns: Column names that must be present.

    Returns:
        True if all columns are present, otherwise raises ``ValueError``.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
