"""Data loading utilities for the employee churn package."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def load_hr_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load structured HR data from a CSV file.

    Args:
        path: Path to the CSV file containing HR data.

    Returns:
        DataFrame with the loaded HR data.
    """
    return pd.read_csv(Path(path))


def load_feedback_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load employee feedback data from a CSV file.

    Args:
        path: Path to the CSV file containing feedback text.

    Returns:
        DataFrame with the loaded feedback data.
    """
    return pd.read_csv(Path(path))
