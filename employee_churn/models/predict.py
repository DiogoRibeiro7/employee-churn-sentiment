"""Prediction utilities for generating churn risk scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import pandas as pd


def score_employees_weekly(
    model: Any,
    df: pd.DataFrame,
    id_column: str,
    date_column: str,
) -> pd.DataFrame:
    """Score employees' churn risk on a weekly basis.

    The function expects a dataframe containing at least an employee identifier
    and a date column alongside feature columns. It computes the positive class
    probability for each row using the provided *model* and associates the score
    with the start date of the corresponding week.

    Args:
        model: Fitted classifier implementing ``predict_proba``.
        df: Dataframe with employee features, identifiers and dates.
        id_column: Name of the column uniquely identifying employees.
        date_column: Name of the column containing date information.

    Returns:
        Dataframe with ``id_column``, ``week_start`` and ``churn_risk`` columns.

    Raises:
        AttributeError: If *model* lacks ``predict_proba``.
        KeyError: If required columns are missing from *df*.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("model must implement predict_proba")
    if not {id_column, date_column}.issubset(df.columns):
        missing = {id_column, date_column}.difference(df.columns)
        raise KeyError(f"df is missing required columns: {missing}")

    features = df.drop(columns=[id_column, date_column])
    proba = model.predict_proba(features)[:, 1]
    week_start = (
        pd.to_datetime(df[date_column]).dt.to_period("W").apply(lambda p: p.start_time)
    )
    return pd.DataFrame(
        {
            id_column: df[id_column].values,
            "week_start": week_start,
            "churn_risk": proba,
        }
    )


def export_scores_csv(
    model: Any,
    df: pd.DataFrame,
    id_column: str,
    date_column: str,
    path: Union[str, Path],
) -> pd.DataFrame:
    """Score employees weekly and export churn risk scores to CSV.

    This helper computes weekly churn probabilities using
    :func:`score_employees_weekly` and writes the resulting dataframe to
    *path*. The directory hierarchy is created if it does not already exist.

    Args:
        model: Fitted classifier implementing ``predict_proba``.
        df: Dataframe with employee features, identifiers and dates.
        id_column: Name of the column uniquely identifying employees.
        date_column: Name of the column containing date information.
        path: Destination file path for the CSV output.

    Returns:
        The scored dataframe written to disk.
    """

    scores = score_employees_weekly(model, df, id_column, date_column)
    scores = scores.copy()
    scores["week_start"] = scores["week_start"].dt.strftime("%Y-%m-%d")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    return scores
