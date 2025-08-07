"""Prediction utilities for generating churn risk scores."""

from __future__ import annotations

from typing import Any

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
