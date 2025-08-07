"""Structured feature engineering utilities."""

from __future__ import annotations

import pandas as pd


def add_career_progression_features(
    df: pd.DataFrame,
    hire_date_col: str,
    last_promotion_col: str,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add basic career progression features.

    This function computes tenure in days and days since last promotion
    based on provided date columns.

    Args:
        df: Input DataFrame containing hire and promotion dates.
        hire_date_col: Column name with hire/start dates.
        last_promotion_col: Column name with last promotion dates.
        reference_date: Optional reference date for calculations. If not
            provided, today's date is used.

    Returns:
        DataFrame with ``tenure_days`` and ``days_since_promotion`` columns
        added.
    """
    engineered = df.copy()
    engineered[hire_date_col] = pd.to_datetime(engineered[hire_date_col])
    engineered[last_promotion_col] = pd.to_datetime(engineered[last_promotion_col])
    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()
    else:
        reference_date = pd.to_datetime(reference_date)
    engineered["tenure_days"] = (reference_date - engineered[hire_date_col]).dt.days
    engineered["days_since_promotion"] = (
        reference_date - engineered[last_promotion_col]
    ).dt.days
    return engineered


def add_team_metrics(df: pd.DataFrame, team_column: str) -> pd.DataFrame:
    """Add simple team metrics such as team size.

    Args:
        df: DataFrame containing a team identifier column.
        team_column: Column representing team membership.

    Returns:
        DataFrame with an additional ``team_size`` column indicating the
        number of members in each team.
    """
    engineered = df.copy()
    engineered["team_size"] = engineered.groupby(team_column)[team_column].transform(
        "count"
    )
    return engineered
