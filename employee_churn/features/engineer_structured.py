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


# Default tenure bucket boundaries in years (right-open intervals). The final
# bucket captures everyone above the last boundary.
DEFAULT_TENURE_BANDS = (1.0, 3.0, 7.0)
DEFAULT_TENURE_LABELS = ("0-1y", "1-3y", "3-7y", "7y+")


def add_tenure_bands(
    df: pd.DataFrame,
    tenure_days_col: str = "tenure_days",
    bands_years: tuple[float, ...] = DEFAULT_TENURE_BANDS,
    labels: tuple[str, ...] = DEFAULT_TENURE_LABELS,
    output_column: str = "tenure_band",
) -> pd.DataFrame:
    """Bucket continuous tenure into ordinal seniority bands.

    Tenure bands often capture churn risk better than raw days because risk is
    highly non-linear (e.g. very high in the first year, then dropping).

    Args:
        df: DataFrame containing a numeric tenure-in-days column.
        tenure_days_col: Name of the tenure-in-days column.
        bands_years: Upper boundaries (in years) for the interior buckets.
        labels: Labels for each band; must contain ``len(bands_years) + 1``
            entries.
        output_column: Name of the resulting categorical column.

    Returns:
        DataFrame with an added ordered categorical ``output_column``.

    Raises:
        ValueError: If ``labels`` length does not match ``bands_years``.
    """
    if len(labels) != len(bands_years) + 1:
        raise ValueError("labels must contain len(bands_years) + 1 entries")
    engineered = df.copy()
    edges = [-float("inf"), *(b * 365.0 for b in bands_years), float("inf")]
    engineered[output_column] = pd.cut(
        engineered[tenure_days_col],
        bins=edges,
        labels=list(labels),
        ordered=True,
    )
    return engineered


def add_promotion_velocity(
    df: pd.DataFrame,
    num_promotions_col: str,
    tenure_days_col: str = "tenure_days",
    output_column: str = "promotions_per_year",
) -> pd.DataFrame:
    """Compute promotion velocity (promotions per year of tenure).

    A stalled career — few promotions relative to tenure — is a common churn
    driver. Employees with less than a day of tenure receive a velocity of 0 to
    avoid division blow-ups.

    Args:
        df: DataFrame with promotion-count and tenure columns.
        num_promotions_col: Column holding the cumulative promotion count.
        tenure_days_col: Column holding tenure in days.
        output_column: Name of the resulting column.

    Returns:
        DataFrame with an added ``output_column`` of promotions per year.
    """
    engineered = df.copy()
    tenure_years = engineered[tenure_days_col] / 365.0
    safe_years = tenure_years.where(tenure_years >= (1.0 / 365.0), other=pd.NA)
    velocity = engineered[num_promotions_col] / safe_years
    engineered[output_column] = velocity.fillna(0.0).astype(float)
    return engineered


def add_compensation_features(
    df: pd.DataFrame,
    salary_col: str,
    group_col: str,
    ratio_column: str = "salary_to_peer_ratio",
    zscore_column: str = "salary_peer_zscore",
) -> pd.DataFrame:
    """Add peer-relative compensation features.

    Absolute salary is a weak churn signal; *relative* pay versus peers (same
    department, role, or team) is far stronger. This computes the ratio of an
    employee's salary to their peer-group median and the within-group z-score.

    Args:
        df: DataFrame with salary and grouping columns.
        salary_col: Column with the compensation figure.
        group_col: Column defining the peer group (e.g. ``department``).
        ratio_column: Name of the salary-to-peer-median ratio column.
        zscore_column: Name of the within-group z-score column.

    Returns:
        DataFrame with the two added compensation columns.
    """
    engineered = df.copy()
    grouped = engineered.groupby(group_col)[salary_col]
    peer_median = grouped.transform("median")
    peer_mean = grouped.transform("mean")
    peer_std = grouped.transform("std").replace(0.0, pd.NA)

    engineered[ratio_column] = (engineered[salary_col] / peer_median).astype(float)
    engineered[zscore_column] = (
        ((engineered[salary_col] - peer_mean) / peer_std).fillna(0.0).astype(float)
    )
    return engineered
