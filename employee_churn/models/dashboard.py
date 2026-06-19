"""Utilities for ranking at-risk employees."""

from __future__ import annotations

import pandas as pd


def build_risk_dashboard(
    scores: pd.DataFrame, id_column: str, top_n: int = 10
) -> pd.DataFrame:
    """Build a ranking of employees at highest risk of churn.

    The function expects the weekly churn risk scores returned by
    :func:`score_employees_weekly` and selects the most recent entry for each
    employee. The resulting dataframe is sorted by ``churn_risk`` in descending
    order and annotated with a rank.

    Args:
        scores: Dataframe with columns for employee identifier, ``week_start``
            and ``churn_risk``.
        id_column: Name of the column uniquely identifying employees.
        top_n: Number of top at-risk employees to return. Defaults to ``10``.

    Returns:
        Dataframe containing ``id_column``, ``week_start``, ``churn_risk`` and
        ``rank`` columns for the top *N* employees.

    Raises:
        KeyError: If required columns are missing from ``scores``.
        ValueError: If ``top_n`` is less than 1.
    """
    required = {id_column, "week_start", "churn_risk"}
    if not required.issubset(scores.columns):
        missing = required.difference(scores.columns)
        raise KeyError(f"scores is missing required columns: {missing}")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    latest = scores.sort_values("week_start").groupby(id_column, as_index=False).tail(1)
    ranked = (
        latest.sort_values("churn_risk", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[["rank", id_column, "week_start", "churn_risk"]]


def build_high_risk_alerts(
    scores: pd.DataFrame,
    id_column: str,
    risk_threshold: float = 0.7,
    min_consecutive_weeks: int = 1,
) -> pd.DataFrame:
    """Return employees whose recent scores breach the configured threshold."""
    required = {id_column, "week_start", "churn_risk"}
    if not required.issubset(scores.columns):
        missing = required.difference(scores.columns)
        raise KeyError(f"scores is missing required columns: {missing}")
    if not 0 <= risk_threshold <= 1:
        raise ValueError("risk_threshold must be between 0 and 1")
    if min_consecutive_weeks < 1:
        raise ValueError("min_consecutive_weeks must be at least 1")

    ordered = scores.sort_values([id_column, "week_start"]).copy()
    ordered["high_risk"] = ordered["churn_risk"] >= risk_threshold

    alert_rows: list[dict[str, object]] = []
    for employee_id, employee_scores in ordered.groupby(id_column):
        consecutive_high_risk = 0
        for is_high_risk in reversed(employee_scores["high_risk"].tolist()):
            if not is_high_risk:
                break
            consecutive_high_risk += 1

        latest = employee_scores.iloc[-1]
        if latest["high_risk"] and consecutive_high_risk >= min_consecutive_weeks:
            alert_rows.append(
                {
                    id_column: employee_id,
                    "week_start": latest["week_start"],
                    "churn_risk": latest["churn_risk"],
                    "consecutive_high_risk_weeks": consecutive_high_risk,
                    "alert_threshold": risk_threshold,
                }
            )

    alerts = pd.DataFrame(alert_rows)
    if alerts.empty:
        return alerts
    return alerts.sort_values("churn_risk", ascending=False).reset_index(drop=True)
