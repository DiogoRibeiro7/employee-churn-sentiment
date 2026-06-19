"""Streamlit dashboard for visualizing employee churn risk."""

from __future__ import annotations

from types import ModuleType

import pandas as pd

from employee_churn.models import build_high_risk_alerts, build_risk_dashboard


def render_risk_dashboard(
    scores: pd.DataFrame,
    id_column: str = "emp_id",
    risk_threshold: float = 0.7,
    st_module: ModuleType | None = None,
) -> None:
    """Render a churn risk dashboard using Streamlit.

    Args:
        scores: Weekly churn-risk scores per employee.
        id_column: Name of the employee identifier column.
        risk_threshold: Threshold above which a score triggers an alert.
        st_module: Optional Streamlit-like module with ``title`` and ``dataframe``.
            If ``None``, the real Streamlit package is imported.

    Raises:
        RuntimeError: If Streamlit is not installed when ``st_module`` is ``None``.
    """
    if st_module is None:
        try:
            import streamlit as st_module  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError("streamlit is required to render the dashboard") from exc

    dashboard = build_risk_dashboard(scores, id_column=id_column)
    alerts = build_high_risk_alerts(
        scores, id_column=id_column, risk_threshold=risk_threshold
    )
    st_module.title("Employee Churn Risk")
    st_module.dataframe(dashboard)
    if not alerts.empty:
        st_module.subheader(f"High-Risk Alerts (>= {risk_threshold:.0%})")
        st_module.dataframe(alerts)


def main() -> None:
    """Entry point for running the Streamlit app."""
    example_scores = pd.DataFrame(
        {
            "emp_id": [1, 2],
            "week_start": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "churn_risk": [0.2, 0.8],
        }
    )
    render_risk_dashboard(example_scores, id_column="emp_id")


if __name__ == "__main__":
    main()
