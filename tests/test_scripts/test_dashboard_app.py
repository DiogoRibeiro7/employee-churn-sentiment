"""Tests for the Streamlit risk dashboard app."""

from types import SimpleNamespace

import pandas as pd

from scripts.dashboard_app import render_risk_dashboard


def test_render_risk_dashboard_uses_streamlit_stub() -> None:
    """Dashboard should call Streamlit APIs with processed scores."""
    scores = pd.DataFrame(
        {
            "emp_id": [1],
            "week_start": pd.to_datetime(["2024-01-01"]),
            "churn_risk": [0.5],
        }
    )
    calls: dict[str, object] = {}

    def title(text: str) -> None:
        calls["title"] = text

    def dataframe(df: pd.DataFrame) -> None:
        calls.setdefault("dataframes", []).append(df)

    def subheader(text: str) -> None:
        calls["subheader"] = text

    st_stub = SimpleNamespace(title=title, dataframe=dataframe, subheader=subheader)
    render_risk_dashboard(scores, st_module=st_stub)

    assert calls["title"] == "Employee Churn Risk"
    assert isinstance(calls["dataframes"][0], pd.DataFrame)


def test_render_risk_dashboard_renders_alerts() -> None:
    scores = pd.DataFrame(
        {
            "emp_id": [1, 1, 2],
            "week_start": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-08"]),
            "churn_risk": [0.75, 0.85, 0.4],
        }
    )
    calls: dict[str, object] = {"dataframes": []}

    def title(text: str) -> None:
        calls["title"] = text

    def dataframe(df: pd.DataFrame) -> None:
        calls["dataframes"].append(df)

    def subheader(text: str) -> None:
        calls["subheader"] = text

    st_stub = SimpleNamespace(title=title, dataframe=dataframe, subheader=subheader)
    render_risk_dashboard(scores, st_module=st_stub)

    assert calls["title"] == "Employee Churn Risk"
    assert calls["subheader"] == "High-Risk Alerts (>= 70%)"
    assert len(calls["dataframes"]) == 2
