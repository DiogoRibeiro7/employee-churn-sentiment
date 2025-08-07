"""Tests for dashboard utilities."""

import pandas as pd

from employee_churn.models import build_risk_dashboard


def test_build_risk_dashboard() -> None:
    scores = pd.DataFrame(
        {
            "emp_id": [1, 1, 2, 2, 3, 3],
            "week_start": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-01",
                    "2024-01-08",
                ]
            ),
            "churn_risk": [0.2, 0.3, 0.5, 0.4, 0.1, 0.7],
        }
    )
    dashboard = build_risk_dashboard(scores, id_column="emp_id", top_n=2)

    assert list(dashboard.columns) == ["rank", "emp_id", "week_start", "churn_risk"]
    assert dashboard["emp_id"].tolist() == [3, 2]
    assert dashboard["rank"].tolist() == [1, 2]
    assert len(dashboard) == 2
    assert dashboard.loc[0, "week_start"] == pd.Timestamp("2024-01-08")
