"""Tests for prediction utilities."""

import pandas as pd
from sklearn.datasets import make_classification

from employee_churn.models import (
    export_scores_csv,
    score_employees_weekly,
    train_baseline_models,
)


def test_score_employees_weekly() -> None:
    X, y = make_classification(
        n_samples=20, n_features=3, n_informative=3, n_redundant=0, random_state=0
    )
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    models, _ = train_baseline_models(df, pd.Series(y), test_size=0.25, random_state=0)
    predict_df = df.copy()
    predict_df["employee_id"] = range(len(df))
    predict_df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")
    scored = score_employees_weekly(
        models["log_reg"], predict_df, id_column="employee_id", date_column="date"
    )
    assert set(scored.columns) == {"employee_id", "week_start", "churn_risk"}
    assert len(scored) == len(predict_df)
    assert scored["churn_risk"].between(0, 1).all()


def test_export_scores_csv(tmp_path) -> None:
    X, y = make_classification(
        n_samples=20, n_features=3, n_informative=3, n_redundant=0, random_state=1
    )
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    models, _ = train_baseline_models(df, pd.Series(y), test_size=0.25, random_state=1)
    predict_df = df.copy()
    predict_df["employee_id"] = range(len(df))
    predict_df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")
    output = tmp_path / "scores.csv"
    scored = export_scores_csv(
        models["log_reg"], predict_df, "employee_id", "date", output
    )
    assert output.exists()
    loaded = pd.read_csv(output)
    pd.testing.assert_frame_equal(loaded, scored)
