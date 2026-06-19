"""Tests for the synthetic dataset generator."""

import pandas as pd
import pytest

from employee_churn.data import make_synthetic_employee_data

EXPECTED_COLUMNS = {
    "employee_id",
    "age",
    "gender",
    "department",
    "team_id",
    "hire_date",
    "last_promotion_date",
    "num_promotions",
    "monthly_salary",
    "satisfaction_score",
    "performance_score",
    "overtime_hours",
    "feedback",
    "churned",
}


def test_shape_and_columns() -> None:
    df = make_synthetic_employee_data(n=200, seed=0)
    assert len(df) == 200
    assert EXPECTED_COLUMNS.issubset(df.columns)
    assert df["employee_id"].is_unique


def test_reproducible_with_seed() -> None:
    a = make_synthetic_employee_data(n=50, seed=7)
    b = make_synthetic_employee_data(n=50, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_target_is_binary_and_not_degenerate() -> None:
    df = make_synthetic_employee_data(n=400, seed=1)
    assert set(df["churned"].unique()).issubset({0, 1})
    # The signal should produce both classes.
    assert 0 < df["churned"].mean() < 1


def test_promotion_date_not_before_hire() -> None:
    df = make_synthetic_employee_data(n=300, seed=2)
    assert (df["last_promotion_date"] >= df["hire_date"]).all()


def test_invalid_n_raises() -> None:
    with pytest.raises(ValueError):
        make_synthetic_employee_data(n=0)
