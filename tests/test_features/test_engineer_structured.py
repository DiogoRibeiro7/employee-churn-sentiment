"""Tests for structured feature engineering."""

import pandas as pd

from employee_churn.features.engineer_structured import (
    add_career_progression_features,
    add_team_metrics,
)


def test_add_career_progression_features_calculates_durations():
    df = pd.DataFrame(
        {
            "hire": ["2020-01-01"],
            "promotion": ["2022-01-01"],
        }
    )
    engineered = add_career_progression_features(
        df, "hire", "promotion", reference_date="2023-01-01"
    )
    assert engineered.loc[0, "tenure_days"] == 1096
    assert engineered.loc[0, "days_since_promotion"] == 365


def test_add_team_metrics_counts_members():
    df = pd.DataFrame(
        {
            "employee": [1, 2, 3],
            "team": ["A", "A", "B"],
        }
    )
    engineered = add_team_metrics(df, "team")
    assert list(engineered["team_size"]) == [2, 2, 1]
