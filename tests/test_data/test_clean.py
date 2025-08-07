"""Tests for data cleaning utilities."""

import pandas as pd

from employee_churn.data.clean import anonymize_text, clean_hr_data


def test_clean_hr_data_standardizes_columns_and_drops_duplicates():
    df = pd.DataFrame({"Employee ID": [1, 1], "Name": ["Alice", "Alice"]})
    cleaned = clean_hr_data(df)
    assert list(cleaned.columns) == ["employee_id", "name"]
    assert len(cleaned) == 1


def test_anonymize_text_replaces_names():
    df = pd.DataFrame({"feedback": ["Alice was great"], "other": ["Bob helped"]})
    anonymized = anonymize_text(df, ["feedback", "other"])
    assert "<NAME>" in anonymized.loc[0, "feedback"]
    assert "<NAME>" in anonymized.loc[0, "other"]
