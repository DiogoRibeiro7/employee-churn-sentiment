"""Tests for group fairness metrics."""

import numpy as np
import pytest

from employee_churn.models.fairness import fairness_summary, group_fairness_report


def test_group_fairness_report_rates() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    sensitive = np.array(["a", "a", "a", "b", "b", "b"])
    report = group_fairness_report(y_true, y_pred, sensitive)
    assert set(report["group"]) == {"a", "b"}
    row_a = report.set_index("group").loc["a"]
    # Group a: 2 of 3 predicted positive.
    assert row_a["selection_rate"] == pytest.approx(2 / 3)


def test_group_fairness_report_length_mismatch() -> None:
    with pytest.raises(ValueError):
        group_fairness_report([1, 0], [1], ["a", "b"])


def test_fairness_summary_four_fifths() -> None:
    y_true = np.array([1, 1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0])
    sensitive = np.array(["a", "a", "a", "b", "b", "b"])
    report = group_fairness_report(y_true, y_pred, sensitive)
    summary = fairness_summary(report)
    assert 0.0 <= summary["disparate_impact_ratio"] <= 1.0
    assert summary["least_selected_group"] == "b"
    assert isinstance(summary["passes_four_fifths"], bool)


def test_fairness_summary_empty_raises() -> None:
    report = group_fairness_report([], [], [])
    with pytest.raises(ValueError):
        fairness_summary(report)
