"""Tests for monitoring utilities."""

import pandas as pd

from employee_churn.models import (
    detect_feature_drift,
    evaluate_model_calibration,
    summarize_monitoring_status,
)


def test_detect_feature_drift_flags_shifted_feature() -> None:
    reference = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4, 5, 6],
            "performance": [3, 3, 4, 4, 5, 5],
        }
    )
    current = pd.DataFrame(
        {
            "tenure": [7, 8, 9, 10, 11, 12],
            "performance": [3, 3, 4, 4, 5, 5],
        }
    )

    report = detect_feature_drift(reference, current, drift_threshold=0.2, bins=5)

    assert list(report.columns) == [
        "feature",
        "reference_mean",
        "current_mean",
        "reference_std",
        "current_std",
        "psi",
        "drift_detected",
    ]
    assert bool(report.loc[report["feature"] == "tenure", "drift_detected"].iloc[0])
    assert not bool(
        report.loc[report["feature"] == "performance", "drift_detected"].iloc[0]
    )


def test_evaluate_model_calibration_and_summary() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_prob = pd.Series([0.1, 0.2, 0.8, 0.9])

    calibration = evaluate_model_calibration(y_true, y_prob, n_bins=4)
    summary = summarize_monitoring_status(
        pd.DataFrame(
            {
                "feature": ["tenure"],
                "drift_detected": [True],
            }
        ),
        calibration,
        calibration_threshold=0.2,
    )

    assert 0.0 <= calibration["brier_score"] <= 1.0
    assert 0.0 <= calibration["expected_calibration_error"] <= 1.0
    assert summary["drifted_features"] == ["tenure"]
    assert summary["drift_alert"] is True
    assert summary["calibration_alert"] is False
