"""Monitoring utilities for drift detection and calibration quality."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def _population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Compute PSI for two numeric series."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0

    if ref.nunique() == 1 and cur.nunique() == 1 and ref.iloc[0] == cur.iloc[0]:
        return 0.0

    combined_min = min(ref.min(), cur.min())
    combined_max = max(ref.max(), cur.max())
    if combined_min == combined_max:
        return 0.0

    edges = np.linspace(combined_min, combined_max, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    epsilon = 1e-6
    ref_ratio = np.clip(ref_counts / len(ref), epsilon, None)
    cur_ratio = np.clip(cur_counts / len(cur), epsilon, None)
    return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_threshold: float = 0.2,
    bins: int = 10,
) -> pd.DataFrame:
    """Compare numeric feature distributions and flag drifted features."""
    common_columns = [
        column
        for column in reference_df.columns.intersection(current_df.columns)
        if pd.api.types.is_numeric_dtype(reference_df[column])
        and pd.api.types.is_numeric_dtype(current_df[column])
    ]
    report_rows: list[dict[str, Any]] = []
    for column in common_columns:
        psi = _population_stability_index(
            reference_df[column], current_df[column], bins=bins
        )
        report_rows.append(
            {
                "feature": column,
                "reference_mean": float(reference_df[column].mean()),
                "current_mean": float(current_df[column].mean()),
                "reference_std": float(reference_df[column].std(ddof=0)),
                "current_std": float(current_df[column].std(ddof=0)),
                "psi": psi,
                "drift_detected": psi >= drift_threshold,
            }
        )

    report = pd.DataFrame(report_rows)
    if report.empty:
        return report
    return report.sort_values("psi", ascending=False).reset_index(drop=True)


def evaluate_model_calibration(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute summary calibration metrics for predicted probabilities."""
    y_true_array = np.asarray(y_true)
    y_prob_array = np.asarray(y_prob, dtype=float)
    if len(y_true_array) != len(y_prob_array):
        raise ValueError("y_true and y_prob must have the same length")
    if len(y_true_array) == 0:
        raise ValueError("y_true and y_prob must not be empty")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    y_prob_array = np.clip(y_prob_array, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_array, bin_edges[1:-1], right=True)

    ece = 0.0
    max_error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        avg_confidence = float(y_prob_array[mask].mean())
        avg_outcome = float(y_true_array[mask].mean())
        error = abs(avg_confidence - avg_outcome)
        ece += error * (mask.sum() / len(y_prob_array))
        max_error = max(max_error, error)

    return {
        "brier_score": float(brier_score_loss(y_true_array, y_prob_array)),
        "expected_calibration_error": float(ece),
        "max_calibration_error": float(max_error),
    }


def summarize_monitoring_status(
    drift_report: pd.DataFrame,
    calibration_metrics: dict[str, float],
    calibration_threshold: float = 0.1,
) -> dict[str, Any]:
    """Build a compact monitoring summary with alert flags."""
    drifted_features = (
        drift_report.loc[drift_report["drift_detected"], "feature"].tolist()
        if not drift_report.empty
        else []
    )
    calibration_alert = (
        calibration_metrics["expected_calibration_error"] >= calibration_threshold
    )
    return {
        "drifted_features": drifted_features,
        "drift_alert": bool(drifted_features),
        "calibration_alert": calibration_alert,
        "calibration_metrics": calibration_metrics,
    }
