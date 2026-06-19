"""Group fairness and bias metrics.

A churn model that flags protected groups at systematically different rates can
expose an employer to legal and ethical risk. This module computes standard
group-fairness diagnostics from predictions and a sensitive attribute, with no
dependency on a specific modeling library.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _rate(numerator: int, denominator: int) -> float:
    """Safe ratio that returns 0.0 for an empty denominator."""
    return float(numerator) / float(denominator) if denominator else 0.0


def group_fairness_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    sensitive: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Compute per-group classification rates for a sensitive attribute.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels (already thresholded).
        sensitive: Group membership for each sample (e.g. gender).

    Returns:
        One row per group with ``count``, ``selection_rate`` (predicted-positive
        rate / demographic parity), ``true_positive_rate`` (recall / equal
        opportunity), ``false_positive_rate`` and ``actual_positive_rate``.

    Raises:
        ValueError: If the inputs do not share the same length.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    sensitive_arr = np.asarray(sensitive, dtype=object)
    if not (len(y_true_arr) == len(y_pred_arr) == len(sensitive_arr)):
        raise ValueError("y_true, y_pred and sensitive must have the same length")

    columns = [
        "group",
        "count",
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "actual_positive_rate",
    ]
    rows = []
    for group in pd.unique(sensitive_arr):
        mask = sensitive_arr == group
        gt = y_true_arr[mask]
        pred = y_pred_arr[mask]
        positives = gt == 1
        negatives = gt == 0
        rows.append(
            {
                "group": group,
                "count": int(mask.sum()),
                "selection_rate": _rate(int((pred == 1).sum()), len(pred)),
                "true_positive_rate": _rate(
                    int(((pred == 1) & positives).sum()), int(positives.sum())
                ),
                "false_positive_rate": _rate(
                    int(((pred == 1) & negatives).sum()), int(negatives.sum())
                ),
                "actual_positive_rate": _rate(int(positives.sum()), len(gt)),
            }
        )
    report = pd.DataFrame(rows, columns=columns)
    return report.sort_values("group").reset_index(drop=True)


def fairness_summary(
    report: pd.DataFrame,
    disparate_impact_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Summarize a :func:`group_fairness_report` into headline fairness metrics.

    Args:
        report: Per-group report produced by :func:`group_fairness_report`.
        disparate_impact_threshold: Minimum acceptable disparate-impact ratio.
            The classic "four-fifths rule" uses 0.8.

    Returns:
        Mapping with the demographic-parity difference, disparate-impact ratio
        (min selection rate / max selection rate), equal-opportunity difference
        (max TPR gap), the least- and most-selected groups, and a boolean
        ``passes_four_fifths`` flag.

    Raises:
        ValueError: If ``report`` is empty.
    """
    if report.empty:
        raise ValueError("report must contain at least one group")

    selection = report["selection_rate"]
    tpr = report["true_positive_rate"]
    max_selection = float(selection.max())
    min_selection = float(selection.min())

    disparate_impact = min_selection / max_selection if max_selection > 0 else 1.0
    return {
        "demographic_parity_difference": max_selection - min_selection,
        "disparate_impact_ratio": float(disparate_impact),
        "equal_opportunity_difference": float(tpr.max() - tpr.min()),
        "least_selected_group": report.loc[selection.idxmin(), "group"],
        "most_selected_group": report.loc[selection.idxmax(), "group"],
        "passes_four_fifths": bool(disparate_impact >= disparate_impact_threshold),
    }
