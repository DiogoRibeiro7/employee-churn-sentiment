"""Comprehensive model evaluation.

A single entry point that combines the three evaluation lenses already in the
package — discrimination (ROC-AUC / F1 / precision@k), probability calibration,
and (optionally) group fairness — into one structured report. This is the
"comprehensive metrics + fairness checks + explainability" deliverable, built on
top of the focused helpers rather than duplicating them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from employee_churn.models.fairness import fairness_summary, group_fairness_report
from employee_churn.models.monitor import evaluate_model_calibration
from employee_churn.models.train import evaluate_models


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 10,
    sensitive: Optional[pd.Series] = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Produce a comprehensive evaluation report for a fitted model.

    Args:
        model: Fitted classifier implementing ``predict_proba``.
        X: Feature matrix for evaluation.
        y: True binary labels.
        top_k: Number of top-scoring samples for precision@k.
        sensitive: Optional sensitive-attribute series for fairness diagnostics.
            When supplied, a fairness report and summary are included.
        threshold: Probability cutoff used to derive hard predictions for the
            fairness analysis.
        n_bins: Number of bins for the calibration metrics.

    Returns:
        Mapping with ``discrimination`` and ``calibration`` metric dictionaries,
        and a ``fairness`` block when ``sensitive`` is provided.

    Raises:
        AttributeError: If ``model`` lacks ``predict_proba``.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("model must implement predict_proba")

    proba = model.predict_proba(X)[:, 1]
    report: Dict[str, Any] = {
        "discrimination": evaluate_models({"model": model}, X, y, top_k=top_k)["model"],
        "calibration": evaluate_model_calibration(y, proba, n_bins=n_bins),
    }

    if sensitive is not None:
        preds = (proba >= threshold).astype(int)
        group_report = group_fairness_report(
            np.asarray(y), preds, np.asarray(sensitive)
        )
        report["fairness"] = {
            "by_group": group_report,
            "summary": fairness_summary(group_report),
        }
    return report


def flatten_report(report: Dict[str, Any]) -> Dict[str, float]:
    """Flatten an evaluation report into a single metric mapping.

    Useful for logging to MLflow or writing a compact metrics row. Nested
    sections are prefixed (e.g. ``calibration_brier_score``); the per-group
    fairness table is omitted, but its headline summary is flattened under a
    ``fairness_`` prefix.

    Args:
        report: A report produced by :func:`evaluate_model`.

    Returns:
        Flat mapping of metric name to float.
    """
    flat: Dict[str, float] = {}
    for key, value in report.get("discrimination", {}).items():
        flat[key] = float(value)
    for key, value in report.get("calibration", {}).items():
        flat[f"calibration_{key}"] = float(value)
    summary = report.get("fairness", {}).get("summary", {})
    for key, value in summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[f"fairness_{key}"] = float(value)
    return flat
