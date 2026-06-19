"""Probability calibration helpers.

Churn scores are only actionable if the predicted probabilities are trustworthy:
a "0.8 risk" employee should churn roughly 80% of the time. Tree ensembles in
particular tend to be poorly calibrated, so this module wraps an estimator with
scikit-learn's :class:`~sklearn.calibration.CalibratedClassifierCV` and reports
the calibration improvement using the existing monitoring metrics.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV

from employee_churn.models.monitor import evaluate_model_calibration


def calibrate_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "isotonic",
    cv: int = 3,
) -> CalibratedClassifierCV:
    """Fit a calibrated wrapper around an estimator.

    Args:
        estimator: Unfitted base estimator (cloned internally).
        X: Training feature dataframe.
        y: Training labels.
        method: ``"isotonic"`` (non-parametric) or ``"sigmoid"`` (Platt scaling).
        cv: Number of cross-validation folds used to fit the calibrator.

    Returns:
        A fitted :class:`CalibratedClassifierCV`.

    Raises:
        ValueError: If ``method`` is not a supported calibration method.
    """
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError("method must be 'isotonic' or 'sigmoid'")
    calibrated = CalibratedClassifierCV(clone(estimator), method=method, cv=cv)
    calibrated.fit(X, y)
    return calibrated


def calibration_improvement(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    method: str = "isotonic",
    cv: int = 3,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compare calibration quality before and after calibrating an estimator.

    Both the raw and calibrated models are fit on the training data and scored on
    the held-out test data using
    :func:`~employee_churn.models.monitor.evaluate_model_calibration`.

    Args:
        estimator: Unfitted base estimator.
        X_train: Training features.
        y_train: Training labels.
        X_test: Held-out features for evaluation.
        y_test: Held-out labels for evaluation.
        method: Calibration method passed to :func:`calibrate_model`.
        cv: Calibration cross-validation folds.
        n_bins: Number of bins for the calibration metrics.

    Returns:
        Mapping with ``baseline`` and ``calibrated`` metric dictionaries and the
        ``ece_reduction`` (positive means calibration helped).
    """
    baseline = clone(estimator).fit(X_train, y_train)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_model_calibration(y_test, baseline_prob, n_bins=n_bins)

    calibrated = calibrate_model(estimator, X_train, y_train, method=method, cv=cv)
    calibrated_prob = calibrated.predict_proba(X_test)[:, 1]
    calibrated_metrics = evaluate_model_calibration(
        y_test, calibrated_prob, n_bins=n_bins
    )

    ece_reduction = float(
        baseline_metrics["expected_calibration_error"]
        - calibrated_metrics["expected_calibration_error"]
    )
    return {
        "method": method,
        "baseline": baseline_metrics,
        "calibrated": calibrated_metrics,
        "ece_reduction": ece_reduction,
        "improved": ece_reduction > 0,
        "calibrated_model": calibrated,
    }


def reliability_curve(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a reliability (calibration) curve table.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted positive-class probabilities.
        n_bins: Number of equal-width probability bins.

    Returns:
        DataFrame with one row per non-empty bin containing ``bin_lower``,
        ``bin_upper``, ``mean_predicted``, ``fraction_positive`` and ``count``.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, edges[1:-1], right=True)

    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_lower": float(edges[bin_id]),
                "bin_upper": float(edges[bin_id + 1]),
                "mean_predicted": float(y_prob_arr[mask].mean()),
                "fraction_positive": float(y_true_arr[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)
