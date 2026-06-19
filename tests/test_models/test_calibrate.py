"""Tests for probability calibration helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from employee_churn.models.calibrate import (
    calibrate_model,
    calibration_improvement,
    reliability_curve,
)


def _split():
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return train_test_split(X_df, pd.Series(y), test_size=0.3, random_state=0)


def test_calibrate_model_returns_probabilities() -> None:
    X_train, X_test, y_train, _ = _split()
    model = calibrate_model(
        RandomForestClassifier(n_estimators=50, random_state=0), X_train, y_train
    )
    proba = model.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_calibrate_model_invalid_method() -> None:
    X_train, _, y_train, _ = _split()
    with pytest.raises(ValueError):
        calibrate_model(
            RandomForestClassifier(random_state=0), X_train, y_train, method="bogus"
        )


def test_calibration_improvement_keys() -> None:
    X_train, X_test, y_train, y_test = _split()
    result = calibration_improvement(
        RandomForestClassifier(n_estimators=50, random_state=0),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    assert {"baseline", "calibrated", "ece_reduction", "improved"} <= set(result)
    assert "expected_calibration_error" in result["baseline"]


def test_reliability_curve_structure() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.3])
    curve = reliability_curve(y_true, y_prob, n_bins=5)
    assert {"mean_predicted", "fraction_positive", "count"} <= set(curve.columns)
    assert curve["count"].sum() == len(y_true)
