"""Tests for the comprehensive evaluation report."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from employee_churn.models.evaluate import evaluate_model, flatten_report


def _fitted_model(n: int = 200):
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y)
    model = LogisticRegression(max_iter=1000).fit(X_df, y_s)
    return model, X_df, y_s


def test_evaluate_model_discrimination_and_calibration() -> None:
    model, X, y = _fitted_model()
    report = evaluate_model(model, X, y, top_k=10)
    assert {"roc_auc", "f1", "precision_at_k"} <= set(report["discrimination"])
    assert "expected_calibration_error" in report["calibration"]
    assert "fairness" not in report


def test_evaluate_model_with_fairness() -> None:
    model, X, y = _fitted_model()
    sensitive = pd.Series(np.where(X["f0"] >= 0, "a", "b"), index=X.index)
    report = evaluate_model(model, X, y, sensitive=sensitive)
    assert "fairness" in report
    assert "disparate_impact_ratio" in report["fairness"]["summary"]
    assert set(report["fairness"]["by_group"]["group"]) == {"a", "b"}


def test_evaluate_model_requires_predict_proba() -> None:
    class Dummy:
        pass

    with pytest.raises(AttributeError):
        evaluate_model(Dummy(), pd.DataFrame({"a": [1]}), pd.Series([0]))


def test_flatten_report() -> None:
    model, X, y = _fitted_model()
    sensitive = pd.Series(np.where(X["f0"] >= 0, "a", "b"), index=X.index)
    flat = flatten_report(evaluate_model(model, X, y, sensitive=sensitive))
    assert "roc_auc" in flat
    assert "calibration_brier_score" in flat
    assert "fairness_disparate_impact_ratio" in flat
    assert all(isinstance(v, float) for v in flat.values())
