"""Tests for the model zoo, cross-validation, and tuning helpers."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from employee_churn.models.train import (
    build_model_zoo,
    cross_validate_models,
    tune_hyperparameters,
)


def _dataset(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), pd.Series(y)


def test_build_model_zoo_contains_gradient_boosting() -> None:
    zoo = build_model_zoo(random_state=0)
    assert {"log_reg", "random_forest", "gradient_boosting"}.issubset(zoo)


def test_cross_validate_models_reports_mean_and_std() -> None:
    X, y = _dataset()
    models = {"log_reg": build_model_zoo()["log_reg"]}
    results = cross_validate_models(models, X, y, cv=3)
    summary = results["log_reg"]
    assert "roc_auc_mean" in summary and "roc_auc_std" in summary
    assert 0.0 <= summary["roc_auc_mean"] <= 1.0
    assert summary["roc_auc_std"] >= 0.0


def test_tune_hyperparameters_with_default_space() -> None:
    X, y = _dataset()
    best, params, score = tune_hyperparameters(
        RandomForestClassifier(random_state=0),
        X,
        y,
        model_name="random_forest",
        n_iter=3,
        cv=3,
        random_state=0,
    )
    assert hasattr(best, "predict_proba")
    assert "n_estimators" in params
    assert 0.0 <= score <= 1.0
