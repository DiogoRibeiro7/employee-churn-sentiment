"""Tests for model training utilities."""

import sys
import pandas as pd
from sklearn.datasets import make_classification

from employee_churn.models import (
    evaluate_models,
    explain_with_shap,
    log_experiment,
    train_baseline_models,
    train_combined_model,
    train_nlp_model,
)


def test_train_and_evaluate_models() -> None:
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X_df = pd.DataFrame(X)
    models, test_df = train_baseline_models(X_df, pd.Series(y), random_state=42)
    assert {"log_reg", "random_forest"}.issubset(models)
    metrics = evaluate_models(
        models, test_df.drop(columns=["target"]), test_df["target"], top_k=5
    )
    for result in metrics.values():
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0
        assert 0.0 <= result["precision_at_k"] <= 1.0


def test_log_experiment_monkeypatch(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Ensure experiment metrics are logged via MLflow."""

    class DummyRun:
        def __enter__(self) -> None:  # pragma: no cover - simple context
            return None

        def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
            return None

    class DummyMLflow:
        def __init__(self) -> None:
            self.logged_params: dict[str, int] = {}
            self.logged_metrics: dict[str, float] = {}
            self.run_name: str | None = None

        def set_tracking_uri(self, uri: str) -> None:  # pragma: no cover
            self.uri = uri

        def start_run(self, run_name: str | None = None) -> DummyRun:
            self.run_name = run_name
            return DummyRun()

        def log_params(self, params: dict[str, int]) -> None:
            self.logged_params.update(params)

        def log_metrics(self, metrics: dict[str, float]) -> None:
            self.logged_metrics.update(metrics)

    dummy_mlflow = DummyMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", dummy_mlflow)

    log_experiment("test_run", {"p": 1}, {"m": 0.5})

    assert dummy_mlflow.run_name == "test_run"
    assert dummy_mlflow.logged_params == {"p": 1}
    assert dummy_mlflow.logged_metrics == {"m": 0.5}


def test_train_combined_model() -> None:
    df = pd.DataFrame(
        {
            "age": [25, 40, 35, 28],
            "tenure": [1, 5, 3, 2],
            "feedback": [
                "I love my job",
                "I hate this place",
                "Happy and delighted",
                "Angry and mad",
            ],
            "target": [0, 1, 0, 1],
        }
    )
    models, test_df = train_combined_model(
        df,
        text_column="feedback",
        target_column="target",
        test_size=0.5,
        random_state=0,
    )
    assert {"log_reg", "random_forest"}.issubset(models)
    assert "sentiment" in test_df.columns
    metrics = evaluate_models(
        models, test_df.drop(columns=["target"]), test_df["target"], top_k=2
    )
    for result in metrics.values():
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0
        assert 0.0 <= result["precision_at_k"] <= 1.0


def test_explain_with_shap() -> None:
    X, y = make_classification(n_samples=50, n_features=4, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    models, test_df = train_baseline_models(X_df, pd.Series(y), random_state=42)
    shap_df = explain_with_shap(models["log_reg"], test_df.drop(columns=["target"]))
    assert isinstance(shap_df, pd.DataFrame)
    assert shap_df.shape == (test_df.shape[0], X_df.shape[1])


def test_train_nlp_model() -> None:
    df = pd.DataFrame(
        {
            "feedback": [
                "I love my job",
                "I hate this place",
                "Happy and delighted",
                "Angry and mad",
            ],
            "target": [0, 1, 0, 1],
        }
    )
    models, test_df = train_nlp_model(
        df,
        text_column="feedback",
        target_column="target",
        test_size=0.5,
        random_state=0,
    )
    assert "nlp_log_reg" in models
    metrics = evaluate_models(
        models, test_df.drop(columns=["target"]), test_df["target"], top_k=2
    )
    for result in metrics.values():
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0
        assert 0.0 <= result["precision_at_k"] <= 1.0
