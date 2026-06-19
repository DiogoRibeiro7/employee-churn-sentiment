"""Tests for SHAP explanations, including tree (multi-output) models."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from employee_churn.models.explain import explain_with_shap


def test_explain_tree_model_returns_2d_frame() -> None:
    X, y = make_classification(n_samples=80, n_features=5, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X_df, y)

    shap_df = explain_with_shap(model, X_df.head(20))

    # Tree models emit per-class SHAP values; the helper must collapse to 2-D.
    assert shap_df.shape == (20, X_df.shape[1])
    assert list(shap_df.columns) == list(X_df.columns)
