"""Training helpers for baseline churn models."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from employee_churn.nlp.emotion import add_emotion_features
from employee_churn.nlp.sentiment import add_sentiment_scores

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - xgboost optional
    XGBClassifier = None  # type: ignore


ModelDict = Dict[str, Any]


def train_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 0,
) -> tuple[ModelDict, pd.DataFrame]:
    """Train baseline churn models.

    Splits the data into train and test sets, fits logistic regression,
    random forest, and XGBoost classifiers, and returns the fitted models
    alongside the held-out test data.

    Args:
        X: Feature dataframe.
        y: Target labels.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple containing a dictionary of trained models and a dataframe
        with the test features and labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    models: ModelDict = {
        "log_reg": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    for model in models.values():
        model.fit(X_train, y_train)
    test_df = pd.DataFrame(X_test)
    test_df["target"] = y_test.values
    return models, test_df


def train_nlp_model(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 0,
) -> tuple[ModelDict, pd.DataFrame]:
    """Train a logistic regression model using only text-derived features.

    The function generates sentiment and basic emotion features from a text
    column, trains a logistic regression classifier, and returns the trained
    model along with a held-out test set.

    Args:
        df: Input dataframe containing text and target columns.
        text_column: Name of the column with text data.
        target_column: Name of the target column.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple with a dictionary containing the trained model and a dataframe
        of test features and labels.
    """
    features = add_sentiment_scores(df, text_column)
    features = add_emotion_features(features, text_column)
    X = features.drop(columns=[text_column, target_column])
    y = features[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    test_df = pd.DataFrame(X_test)
    test_df[target_column] = y_test.values
    return {"nlp_log_reg": model}, test_df


def train_combined_model(
    df: pd.DataFrame,
    text_column: str,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 0,
) -> tuple[ModelDict, pd.DataFrame]:
    """Train baseline models using both structured and text features.

    This function augments structured features with sentiment and emotion
    signals extracted from a text column before delegating to
    :func:`train_baseline_models` for fitting the baseline classifiers.

    Args:
        df: Input dataframe containing structured features, a text column and
            the target label.
        text_column: Name of the column with unstructured text data.
        target_column: Name of the target column.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple containing the trained models and a dataframe with held-out
        test features and labels.
    """
    features = add_sentiment_scores(df, text_column)
    features = add_emotion_features(features, text_column)
    X = features.drop(columns=[text_column, target_column])
    y = features[target_column]
    models, test_df = train_baseline_models(
        X, y, test_size=test_size, random_state=random_state
    )
    test_df.rename(columns={"target": target_column}, inplace=True)
    return models, test_df


def precision_at_k(y_true: pd.Series, y_scores: pd.Series, k: int) -> float:
    """Compute precision at top *k* predictions.

    Args:
        y_true: True binary labels.
        y_scores: Predicted positive class probabilities.
        k: Number of top instances to evaluate. If *k* exceeds the number of
            samples, the full sample is used.

    Returns:
        Precision among the top-*k* highest scoring samples.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(y_true))
    topk_indices = y_scores.nlargest(k).index
    return float(y_true.loc[topk_indices].sum() / k)


def evaluate_models(
    models: ModelDict, X: pd.DataFrame, y: pd.Series, top_k: int = 10
) -> Dict[str, Dict[str, float]]:
    """Evaluate trained models using multiple metrics.

    Args:
        models: Dictionary of model instances.
        X: Feature dataframe for evaluation.
        y: True labels.
        top_k: Number of highest-scoring samples to use for precision@k.

    Returns:
        Mapping of model names to metric dictionaries containing ROC-AUC, F1,
        and Precision@K scores.
    """
    scores: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        y_scores = pd.Series(proba, index=y.index)
        scores[name] = {
            "roc_auc": float(roc_auc_score(y, proba)),
            "f1": float(f1_score(y, preds)),
            "precision_at_k": precision_at_k(y, y_scores, top_k),
        }
    return scores
