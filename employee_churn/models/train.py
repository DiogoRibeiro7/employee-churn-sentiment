"""Training helpers for baseline churn models."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from employee_churn.nlp.emotion import add_emotion_features
from employee_churn.nlp.sentiment import add_sentiment_scores

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - xgboost optional
    XGBClassifier = None  # type: ignore


ModelDict = Dict[str, Any]


def _numeric_features(df: pd.DataFrame, drop_columns: Sequence[str]) -> pd.DataFrame:
    """Drop the given columns and keep only numeric/boolean feature columns.

    This makes the text-based training paths robust to categorical feature
    columns (e.g. a dominant-emotion label) that cannot be fed to a plain
    scikit-learn estimator.
    """
    features = df.drop(columns=list(drop_columns))
    return features.select_dtypes(include=["number", "bool"])


def build_model_zoo(random_state: int = 0) -> ModelDict:
    """Instantiate the supported classifier zoo.

    Includes logistic regression, random forest, gradient boosting, and
    histogram gradient boosting; XGBoost is added when the optional dependency
    is installed.

    Args:
        random_state: Seed shared by every estimator for reproducibility.

    Returns:
        Mapping of model name to an unfitted estimator instance.
    """
    models: ModelDict = {
        "log_reg": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=random_state
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    return models


def train_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 0,
    models: ModelDict | None = None,
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
        models: Optional mapping of unfitted estimators to train. Defaults to
            logistic regression, random forest, and XGBoost (when available).
            Pass :func:`build_model_zoo` to train the full classifier set.

    Returns:
        A tuple containing a dictionary of trained models and a dataframe
        with the test features and labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if models is None:
        models = {
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
    X = _numeric_features(features, [text_column, target_column])
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
    X = _numeric_features(features, [text_column, target_column])
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


def cross_validate_models(
    models: ModelDict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: Sequence[str] = ("roc_auc", "f1"),
    random_state: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Stratified k-fold cross-validation for a set of models.

    Reports the mean and standard deviation of each scoring metric across folds,
    which is a far more reliable estimate of generalization than a single
    train/test split.

    Args:
        models: Mapping of model name to an unfitted estimator.
        X: Feature dataframe.
        y: Target labels.
        cv: Number of stratified folds.
        scoring: Scikit-learn scoring metric names to evaluate.
        random_state: Seed for the fold shuffling.

    Returns:
        Mapping of model name to ``{metric}_mean`` / ``{metric}_std`` floats.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        cv_scores = cross_validate(
            clone(model),
            X,
            y,
            cv=splitter,
            scoring=list(scoring),
            n_jobs=None,
        )
        summary: Dict[str, float] = {}
        for metric in scoring:
            fold_scores = cv_scores[f"test_{metric}"]
            summary[f"{metric}_mean"] = float(np.mean(fold_scores))
            summary[f"{metric}_std"] = float(np.std(fold_scores))
        results[name] = summary
    return results


# Reasonable default search spaces keyed by model name. Used when the caller
# does not supply an explicit ``param_distributions``.
DEFAULT_PARAM_DISTRIBUTIONS: Dict[str, Dict[str, Sequence[Any]]] = {
    "log_reg": {"C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 4, 8, 16],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    },
    "hist_gradient_boosting": {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_iter": [100, 200, 400],
    },
}


def tune_hyperparameters(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    param_distributions: Dict[str, Sequence[Any]] | None = None,
    model_name: str | None = None,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 0,
) -> tuple[BaseEstimator, Dict[str, Any], float]:
    """Tune an estimator with randomized search over a parameter grid.

    Args:
        estimator: Unfitted scikit-learn estimator to tune.
        X: Feature dataframe.
        y: Target labels.
        param_distributions: Search space. Falls back to
            :data:`DEFAULT_PARAM_DISTRIBUTIONS` for ``model_name`` when omitted.
        model_name: Key into the default search spaces.
        n_iter: Number of parameter settings sampled.
        cv: Number of cross-validation folds per candidate.
        scoring: Scikit-learn scoring metric to optimize.
        random_state: Seed for the sampler and fold shuffling.

    Returns:
        A tuple of (refit best estimator, best parameters, best CV score).

    Raises:
        ValueError: If no search space is provided or resolvable.
    """
    if param_distributions is None:
        if model_name is None or model_name not in DEFAULT_PARAM_DISTRIBUTIONS:
            raise ValueError(
                "Provide param_distributions or a known model_name from "
                f"{sorted(DEFAULT_PARAM_DISTRIBUTIONS)}"
            )
        param_distributions = DEFAULT_PARAM_DISTRIBUTIONS[model_name]

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        clone(estimator),
        param_distributions=dict(param_distributions),
        n_iter=n_iter,
        cv=splitter,
        scoring=scoring,
        random_state=random_state,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)
