"""CLI for training and saving churn models."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone

from employee_churn.models import (
    evaluate_models,
    train_baseline_models,
)
from employee_churn.nlp.emotion import add_emotion_features
from employee_churn.nlp.sentiment import add_sentiment_scores


def _prepare_feature_frame(
    df: pd.DataFrame,
    target_column: str,
    text_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    prepared = df.copy()
    if text_column is not None:
        prepared = add_sentiment_scores(prepared, text_column)
        prepared = add_emotion_features(prepared, text_column)
        X = prepared.drop(columns=[text_column, target_column])
    else:
        X = prepared.drop(columns=[target_column])
    X = X.select_dtypes(include=["number", "bool"])
    identifier_columns = [
        column for column in X.columns if column == "id" or column.endswith("_id")
    ]
    if identifier_columns:
        X = X.drop(columns=identifier_columns)
    y = prepared[target_column]
    return X, y


def train_model_bundle(
    df: pd.DataFrame,
    target_column: str,
    model_name: str = "log_reg",
    text_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 0,
    top_k: int = 10,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Train a model, evaluate it, and return a serializable artifact bundle."""
    X, y = _prepare_feature_frame(df, target_column, text_column=text_column)
    models, test_df = train_baseline_models(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    if model_name not in models:
        raise ValueError(
            f"model_name must be one of {sorted(models)}; got '{model_name}'"
        )

    metrics = evaluate_models(
        {model_name: models[model_name]},
        test_df.drop(columns=[target_column if text_column else "target"]),
        test_df[target_column if text_column else "target"],
        top_k=top_k,
    )[model_name]

    X_full, y_full = _prepare_feature_frame(df, target_column, text_column=text_column)
    final_model = clone(models[model_name])
    final_model.fit(X_full, y_full)

    bundle = {
        "model": final_model,
        "model_name": model_name,
        "training_mode": "combined" if text_column is not None else "baseline",
        "target_column": target_column,
        "text_column": text_column,
        "feature_columns": X_full.columns.tolist(),
        "metrics": metrics,
    }
    return bundle, metrics


def save_model_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Serialize a trained model bundle to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(bundle, handle)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a churn model artifact.")
    parser.add_argument("input_csv", help="Path to the training dataset CSV.")
    parser.add_argument("target_column", help="Binary target column name.")
    parser.add_argument("output_model", help="Destination path for the model artifact.")
    parser.add_argument(
        "--model-name",
        default="log_reg",
        help="Model to train: log_reg, random_forest, or xgboost if available.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Optional text column to include via sentiment and emotion features.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional path for writing evaluation metrics as JSON.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    bundle, metrics = train_model_bundle(
        df,
        target_column=args.target_column,
        model_name=args.model_name,
        text_column=args.text_column,
        test_size=args.test_size,
        random_state=args.random_state,
        top_k=args.top_k,
    )
    save_model_bundle(bundle, args.output_model)

    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
