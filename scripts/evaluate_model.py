"""CLI for evaluating a saved churn model artifact on labeled data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from employee_churn.models import evaluate_models
from employee_churn.nlp.emotion import add_emotion_features
from employee_churn.nlp.sentiment import add_sentiment_scores
from scripts.predict_risk import load_model_bundle


def _prepare_evaluation_frame(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    text_column = bundle.get("text_column")
    if text_column is not None:
        prepared = add_sentiment_scores(prepared, text_column)
        prepared = add_emotion_features(prepared, text_column)
    return prepared


def evaluate_model_bundle(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    top_k: int = 10,
) -> dict[str, float]:
    """Evaluate a saved model bundle on a labeled dataset."""
    prepared = _prepare_evaluation_frame(bundle, df)
    target_column = bundle["target_column"]
    feature_columns = bundle["feature_columns"]
    missing = [
        column
        for column in [target_column, *feature_columns]
        if column not in prepared.columns
    ]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    return evaluate_models(
        {bundle["model_name"]: bundle["model"]},
        prepared[feature_columns],
        prepared[target_column],
        top_k=top_k,
    )[bundle["model_name"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved churn model.")
    parser.add_argument("model_path", help="Path to a trained model artifact.")
    parser.add_argument("input_csv", help="Path to a labeled evaluation dataset.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k cutoff for precision@k.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional destination for evaluation metrics JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    bundle = load_model_bundle(args.model_path)
    df = pd.read_csv(args.input_csv)
    metrics = evaluate_model_bundle(bundle, df, top_k=args.top_k)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
