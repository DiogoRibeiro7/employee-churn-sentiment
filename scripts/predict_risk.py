"""CLI for scoring employee churn risk from a saved model artifact."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from employee_churn.models import export_scores_csv
from employee_churn.nlp.emotion import add_emotion_features
from employee_churn.nlp.sentiment import add_sentiment_scores


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    """Load a serialized model bundle."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def _prepare_scoring_frame(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    text_column = bundle.get("text_column")
    if text_column is not None:
        prepared = add_sentiment_scores(prepared, text_column)
        prepared = add_emotion_features(prepared, text_column)

    feature_columns = bundle["feature_columns"]
    missing = [column for column in feature_columns if column not in prepared.columns]
    if missing:
        raise ValueError(f"Input data is missing required feature columns: {missing}")
    return prepared


def score_dataset(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    id_column: str,
    date_column: str,
    output_path: str | Path,
) -> pd.DataFrame:
    """Score a dataset and export weekly churn-risk scores to CSV."""
    prepared = _prepare_scoring_frame(bundle, df)
    scoring_df = prepared[[id_column, date_column, *bundle["feature_columns"]]].copy()
    return export_scores_csv(
        bundle["model"],
        scoring_df,
        id_column=id_column,
        date_column=date_column,
        path=output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score employee churn risk.")
    parser.add_argument("model_path", help="Path to a trained model artifact.")
    parser.add_argument("input_csv", help="Path to the scoring dataset CSV.")
    parser.add_argument("id_column", help="Employee identifier column.")
    parser.add_argument("date_column", help="Date column used for weekly scoring.")
    parser.add_argument("output_csv", help="Destination CSV path for risk scores.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    bundle = load_model_bundle(args.model_path)
    df = pd.read_csv(args.input_csv)
    score_dataset(
        bundle,
        df,
        id_column=args.id_column,
        date_column=args.date_column,
        output_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
