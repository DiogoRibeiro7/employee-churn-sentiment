"""Sentiment analysis utilities."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from nltk import download
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    _SIA = SentimentIntensityAnalyzer()
except LookupError:  # pragma: no cover - download path
    download("vader_lexicon")
    _SIA = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> float:
    """Return the VADER compound sentiment score for a given text."""
    return _SIA.polarity_scores(text)["compound"]


def add_sentiment_scores(
    df: pd.DataFrame, text_column: str, output_column: str = "sentiment"
) -> pd.DataFrame:
    """Add sentiment scores to a DataFrame.

    Args:
        df: DataFrame containing a text column.
        text_column: Name of the column with text data.
        output_column: Name for the resulting sentiment scores column.

    Returns:
        DataFrame with an additional column containing sentiment scores.
    """
    scored = df.copy()
    scored[output_column] = scored[text_column].astype(str).map(analyze_sentiment)
    return scored
