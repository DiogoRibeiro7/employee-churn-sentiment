"""Basic emotion extraction utilities."""

from __future__ import annotations

from collections import Counter
from typing import Dict

import pandas as pd

EMOTION_LEXICON: Dict[str, set[str]] = {
    "joy": {"happy", "joy", "pleased", "delighted"},
    "fear": {"afraid", "scared", "fear", "terrified"},
    "trust": {"trust", "faith", "confidence"},
    "anger": {"angry", "mad", "annoyed", "furious"},
}


def detect_emotions(text: str) -> Dict[str, int]:
    """Detect simple emotions present in text using a keyword lexicon.

    Args:
        text: Input text to analyse.

    Returns:
        A mapping from emotion to occurrence count.
    """
    words = {word.strip(".,!?").lower() for word in text.split()}
    counts = {
        emotion: len(words & lexicon) for emotion, lexicon in EMOTION_LEXICON.items()
    }
    return counts


def add_emotion_features(
    df: pd.DataFrame, text_column: str, prefix: str = "emotion_"
) -> pd.DataFrame:
    """Expand a DataFrame with emotion count features."""
    expanded = df.copy()
    counts = expanded[text_column].astype(str).map(detect_emotions)
    for emotion in EMOTION_LEXICON:
        expanded[f"{prefix}{emotion}"] = counts.map(lambda x: x.get(emotion, 0))
    return expanded
