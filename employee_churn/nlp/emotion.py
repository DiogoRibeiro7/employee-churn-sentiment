"""Lexicon-based emotion extraction.

The lexicon follows Plutchik's eight primary emotions. It is intentionally
lightweight (keyword matching) and is meant as an interpretable baseline rather
than a replacement for a trained emotion classifier.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

# Plutchik's eight primary emotions. Keep the four original keys
# (joy, fear, trust, anger) first for backward compatibility with existing
# feature columns and tests.
EMOTION_LEXICON: Dict[str, set[str]] = {
    "joy": {"happy", "joy", "pleased", "delighted", "glad", "proud", "excited"},
    "fear": {"afraid", "scared", "fear", "terrified", "anxious", "worried"},
    "trust": {"trust", "faith", "confidence", "confident", "reliable", "supported"},
    "anger": {"angry", "mad", "annoyed", "furious", "frustrated", "irritated"},
    "sadness": {"sad", "unhappy", "depressed", "disappointed", "exhausted", "tired"},
    "disgust": {"disgusted", "disgust", "awful", "horrible", "dislike", "hate"},
    "surprise": {"surprised", "shocked", "amazed", "astonished", "unexpected"},
    "anticipation": {"hopeful", "eager", "looking", "forward", "expecting", "ready"},
}

# Emotions carrying negative valence, used to derive a coarse polarity signal.
NEGATIVE_EMOTIONS = frozenset({"fear", "anger", "sadness", "disgust"})


def detect_emotions(text: str) -> Dict[str, int]:
    """Detect Plutchik emotions present in text using a keyword lexicon.

    Args:
        text: Input text to analyse.

    Returns:
        A mapping from emotion to occurrence count.
    """
    words = {word.strip(".,!?;:\"'").lower() for word in str(text).split()}
    counts = {
        emotion: len(words & lexicon) for emotion, lexicon in EMOTION_LEXICON.items()
    }
    return counts


def dominant_emotion(counts: Dict[str, int]) -> str:
    """Return the most frequent emotion, or ``"neutral"`` if none are present.

    Ties are broken by the fixed lexicon ordering (Plutchik order) so the result
    is deterministic.

    Args:
        counts: Emotion-count mapping as produced by :func:`detect_emotions`.

    Returns:
        The dominant emotion name, or ``"neutral"`` when all counts are zero.
    """
    best_emotion = "neutral"
    best_count = 0
    for emotion in EMOTION_LEXICON:
        count = counts.get(emotion, 0)
        if count > best_count:
            best_emotion = emotion
            best_count = count
    return best_emotion


def add_emotion_features(
    df: pd.DataFrame, text_column: str, prefix: str = "emotion_"
) -> pd.DataFrame:
    """Expand a DataFrame with emotion features.

    For each row this adds one count column per Plutchik emotion plus three
    aggregate columns:

    * ``{prefix}intensity``: total emotion-word hits.
    * ``{prefix}polarity``: negative-emotion hits minus positive-emotion hits.
    * ``{prefix}dominant``: the dominant emotion label (categorical).

    Args:
        df: DataFrame containing a text column.
        text_column: Name of the column with text data.
        prefix: Prefix applied to each generated feature column.

    Returns:
        DataFrame with the added emotion feature columns.
    """
    expanded = df.copy()
    counts = expanded[text_column].astype(str).map(detect_emotions)
    for emotion in EMOTION_LEXICON:
        expanded[f"{prefix}{emotion}"] = counts.map(
            lambda c, e=emotion: c.get(e, 0)
        ).astype(int)

    expanded[f"{prefix}intensity"] = counts.map(lambda c: sum(c.values())).astype(int)
    expanded[f"{prefix}polarity"] = counts.map(
        lambda c: sum(c[e] for e in NEGATIVE_EMOTIONS)
        - sum(c[e] for e in EMOTION_LEXICON if e not in NEGATIVE_EMOTIONS)
    ).astype(int)
    expanded[f"{prefix}dominant"] = counts.map(dominant_emotion)
    return expanded
