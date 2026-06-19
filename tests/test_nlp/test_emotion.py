"""Tests for the Plutchik emotion features."""

import pandas as pd

from employee_churn.nlp.emotion import (
    EMOTION_LEXICON,
    add_emotion_features,
    detect_emotions,
    dominant_emotion,
)


def test_detect_emotions_counts() -> None:
    counts = detect_emotions("I am happy and proud but also a little afraid.")
    assert counts["joy"] >= 2
    assert counts["fear"] == 1
    assert counts["anger"] == 0


def test_dominant_emotion_neutral() -> None:
    assert dominant_emotion({e: 0 for e in EMOTION_LEXICON}) == "neutral"


def test_dominant_emotion_picks_max() -> None:
    counts = detect_emotions("angry angry mad and furious")
    assert dominant_emotion(counts) == "anger"


def test_add_emotion_features_columns() -> None:
    df = pd.DataFrame({"feedback": ["happy and delighted", "angry and scared"]})
    out = add_emotion_features(df, "feedback")
    for emotion in EMOTION_LEXICON:
        assert f"emotion_{emotion}" in out.columns
    assert "emotion_intensity" in out.columns
    assert "emotion_polarity" in out.columns
    assert "emotion_dominant" in out.columns
    # Row 0 is positive -> negative polarity value, row 1 negative -> positive.
    assert out.loc[0, "emotion_polarity"] < out.loc[1, "emotion_polarity"]
    assert out.loc[1, "emotion_dominant"] in {"anger", "fear"}
