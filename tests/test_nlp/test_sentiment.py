"""Tests for NLP sentiment and emotion utilities."""

import pandas as pd

from employee_churn.nlp.emotion import add_emotion_features, detect_emotions
from employee_churn.nlp.sentiment import add_sentiment_scores, analyze_sentiment


def test_analyze_sentiment_distinguishes_polarity():
    positive = analyze_sentiment("I love my job")
    negative = analyze_sentiment("I hate my job")
    assert positive > negative


def test_add_sentiment_scores_adds_column():
    df = pd.DataFrame({"text": ["I love my job"]})
    scored = add_sentiment_scores(df, "text")
    assert "sentiment" in scored.columns


def test_detect_emotions_counts_keywords():
    text = "I am happy and delighted but also afraid"
    emotions = detect_emotions(text)
    assert emotions["joy"] == 2
    assert emotions["fear"] == 1


def test_add_emotion_features_expands_dataframe():
    df = pd.DataFrame({"text": ["angry but hopeful"]})
    expanded = add_emotion_features(df, "text")
    assert "emotion_anger" in expanded.columns
