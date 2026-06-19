"""NLP utilities for sentiment and emotion analysis."""

from .emotion import add_emotion_features, detect_emotions, dominant_emotion
from .sentiment import add_sentiment_scores, analyze_sentiment

__all__ = [
    "sentiment",
    "emotion",
    "add_sentiment_scores",
    "analyze_sentiment",
    "add_emotion_features",
    "detect_emotions",
    "dominant_emotion",
]
