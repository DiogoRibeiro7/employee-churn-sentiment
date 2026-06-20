"""Feature engineering subpackage."""

from .engineer_structured import (
    add_career_progression_features,
    add_compensation_features,
    add_promotion_velocity,
    add_team_metrics,
    add_tenure_bands,
)
from .engineer_text import add_text_statistics, text_statistics
from .feature_store import FeatureStore, default_feature_store

__all__ = [
    "add_career_progression_features",
    "add_team_metrics",
    "add_tenure_bands",
    "add_promotion_velocity",
    "add_compensation_features",
    "add_text_statistics",
    "text_statistics",
    "FeatureStore",
    "default_feature_store",
]
