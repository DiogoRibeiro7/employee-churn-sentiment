"""Centralized feature management with optional on-disk caching.

A small registry that maps feature names to transform functions, so feature
engineering is declared in one place and applied consistently across training,
scoring, and notebooks. Transforms can be cached to disk keyed by the input
data's content hash, avoiding recomputation of expensive features (e.g.
sentiment scoring) on unchanged inputs.

Each registered transform has the uniform signature ``func(df, **kwargs) ->
DataFrame`` and is expected to return ``df`` with new feature columns added.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

Transform = Callable[..., pd.DataFrame]


class FeatureStore:
    """Register, apply, and cache feature transforms."""

    def __init__(self, cache_dir: Optional[str | Path] = None) -> None:
        """Create a feature store.

        Args:
            cache_dir: Optional directory for the on-disk cache. When ``None``,
                caching is disabled and every ``compute`` recomputes.
        """
        self._registry: Dict[str, Transform] = {}
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def register(
        self, name: str, func: Optional[Transform] = None
    ) -> Transform | Callable[[Transform], Transform]:
        """Register a transform, usable directly or as a decorator.

        Direct: ``store.register("sentiment", add_sentiment_scores)``.
        Decorator::

            @store.register("my_feature")
            def my_feature(df, **kwargs): ...

        Args:
            name: Unique feature name.
            func: The transform. Omit to use as a decorator.

        Returns:
            The registered function (or the decorator when ``func`` is omitted).

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self._registry:
            raise ValueError(f"feature '{name}' is already registered")

        def _do_register(fn: Transform) -> Transform:
            self._registry[name] = fn
            return fn

        if func is not None:
            return _do_register(func)
        return _do_register

    def names(self) -> List[str]:
        """Return the registered feature names in sorted order."""
        return sorted(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def _cache_key(self, name: str, df: pd.DataFrame, kwargs: dict) -> str:
        """Build a stable cache key from the feature name, data, and kwargs."""
        row_hash = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(kwargs, sort_keys=True, default=str).encode("utf-8"))
        digest.update(row_hash)
        return digest.hexdigest()

    def compute(
        self,
        name: str,
        df: pd.DataFrame,
        use_cache: bool = True,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Apply a single registered transform, with optional caching.

        Args:
            name: Registered feature name.
            df: Input DataFrame.
            use_cache: Whether to read/write the on-disk cache (only effective
                when the store has a ``cache_dir``).
            **kwargs: Forwarded to the transform.

        Returns:
            The transformed DataFrame.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"unknown feature '{name}'")

        caching = use_cache and self.cache_dir is not None
        cache_path: Optional[Path] = None
        if caching:
            cache_path = self.cache_dir / f"{self._cache_key(name, df, kwargs)}.pkl"
            if cache_path.is_file():
                with cache_path.open("rb") as handle:
                    return pickle.load(handle)

        result = self._registry[name](df, **kwargs)

        if caching and cache_path is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                pickle.dump(result, handle)
        return result

    def build(
        self,
        df: pd.DataFrame,
        names: Sequence[str],
        use_cache: bool = True,
        params: Optional[Dict[str, dict]] = None,
    ) -> pd.DataFrame:
        """Apply several transforms in sequence, threading the output forward.

        Args:
            df: Input DataFrame.
            names: Ordered feature names to apply.
            use_cache: Whether to use the on-disk cache for each step.
            params: Optional per-feature keyword arguments, keyed by name.

        Returns:
            The DataFrame after all transforms have been applied.
        """
        params = params or {}
        result = df
        for name in names:
            result = self.compute(
                name, result, use_cache=use_cache, **params.get(name, {})
            )
        return result


def default_feature_store(cache_dir: Optional[str | Path] = None) -> FeatureStore:
    """Build a feature store pre-registered with the package's transforms.

    Registered names: ``career``, ``team``, ``tenure_bands``,
    ``promotion_velocity``, ``compensation``, ``text_stats``, ``sentiment``,
    ``emotion``.

    Args:
        cache_dir: Optional cache directory passed to :class:`FeatureStore`.

    Returns:
        A ready-to-use :class:`FeatureStore`.
    """
    from employee_churn.features.engineer_structured import (
        add_career_progression_features,
        add_compensation_features,
        add_promotion_velocity,
        add_team_metrics,
        add_tenure_bands,
    )
    from employee_churn.features.engineer_text import add_text_statistics
    from employee_churn.nlp.emotion import add_emotion_features
    from employee_churn.nlp.sentiment import add_sentiment_scores

    store = FeatureStore(cache_dir=cache_dir)
    store.register("career", add_career_progression_features)
    store.register("team", add_team_metrics)
    store.register("tenure_bands", add_tenure_bands)
    store.register("promotion_velocity", add_promotion_velocity)
    store.register("compensation", add_compensation_features)
    store.register("text_stats", add_text_statistics)
    store.register("sentiment", add_sentiment_scores)
    store.register("emotion", add_emotion_features)
    return store
