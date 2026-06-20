"""Tests for the feature store registry and caching."""

from pathlib import Path

import pandas as pd
import pytest

from employee_churn.features.feature_store import FeatureStore, default_feature_store


def _add_double(df: pd.DataFrame, column: str = "x") -> pd.DataFrame:
    out = df.copy()
    out["double"] = out[column] * 2
    return out


def test_register_and_compute() -> None:
    store = FeatureStore()
    store.register("double", _add_double)
    out = store.compute("double", pd.DataFrame({"x": [1, 2]}))
    assert list(out["double"]) == [2, 4]


def test_register_as_decorator() -> None:
    store = FeatureStore()

    @store.register("inc")
    def _inc(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["y"] = out["x"] + 1
        return out

    assert "inc" in store
    assert store.names() == ["inc"]
    assert list(store.compute("inc", pd.DataFrame({"x": [1]}))["y"]) == [2]


def test_duplicate_registration_raises() -> None:
    store = FeatureStore()
    store.register("double", _add_double)
    with pytest.raises(ValueError):
        store.register("double", _add_double)


def test_unknown_feature_raises() -> None:
    with pytest.raises(KeyError):
        FeatureStore().compute("missing", pd.DataFrame({"x": [1]}))


def test_build_chains_transforms() -> None:
    store = FeatureStore()
    store.register("double", _add_double)
    store.register(
        "plus_double",
        lambda df: df.assign(triple=df["x"] + df["double"]),
    )
    out = store.build(pd.DataFrame({"x": [3]}), ["double", "plus_double"])
    assert out.loc[0, "double"] == 6
    assert out.loc[0, "triple"] == 9


def test_caching_reuses_result(tmp_path: Path) -> None:
    calls = {"n": 0}

    def _counting(df: pd.DataFrame) -> pd.DataFrame:
        calls["n"] += 1
        return df.assign(z=df["x"] * 10)

    store = FeatureStore(cache_dir=tmp_path)
    store.register("counting", _counting)
    df = pd.DataFrame({"x": [1, 2]})

    first = store.compute("counting", df)
    second = store.compute("counting", df)  # served from cache
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(first, second)
    assert any(tmp_path.glob("*.pkl"))


def test_cache_key_changes_with_data(tmp_path: Path) -> None:
    calls = {"n": 0}

    def _counting(df: pd.DataFrame) -> pd.DataFrame:
        calls["n"] += 1
        return df.assign(z=df["x"])

    store = FeatureStore(cache_dir=tmp_path)
    store.register("counting", _counting)
    store.compute("counting", pd.DataFrame({"x": [1]}))
    store.compute("counting", pd.DataFrame({"x": [2]}))  # different data -> recompute
    assert calls["n"] == 2


def test_default_feature_store_end_to_end() -> None:
    store = default_feature_store()
    assert {"sentiment", "emotion", "text_stats", "tenure_bands"} <= set(store.names())
    df = pd.DataFrame({"feedback": ["I am happy and proud", "angry and sad"]})
    out = store.build(
        df,
        ["sentiment", "emotion", "text_stats"],
        params={
            "sentiment": {"text_column": "feedback"},
            "emotion": {"text_column": "feedback"},
            "text_stats": {"text_column": "feedback"},
        },
    )
    assert "sentiment" in out.columns
    assert "emotion_joy" in out.columns
    assert "text_word_count" in out.columns
