"""Tests for the filesystem model registry."""

from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

from employee_churn.models.registry import ModelRegistry


def _bundle(seed: int = 0) -> dict:
    model = LogisticRegression()
    return {"model": model, "model_name": "log_reg", "metrics": {"roc_auc": 0.8}}


def test_register_and_load_roundtrip(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    version = registry.register(_bundle(), "churn", metrics={"roc_auc": 0.81})
    assert version == "v1"

    loaded = registry.load("churn", "v1")
    assert loaded["model_name"] == "log_reg"
    assert isinstance(loaded["model"], LogisticRegression)


def test_versions_increment_and_latest(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    registry.register(_bundle(), "churn")
    registry.register(_bundle(), "churn")
    assert registry.latest_version("churn") == "v2"
    assert len(registry.list_versions("churn")) == 2
    assert registry.list_models() == ["churn"]


def test_load_latest_default(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    registry.register({"model_name": "a"}, "churn")
    registry.register({"model_name": "b"}, "churn")
    assert registry.load("churn")["model_name"] == "b"


def test_metadata_records_metrics(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    registry.register(_bundle(), "churn", metrics={"f1": 0.7}, tags={"owner": "hr"})
    meta = registry.get_metadata("churn", "v1")
    assert meta["metrics"]["f1"] == 0.7
    assert meta["tags"]["owner"] == "hr"


def test_unknown_model_and_version_raise(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    with pytest.raises(KeyError):
        registry.list_versions("missing")
    registry.register(_bundle(), "churn")
    with pytest.raises(KeyError):
        registry.load("churn", "v9")


def test_persists_across_instances(tmp_path: Path) -> None:
    ModelRegistry(tmp_path).register(_bundle(), "churn")
    # A fresh instance reads the same on-disk index.
    assert ModelRegistry(tmp_path).latest_version("churn") == "v1"
