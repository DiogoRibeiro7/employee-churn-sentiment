"""Tests for the environment-aware configuration layer."""

from pathlib import Path

from employee_churn.config import (
    AppConfig,
    load_config,
    load_dotenv,
)


def test_defaults_from_yaml() -> None:
    cfg = load_config(use_dotenv=False, environ={})
    assert isinstance(cfg, AppConfig)
    assert cfg.environment == "development"
    assert cfg.model.model_name == "log_reg"
    assert cfg.data.target_column == "churned"
    assert "employee_id" in cfg.data.required_columns


def test_env_override_scalar_and_nested() -> None:
    environ = {
        "ECS_ENVIRONMENT": "production",
        "ECS_MODEL__RANDOM_STATE": "123",
        "ECS_MODEL__CALIBRATION_METHOD": "sigmoid",
        "ECS_DATA__TEXT_COLUMN": "notes",
    }
    cfg = load_config(use_dotenv=False, environ=environ)
    assert cfg.environment == "production"
    assert cfg.model.random_state == 123  # coerced to int
    assert cfg.model.calibration_method == "sigmoid"
    assert cfg.data.text_column == "notes"


def test_env_override_ignores_unprefixed() -> None:
    cfg = load_config(use_dotenv=False, environ={"RANDOM_STATE": "999"})
    assert cfg.model.random_state == 0


def test_load_dotenv(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nECS_MODEL__TOP_K=7\nEMPTY_LINE_BELOW=\n\n", encoding="utf-8"
    )
    monkeypatch.delenv("ECS_MODEL__TOP_K", raising=False)
    try:
        applied = load_dotenv(env_file)
        assert applied["ECS_MODEL__TOP_K"] == "7"

        cfg = load_config(use_dotenv=False)  # value now present in os.environ
        assert cfg.model.top_k == 7
    finally:
        # load_dotenv writes straight to os.environ; clean up so other tests
        # are not affected.
        __import__("os").environ.pop("ECS_MODEL__TOP_K", None)


def test_dotenv_does_not_override_real_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ECS_ENVIRONMENT", "staging")
    env_file = tmp_path / ".env"
    env_file.write_text("ECS_ENVIRONMENT=production\n", encoding="utf-8")
    load_dotenv(env_file)
    assert __import__("os").environ["ECS_ENVIRONMENT"] == "staging"
