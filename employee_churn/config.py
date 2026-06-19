"""Environment-aware configuration management.

Loads layered settings from ``configs/*.yaml`` and overlays environment
variables (and an optional ``.env`` file), then validates the result with
pydantic. This removes hardcoded paths and magic numbers from the codebase:

    >>> from employee_churn.config import load_config
    >>> cfg = load_config()
    >>> cfg.model.random_state
    0

Environment overrides use the ``ECS_<SECTION>__<KEY>`` convention, e.g.
``ECS_MODEL__RANDOM_STATE=42``. Values are parsed as YAML so numbers, booleans,
and lists are coerced naturally.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

ENV_PREFIX = "ECS_"
_NESTED_DELIMITER = "__"

# YAML files merged, in order, into a single configuration mapping.
_CONFIG_FILES = ("model_config.yaml", "data_config.yaml", "logging_config.yaml")


class PathsConfig(BaseModel):
    """Filesystem layout for data and artifacts."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("models/artifacts")
    outputs_dir: Path = Path("outputs")


class ModelConfig(BaseModel):
    """Model training and evaluation hyper-parameters."""

    model_name: str = "log_reg"
    test_size: float = 0.2
    random_state: int = 0
    top_k: int = 10
    cv_folds: int = 5
    tuning_iterations: int = 10
    calibration_method: str = "isotonic"


class DataConfig(BaseModel):
    """Dataset schema (column names) and validation requirements."""

    id_column: str = "employee_id"
    date_column: str = "snapshot_date"
    target_column: str = "churned"
    text_column: Optional[str] = "feedback"
    required_columns: List[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    """Logging format and level."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%dT%H:%M:%S"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    environment: str = "development"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_dotenv(path: str | Path = PROJECT_ROOT / ".env") -> Dict[str, str]:
    """Load a simple ``KEY=VALUE`` ``.env`` file into ``os.environ``.

    Existing environment variables are not overwritten, so real environment
    settings always win over the file. Lines that are blank or start with ``#``
    are ignored. Returns the mapping that was applied.

    Args:
        path: Path to the ``.env`` file. Missing files are a no-op.

    Returns:
        The key/value pairs that were newly set.
    """
    env_path = Path(path)
    applied: Dict[str, str] = {}
    if not env_path.is_file():
        return applied
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides(environ: Dict[str, str], prefix: str = ENV_PREFIX) -> Dict[str, Any]:
    """Build a nested override mapping from ``ECS_SECTION__KEY`` variables."""
    overrides: Dict[str, Any] = {}
    for raw_key, raw_value in environ.items():
        if not raw_key.startswith(prefix):
            continue
        remainder = raw_key[len(prefix) :].lower()
        parts = remainder.split(_NESTED_DELIMITER)
        # YAML parsing coerces "42" -> int, "true" -> bool, "[a, b]" -> list.
        value = yaml.safe_load(raw_value)
        cursor = overrides
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return overrides


def load_config(
    config_dir: str | Path = CONFIG_DIR,
    use_dotenv: bool = True,
    environ: Optional[Dict[str, str]] = None,
) -> AppConfig:
    """Load and validate the layered application configuration.

    Resolution order (later layers win):

    1. Built-in pydantic defaults.
    2. ``configs/*.yaml`` files.
    3. ``.env`` file (only if ``use_dotenv`` and the variable is unset).
    4. Process environment variables (``ECS_`` prefix).

    Args:
        config_dir: Directory containing the YAML config files.
        use_dotenv: Whether to load a ``.env`` file before reading env vars.
        environ: Environment mapping to read overrides from. Defaults to
            ``os.environ``; injectable for testing.

    Returns:
        A validated :class:`AppConfig`.
    """
    if use_dotenv:
        load_dotenv()
    env = dict(os.environ if environ is None else environ)

    merged: Dict[str, Any] = {}
    config_path = Path(config_dir)
    for filename in _CONFIG_FILES:
        file_path = config_path / filename
        if file_path.is_file():
            loaded = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
            merged = _deep_merge(merged, loaded)

    merged = _deep_merge(merged, _env_overrides(env))
    return AppConfig(**merged)


def configure_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """Configure and return the package root logger from config.

    Args:
        config: Logging configuration. Falls back to :func:`load_config` when
            not supplied.

    Returns:
        The configured ``employee_churn`` logger.
    """
    cfg = config or load_config().logging
    logging.basicConfig(
        level=getattr(logging, cfg.level.upper(), logging.INFO),
        format=cfg.format,
        datefmt=cfg.datefmt,
    )
    logger = logging.getLogger("employee_churn")
    logger.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    return logger
