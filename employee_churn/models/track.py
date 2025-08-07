"""Experiment tracking utilities using MLflow."""

from __future__ import annotations

from typing import Any, Mapping


def log_experiment(
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, float],
    tracking_uri: str | None = None,
) -> None:
    """Log experiment parameters and metrics to MLflow if available.

    Args:
        run_name: Name of the MLflow run.
        params: Parameters to record.
        metrics: Metrics to record.
        tracking_uri: Optional tracking server URI. If provided, ``mlflow`` will
            send data to this URI; otherwise the default local setup is used.
    """
    try:
        import mlflow
    except Exception:  # pragma: no cover - MLflow optional
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(dict(params))
        mlflow.log_metrics(dict(metrics))
