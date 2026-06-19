"""Model training and evaluation utilities."""

from .train import (
    evaluate_models,
    train_baseline_models,
    train_combined_model,
    train_nlp_model,
)
from .track import log_experiment
from .explain import explain_with_shap

from .predict import export_scores_csv, score_employees_weekly
from .dashboard import build_high_risk_alerts, build_risk_dashboard
from .monitor import (
    detect_feature_drift,
    evaluate_model_calibration,
    summarize_monitoring_status,
)

__all__ = [
    "train_baseline_models",
    "train_combined_model",
    "train_nlp_model",
    "evaluate_models",
    "log_experiment",
    "explain_with_shap",
    "score_employees_weekly",
    "export_scores_csv",
    "build_risk_dashboard",
    "build_high_risk_alerts",
    "detect_feature_drift",
    "evaluate_model_calibration",
    "summarize_monitoring_status",
]
