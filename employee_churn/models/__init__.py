"""Model training and evaluation utilities."""

from .train import (
    build_model_zoo,
    cross_validate_models,
    evaluate_models,
    train_baseline_models,
    train_combined_model,
    train_nlp_model,
    tune_hyperparameters,
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
from .calibrate import (
    calibrate_model,
    calibration_improvement,
    reliability_curve,
)
from .fairness import fairness_summary, group_fairness_report

__all__ = [
    "train_baseline_models",
    "train_combined_model",
    "train_nlp_model",
    "build_model_zoo",
    "cross_validate_models",
    "tune_hyperparameters",
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
    "calibrate_model",
    "calibration_improvement",
    "reliability_curve",
    "fairness_summary",
    "group_fairness_report",
]
