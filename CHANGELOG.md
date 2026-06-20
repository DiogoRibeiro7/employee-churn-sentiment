# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `employee_churn.config`: environment-aware configuration with layered YAML
  defaults, `.env` support, and `ECS_<SECTION>__<KEY>` overrides; `configs/`
  files and `.env.example`.
- `employee_churn.data.synthetic`: reproducible synthetic employee dataset
  generator with a known churn signal.
- Structured features: tenure bands, promotion velocity, peer-relative
  compensation (`features.engineer_structured`).
- Text-statistics features (`features.engineer_text`).
- Expanded emotion lexicon to Plutchik's eight emotions with intensity,
  polarity, and dominant-emotion signals (`nlp.emotion`).
- Model zoo plus stratified cross-validation and randomized hyperparameter
  tuning (`models.train`).
- Probability calibration and reliability curves (`models.calibrate`).
- Group fairness diagnostics (`models.fairness`).
- Comprehensive evaluation report combining discrimination, calibration, and
  fairness (`models.evaluate`).
- Filesystem model registry with versioning (`models.registry`).
- Text preprocessing pipeline: cleaning, tokenization, stopword removal,
  normalization (`nlp.preprocessing`).
- Central feature-transform registry with on-disk caching
  (`features.feature_store`).
- End-to-end walkthrough notebook and deterministic builder.
- Three deep-dive, fully executed analysis notebooks with written
  interpretation: exploratory data analysis (02), modeling and evaluation
  (03), and fairness and explainability (04), plus their builder
  (`notebooks/build_notebooks.py`).
- Model card (`docs/model_card.md`) and contributor guide (`CONTRIBUTING.md`).

### Changed
- `models.explain.explain_with_shap` now handles tree models' per-class
  (3-D) SHAP output.
- NLP/combined training paths are numeric-safe against categorical features.
- CI runs the full test suite (previously a subset).

## [0.1.0]

### Added
- Initial package: data loading/cleaning/validation, structured feature
  engineering, VADER sentiment and lexicon emotion features, baseline/NLP/
  combined models, MLflow tracking, SHAP explanations, weekly risk scoring,
  drift and calibration monitoring, a Streamlit dashboard, and CLI scripts.
