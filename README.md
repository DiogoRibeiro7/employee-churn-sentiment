# Employee Churn with Sentiment Analysis

This repository packages a small end-to-end churn modeling workflow that combines structured HR attributes with sentiment and emotion features extracted from employee feedback.

It currently covers:
- data loading, cleaning, anonymization, column validation, and a reproducible synthetic dataset generator
- structured feature engineering for tenure (raw + ordinal bands), promotions, promotion velocity, peer-relative compensation, and team size
- sentiment scoring with VADER plus lexicon-based emotion features over Plutchik's eight emotions (with intensity, polarity, and dominant-emotion signals)
- shape-based text-statistics features (length, punctuation, negation, lexical diversity)
- baseline, NLP-only, and combined churn models across a multi-model zoo (logistic regression, random forest, gradient boosting, hist gradient boosting, XGBoost)
- stratified cross-validation and randomized hyperparameter tuning
- probability calibration (isotonic / Platt) with before/after reliability metrics
- group fairness diagnostics (selection-rate parity, equal opportunity, four-fifths rule)
- experiment logging with MLflow
- SHAP-based explainability (including tree multi-output models)
- weekly risk scoring, CSV export, risk ranking, alert thresholds, drift checks, and calibration summaries
- a minimal Streamlit dashboard, CLI scripts for training, scoring, and evaluation, and a runnable end-to-end notebook

## Project Layout

```text
employee-churn-sentiment/
├── employee_churn/
│   ├── data/        # load, clean, validate, synthetic
│   ├── features/    # engineer_structured, engineer_text
│   ├── models/      # train, calibrate, fairness, predict, monitor, explain, track
│   └── nlp/         # sentiment, emotion
├── scripts/
│   ├── dashboard_app.py
│   ├── evaluate_model.py
│   ├── predict_risk.py
│   └── train_model.py
├── configs/         # model/data/logging YAML
├── docs/
│   └── model_card.md
├── notebooks/
│   ├── build_walkthrough.py
│   ├── build_notebooks.py
│   └── exploratory/        # 01 walkthrough + 02 EDA, 03 modeling, 04 fairness
├── tests/
├── CHANGELOG.md
├── CONTRIBUTING.md
├── README.md
├── ROADMAP.md
└── pyproject.toml
```

## Installation

```bash
poetry install
```

## Configuration

Settings live in [configs/](configs/) (`model_config.yaml`, `data_config.yaml`,
`logging_config.yaml`) and are loaded and validated through
[employee_churn/config.py](employee_churn/config.py):

```python
from employee_churn import load_config

cfg = load_config()
cfg.model.random_state   # 0
cfg.data.target_column   # "churned"
```

Any value can be overridden by an environment variable using the
`ECS_<SECTION>__<KEY>` convention (values are parsed as YAML):

```bash
export ECS_MODEL__RANDOM_STATE=123
export ECS_MODEL__CALIBRATION_METHOD=sigmoid
```

Copy [.env.example](.env.example) to `.env` for a local, file-based override
layer (real environment variables always take precedence).

## Main Workflows

Train and save a baseline artifact:

```bash
python -m scripts.train_model data/train.csv target artifacts/model.pkl
```

Train a combined structured + text model:

```bash
python -m scripts.train_model data/train.csv target artifacts/model.pkl --text-column feedback
```

Score weekly churn risk from a saved artifact:

```bash
python -m scripts.predict_risk artifacts/model.pkl data/score.csv employee_id snapshot_date outputs/scores.csv
```

Evaluate a saved artifact on labeled data:

```bash
python -m scripts.evaluate_model artifacts/model.pkl data/eval.csv --output-json outputs/metrics.json
```

Run the dashboard:

```bash
streamlit run scripts/dashboard_app.py
```

## Package Highlights

`employee_churn.data`

- `load.py`: CSV ingestion helpers for HR and feedback data
- `clean.py`: column normalization and text anonymization
- `validate.py`: required-column validation
- `synthetic.py`: reproducible synthetic employee dataset with a known churn signal

`employee_churn.features`

- `engineer_structured.py`: tenure, tenure bands, days since promotion, promotion velocity, peer-relative compensation, and team size features
- `engineer_text.py`: shape-based text statistics (length, punctuation, negation, lexical diversity)
- `feature_store.py`: central feature-transform registry with optional on-disk caching

`employee_churn.nlp`

- `sentiment.py`: VADER-based sentiment scoring
- `emotion.py`: Plutchik eight-emotion lexicon features with intensity, polarity, and dominant emotion
- `preprocessing.py`: text cleaning, tokenization, stopword removal, and normalization

`employee_churn.models`

- `train.py`: baseline/NLP/combined training, the multi-model zoo, cross-validation, and hyperparameter tuning
- `evaluate.py`: comprehensive evaluation report (discrimination + calibration + fairness)
- `calibrate.py`: probability calibration and reliability-curve helpers
- `fairness.py`: group fairness report and four-fifths-rule summary
- `registry.py`: filesystem model registry with versioning and a metadata index
- `predict.py`: weekly risk scoring and CSV export
- `dashboard.py`: ranking and threshold-based high-risk alerts
- `monitor.py`: feature drift, calibration metrics, and monitoring summary helpers
- `track.py`: MLflow experiment logging
- `explain.py`: SHAP explanation generation (linear and tree models)

## Notebooks

The [notebooks/exploratory/](notebooks/exploratory/) directory holds runnable,
**fully executed** analyses (outputs committed) on a reproducible synthetic
dataset — no real data required. Each contains charts plus written analysis of
the results:

- [01_churn_modeling_walkthrough.ipynb](notebooks/exploratory/01_churn_modeling_walkthrough.ipynb)
  — end-to-end tour (data → features → model zoo → tuning → calibration →
  fairness → SHAP).
- [02_exploratory_data_analysis.ipynb](notebooks/exploratory/02_exploratory_data_analysis.ipynb)
  — what drives churn; shows structured features are weak (best |r| ≈ 0.17)
  while text sentiment/emotion cleanly separate the classes.
- [03_modeling_and_evaluation.ipynb](notebooks/exploratory/03_modeling_and_evaluation.ipynb)
  — model comparison (CV AUC ≈ 0.70), ROC/PR, precision@k ≈ 0.70, an honest
  calibration result (isotonic does not help here), and feature importances
  dominated by text features.
- [04_fairness_and_explainability.ipynb](notebooks/exploratory/04_fairness_and_explainability.ipynb)
  — a group-fairness audit (the tuned model **fails the four-fifths rule**,
  disparate impact ≈ 0.64), a group-aware threshold mitigation (→ ≈ 0.97), and
  SHAP global + per-employee explanations.
- [05_business_impact_and_roi.ipynb](notebooks/exploratory/05_business_impact_and_roi.ipynb)
  — turns scores into dollars: a cost model, break-even probability, net-savings
  curve, and a capacity-constrained comparison where model targeting beats
  random by **6.6×** and even beats blanket outreach, plus a sensitivity grid.

Regenerate the notebooks deterministically, then execute them, with:

```bash
python notebooks/build_walkthrough.py      # notebook 01
python notebooks/build_notebooks.py        # notebooks 02–04
jupyter nbconvert --to notebook --execute --inplace notebooks/exploratory/*.ipynb
```

## Testing

Run the test suite with:

```bash
pytest
```

The roadmap is currently complete through deployment and monitoring in [ROADMAP.md](ROADMAP.md).

## Notes

- The anonymization and emotion extraction logic is intentionally lightweight and should be treated as baseline functionality.
- The CLI flow stores a pickled model artifact with metadata about feature preparation so scoring and evaluation stay consistent with training.

## License

MIT. See [LICENSE](LICENSE).
