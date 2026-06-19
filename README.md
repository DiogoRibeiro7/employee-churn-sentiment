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
├── notebooks/
│   ├── build_walkthrough.py
│   └── exploratory/
│       └── 01_churn_modeling_walkthrough.ipynb
├── tests/
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

`employee_churn.nlp`

- `sentiment.py`: VADER-based sentiment scoring
- `emotion.py`: Plutchik eight-emotion lexicon features with intensity, polarity, and dominant emotion

`employee_churn.models`

- `train.py`: baseline/NLP/combined training, the multi-model zoo, cross-validation, and hyperparameter tuning
- `calibrate.py`: probability calibration and reliability-curve helpers
- `fairness.py`: group fairness report and four-fifths-rule summary
- `predict.py`: weekly risk scoring and CSV export
- `dashboard.py`: ranking and threshold-based high-risk alerts
- `monitor.py`: feature drift, calibration metrics, and monitoring summary helpers
- `track.py`: MLflow experiment logging
- `explain.py`: SHAP explanation generation (linear and tree models)

## Example Notebook

An end-to-end walkthrough — synthetic data, feature engineering, the model zoo
with cross-validation, tuning, calibration, fairness diagnostics, and SHAP — is
in [notebooks/exploratory/01_churn_modeling_walkthrough.ipynb](notebooks/exploratory/01_churn_modeling_walkthrough.ipynb).
It runs anywhere with no real data. Regenerate it deterministically with:

```bash
python notebooks/build_walkthrough.py
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
