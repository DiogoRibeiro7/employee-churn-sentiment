# Employee Churn with Sentiment Analysis

This repository packages a small end-to-end churn modeling workflow that combines structured HR attributes with sentiment and emotion features extracted from employee feedback.

It currently covers:
- data loading, cleaning, anonymization, and column validation
- structured feature engineering for tenure, promotions, and team size
- sentiment scoring with VADER and lightweight lexicon-based emotion features
- baseline, NLP-only, and combined churn models
- experiment logging with MLflow
- SHAP-based explainability
- weekly risk scoring, CSV export, risk ranking, alert thresholds, drift checks, and calibration summaries
- a minimal Streamlit dashboard and CLI scripts for training, scoring, and evaluation

## Project Layout

```text
employee-churn-sentiment/
├── employee_churn/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── nlp/
├── scripts/
│   ├── dashboard_app.py
│   ├── evaluate_model.py
│   ├── predict_risk.py
│   └── train_model.py
├── tests/
├── README.md
├── ROADMAP.md
└── pyproject.toml
```

## Installation

```bash
poetry install
```

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

`employee_churn.features`

- `engineer_structured.py`: tenure, days since promotion, and team size features

`employee_churn.nlp`

- `sentiment.py`: VADER-based sentiment scoring
- `emotion.py`: simple lexicon-based emotion features

`employee_churn.models`

- `train.py`: baseline, NLP-only, and combined training helpers plus evaluation metrics
- `predict.py`: weekly risk scoring and CSV export
- `dashboard.py`: ranking and threshold-based high-risk alerts
- `monitor.py`: feature drift, calibration metrics, and monitoring summary helpers
- `track.py`: MLflow experiment logging
- `explain.py`: SHAP explanation generation

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
