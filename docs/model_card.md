# Model Card: Employee Churn with Sentiment

This card follows the spirit of [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993).
It documents the intended use, data, metrics, and limitations of the churn
models produced by this package.

## Model details

- **Owner**: People Analytics / HR Data Science.
- **Type**: Binary classifier (churn vs. retain). Configurable estimator —
  logistic regression (default), random forest, gradient boosting, histogram
  gradient boosting, or XGBoost — see `employee_churn.models.build_model_zoo`.
- **Inputs**: Structured HR attributes (tenure, promotions, compensation, team
  size, performance/satisfaction) plus features derived from free-text feedback
  (VADER sentiment, Plutchik emotion counts, text statistics).
- **Output**: A calibrated churn-risk probability in `[0, 1]`, optionally
  thresholded into a high-risk alert.
- **Training**: `scripts/train_model.py` → a pickled bundle with the fitted
  model and the feature schema; versionable via `employee_churn.models.ModelRegistry`.

## Intended use

- **In scope**: Aggregate, decision-support insight to help HR prioritize
  retention conversations and identify systemic drivers of attrition.
- **Out of scope**: Automated or punitive decisions about individuals
  (termination, compensation, promotion). Predictions must not be the sole basis
  for any employment action.

## Metrics

Evaluate with `employee_churn.models.evaluate_model`, which reports three lenses:

- **Discrimination**: ROC-AUC, F1, precision@k.
- **Calibration**: Brier score, expected/maximum calibration error.
- **Fairness**: per-group selection rate, true/false-positive rates, disparate
  impact ratio (four-fifths rule), and equal-opportunity difference.

Target baseline (see `ROADMAP.md`): ROC-AUC > 0.80, with calibrated
probabilities and no group failing the four-fifths rule.

## Training data

The shipped examples and tests use a **synthetic** dataset
(`employee_churn.data.make_synthetic_employee_data`) with a known churn signal —
no real employee records are included in the repository. Production deployments
must supply their own governed dataset.

## Ethical considerations & limitations

- **Sensitive context**: churn models operate on employee data and can encode or
  amplify bias. Always run the fairness report before deployment and monitor it
  over time.
- **Text anonymization** and **emotion extraction** are intentionally
  lightweight (regex / lexicon) and are not a substitute for robust PII handling
  or a trained emotion classifier.
- **Drift**: feature distributions and the churn relationship change over time.
  Use `employee_churn.models.monitor` to track drift and recalibrate/retrain.
- **Consent & transparency**: feedback-derived features should only be used with
  appropriate employee consent and disclosure.

## Maintenance

- Monitor drift and calibration on a recurring basis.
- Re-evaluate fairness whenever the model or population changes.
- Record each promoted model in the registry with its metrics and tags.
