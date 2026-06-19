# ROADMAP.md

# Roadmap: Employee Churn with Sentiment Analysis

This document outlines the development phases and goals for the project.

---

## Phase 1: Problem Definition & Data Collection ✅

- [x] Define churn target (binary, e.g., resignation within 60 days)
- [x] Collect structured HR data: age, tenure, performance, promotions
- [x] Collect unstructured data: reviews, surveys, exit interviews

---

## Phase 2: Data Preprocessing & Feature Engineering 🚧

- [x] Clean and standardize HR fields
- [x] Anonymize unstructured text
- [x] Generate sentiment scores (VADER, TextBlob, etc.)
- [x] Extract emotion categories (e.g., joy, fear, trust)
- [x] Engineer career progression and team metrics

---

## Phase 3: Modeling 🔄

- [x] Baseline churn models (LogReg, Random Forest, XGBoost)
- [x] NLP-only model (e.g., sentiment-based)
- [x] Combined model (structured + text)
- [x] Evaluate using AUC, F1, Precision@TopK
- [x] Track experiments with MLflow or Weights & Biases

---

## Phase 4: Explainability & Risk Scoring 🧠

- [x] Add SHAP explanations per prediction
- [x] Score employees weekly based on latest inputs
- [x] Build ranking dashboard of at-risk individuals

---

## Phase 5: Deployment & Monitoring 🚀

 - [x] Create Streamlit or web-based dashboard for HR
 - [x] Export scores via API endpoint or CSV
 - [x] Monitor performance drift and model calibration
 - [x] Set alert thresholds for high-risk scores

---

## Phase 6: ML Rigor & Feature Depth ✅

- [x] Reproducible synthetic dataset generator for examples and tests
- [x] Richer structured features (tenure bands, promotion velocity, peer-relative compensation)
- [x] Shape-based text-statistics features
- [x] Plutchik eight-emotion lexicon with intensity, polarity, and dominant emotion
- [x] Expanded model zoo (gradient boosting, hist gradient boosting)
- [x] Stratified cross-validation and randomized hyperparameter tuning
- [x] Probability calibration with before/after reliability metrics
- [x] Group fairness diagnostics (selection parity, equal opportunity, four-fifths rule)
- [x] End-to-end runnable walkthrough notebook

---

## Future Enhancements ✨

- Add temporal models for event streams (e.g., LSTM, Transformer)
- Integrate passive sentiment from internal chat (with consent)
- Build personalized retention recommendation engine
- Transformer-based sentiment/emotion models as an optional backend
- FastAPI scoring endpoint for real-time predictions
