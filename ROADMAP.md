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

- [ ] Clean and standardize HR fields
- [ ] Anonymize unstructured text
- [ ] Generate sentiment scores (VADER, TextBlob, etc.)
- [ ] Extract emotion categories (e.g., joy, fear, trust)
- [ ] Engineer career progression and team metrics

---

## Phase 3: Modeling 🔄

- [ ] Baseline churn models (LogReg, Random Forest, XGBoost)
- [ ] NLP-only model (e.g., sentiment-based)
- [ ] Combined model (structured + text)
- [ ] Evaluate using AUC, F1, Precision@TopK
- [ ] Track experiments with MLflow or Weights & Biases

---

## Phase 4: Explainability & Risk Scoring 🧠

- [ ] Add SHAP explanations per prediction
- [ ] Score employees weekly based on latest inputs
- [ ] Build ranking dashboard of at-risk individuals

---

## Phase 5: Deployment & Monitoring 🚀

- [ ] Create Streamlit or web-based dashboard for HR
- [ ] Export scores via API endpoint or CSV
- [ ] Monitor performance drift and model calibration
- [ ] Set alert thresholds for high-risk scores

---

## Future Enhancements ✨

- Add temporal models for event streams (e.g., LSTM, Transformer)
- Integrate passive sentiment from internal chat (with consent)
- Build personalized retention recommendation engine
