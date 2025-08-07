# README.md

# Employee Churn with Sentiment Analysis

This project aims to predict employee churn by combining structured HR data (e.g., tenure, promotions, performance) with unstructured textual feedback from performance reviews, surveys, and exit interviews. We use natural language processing (NLP) to extract sentiment and emotional signals, improving early detection of disengagement.

## 🔍 Objective

To identify at-risk employees before they leave by leveraging emotional and behavioral signals, allowing HR and managers to take timely action.

## 📦 Features

- Churn prediction models using structured and unstructured data
- Sentiment and emotion analysis on employee feedback
- Feature engineering on career progression, team dynamics, and satisfaction
- Explainable outputs for HR decision-making
- Early warning dashboards for high-risk top performers

## 📁 Project Structure

```text
employee-churn-sentiment/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── inference/
│   ├── visualization/
│   └── tracking/
├── tests/
├── scripts/
├── README.md
└── ROADMAP.md
```

## 🚀 Getting Started

1. Clone this repository.
2. Install dependencies via Poetry:

```bash
poetry install
```

3. Run the first notebook:

```bash
jupyter lab notebooks/01_eda.ipynb
```

## 📊 Tech Stack

- Python, scikit-learn, XGBoost
- spaCy, NLTK, VADER, BERTopic
- SHAP, MLflow or Weights & Biases
- Streamlit (optional dashboard)

## 📄 License

MIT License (see `LICENSE` file)# README.md

# Employee Churn with Sentiment Analysis

This project aims to predict employee churn by combining structured HR data (e.g., tenure, promotions, performance) with unstructured textual feedback from performance reviews, surveys, and exit interviews. We use natural language processing (NLP) to extract sentiment and emotional signals, improving early detection of disengagement.

## 🔍 Objective

To identify at-risk employees before they leave by leveraging emotional and behavioral signals, allowing HR and managers to take timely action.

## 📦 Features

- Churn prediction models using structured and unstructured data
- Sentiment and emotion analysis on employee feedback
- Feature engineering on career progression, team dynamics, and satisfaction
- Explainable outputs for HR decision-making
- Early warning dashboards for high-risk top performers

## 📁 Project Structure

```text
employee-churn-sentiment/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── inference/
│   ├── visualization/
│   └── tracking/
├── tests/
├── scripts/
├── README.md
└── ROADMAP.md
```

## 🚀 Getting Started

1. Clone this repository.
2. Install dependencies via Poetry:

```bash
poetry install
```

3. Run the first notebook:

```bash
jupyter lab notebooks/01_eda.ipynb
```

## 📊 Tech Stack

- Python, scikit-learn, XGBoost
- spaCy, NLTK, VADER, BERTopic
- SHAP, MLflow or Weights & Biases
- Streamlit (optional dashboard)

## 📄 License

MIT License (see `LICENSE` file)
