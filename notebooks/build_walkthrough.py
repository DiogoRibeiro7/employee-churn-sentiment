"""Generate the churn-modeling walkthrough notebook.

Run with ``python notebooks/build_walkthrough.py``. Kept in-repo so the notebook
can be regenerated deterministically rather than hand-edited as JSON.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELLS = [
    new_markdown_cell(
        "# Employee Churn — End-to-End Walkthrough\n\n"
        "This notebook runs the full modeling workflow on a **reproducible "
        "synthetic dataset**, so it executes anywhere with no access to real HR "
        "records. It covers:\n\n"
        "1. Synthetic data generation\n"
        "2. Structured + text feature engineering\n"
        "3. A multi-model zoo with cross-validation\n"
        "4. Probability calibration\n"
        "5. Group fairness diagnostics\n"
        "6. SHAP explainability"
    ),
    new_code_cell(
        "# Make the package importable when running from a fresh checkout\n"
        "# (no installation required).\n"
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd()\n"
        "while not (ROOT / 'employee_churn').exists() and ROOT != ROOT.parent:\n"
        "    ROOT = ROOT.parent\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "import pandas as pd\n"
        "from employee_churn.data import make_synthetic_employee_data\n\n"
        "df = make_synthetic_employee_data(n=800, seed=42)\n"
        "print(df.shape)\n"
        "print('churn rate:', round(df['churned'].mean(), 3))\n"
        "df.head()"
    ),
    new_markdown_cell(
        "## 1. Feature engineering\n\n"
        "We combine career-progression, tenure-band, promotion-velocity, "
        "peer-relative compensation, sentiment, emotion, and text-shape features."
    ),
    new_code_cell(
        "from employee_churn.features.engineer_structured import (\n"
        "    add_career_progression_features,\n"
        "    add_tenure_bands,\n"
        "    add_promotion_velocity,\n"
        "    add_compensation_features,\n"
        "    add_team_metrics,\n"
        ")\n"
        "from employee_churn.features.engineer_text import add_text_statistics\n"
        "from employee_churn.nlp.sentiment import add_sentiment_scores\n"
        "from employee_churn.nlp.emotion import add_emotion_features\n\n"
        "feat = add_career_progression_features(df, 'hire_date', 'last_promotion_date')\n"
        "feat = add_tenure_bands(feat)\n"
        "feat = add_promotion_velocity(feat, 'num_promotions')\n"
        "feat = add_compensation_features(feat, 'monthly_salary', 'department')\n"
        "feat = add_team_metrics(feat, 'team_id')\n"
        "feat = add_sentiment_scores(feat, 'feedback')\n"
        "feat = add_emotion_features(feat, 'feedback')\n"
        "feat = add_text_statistics(feat, 'feedback')\n"
        "feat.filter(regex='sentiment|emotion_|text_|salary_|promotions_per').head()"
    ),
    new_markdown_cell(
        "## 2. Build the modeling matrix\n\n"
        "Keep numeric/boolean columns, drop identifiers and raw dates, and "
        "one-hot encode the remaining low-cardinality categoricals."
    ),
    new_code_cell(
        "sensitive = feat['gender'].copy()\n"
        "target = feat['churned'].copy()\n\n"
        "drop_cols = [\n"
        "    'employee_id', 'churned', 'feedback', 'gender', 'department',\n"
        "    'hire_date', 'last_promotion_date', 'team_id', 'tenure_band',\n"
        "    'emotion_dominant',\n"
        "]\n"
        "X = feat.drop(columns=drop_cols).select_dtypes(include=['number', 'bool'])\n"
        "y = target\n"
        "print(X.shape)\n"
        "X.head()"
    ),
    new_markdown_cell(
        "## 3. Model zoo + cross-validation\n\n"
        "Compare every classifier with stratified 5-fold cross-validation."
    ),
    new_code_cell(
        "from employee_churn.models.train import build_model_zoo, cross_validate_models\n\n"
        "zoo = build_model_zoo(random_state=0)\n"
        "cv_results = cross_validate_models(zoo, X, y, cv=5)\n"
        "pd.DataFrame(cv_results).T.sort_values('roc_auc_mean', ascending=False)"
    ),
    new_markdown_cell(
        "## 4. Hyperparameter tuning + calibration\n\n"
        "Tune the strongest tree model, then check whether calibration improves "
        "the reliability of its probabilities."
    ),
    new_code_cell(
        "from sklearn.model_selection import train_test_split\n"
        "from employee_churn.models.train import tune_hyperparameters\n"
        "from employee_churn.models.calibrate import calibration_improvement\n\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(\n"
        "    X, y, test_size=0.25, random_state=0, stratify=y\n"
        ")\n"
        "best, params, score = tune_hyperparameters(\n"
        "    zoo['random_forest'], X_tr, y_tr, model_name='random_forest', n_iter=5\n"
        ")\n"
        "print('best params:', params, 'cv roc_auc:', round(score, 3))\n\n"
        "calib = calibration_improvement(best, X_tr, y_tr, X_te, y_te)\n"
        "print('ECE before:', round(calib['baseline']['expected_calibration_error'], 4))\n"
        "print('ECE after :', round(calib['calibrated']['expected_calibration_error'], 4))\n"
        "print('improved  :', calib['improved'])"
    ),
    new_markdown_cell(
        "## 5. Fairness diagnostics\n\n"
        "Check selection-rate parity across the (synthetic) gender attribute "
        "using the four-fifths rule."
    ),
    new_code_cell(
        "from employee_churn.models.fairness import (\n"
        "    group_fairness_report,\n"
        "    fairness_summary,\n"
        ")\n\n"
        "preds = best.fit(X_tr, y_tr).predict(X_te)\n"
        "report = group_fairness_report(y_te.values, preds, sensitive.loc[y_te.index].values)\n"
        "display(report)\n"
        "fairness_summary(report)"
    ),
    new_markdown_cell(
        "## 6. Explainability\n\n" "SHAP attributions for a sample of employees."
    ),
    new_code_cell(
        "from employee_churn.models.explain import explain_with_shap\n\n"
        "sample = X_te.head(50)\n"
        "shap_df = explain_with_shap(best, sample)\n"
        "shap_df.abs().mean().sort_values(ascending=False).head(10)"
    ),
]


def main() -> None:
    """Build and write the walkthrough notebook to disk."""
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = Path(__file__).parent / "exploratory" / "01_churn_modeling_walkthrough.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
