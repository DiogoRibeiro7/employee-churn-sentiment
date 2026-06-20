"""Generate the deep-dive analysis notebooks (02–04).

Run with ``python notebooks/build_notebooks.py``. Notebooks are authored here as
code so they regenerate deterministically; execute them with nbconvert to embed
outputs. All three share a reproducible synthetic dataset (n=1500, seed=42) so
the prose analysis matches the committed outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT_DIR = Path(__file__).parent / "exploratory"

# Bootstrap so the package imports from a fresh checkout without installation.
BOOTSTRAP = (
    "# Make the package importable from a fresh checkout (no install needed).\n"
    "import sys\n"
    "from pathlib import Path\n"
    "ROOT = Path.cwd()\n"
    "while not (ROOT / 'employee_churn').exists() and ROOT != ROOT.parent:\n"
    "    ROOT = ROOT.parent\n"
    "if str(ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(ROOT))\n\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "sns.set_theme(style='whitegrid', palette='deep')\n"
    "plt.rcParams['figure.dpi'] = 110\n"
    "pd.set_option('display.float_format', lambda v: f'{v:.3f}')"
)

# Shared feature-engineering block reused by the modeling/fairness notebooks.
FEATURE_BLOCK = (
    "from employee_churn.data import make_synthetic_employee_data\n"
    "from employee_churn.features.engineer_structured import (\n"
    "    add_career_progression_features, add_tenure_bands,\n"
    "    add_promotion_velocity, add_compensation_features, add_team_metrics,\n"
    ")\n"
    "from employee_churn.features.engineer_text import add_text_statistics\n"
    "from employee_churn.nlp.sentiment import add_sentiment_scores\n"
    "from employee_churn.nlp.emotion import add_emotion_features\n\n"
    "df = make_synthetic_employee_data(n=1500, seed=42)\n"
    "f = add_career_progression_features(df, 'hire_date', 'last_promotion_date')\n"
    "f = add_tenure_bands(f)\n"
    "f = add_promotion_velocity(f, 'num_promotions')\n"
    "f = add_compensation_features(f, 'monthly_salary', 'department')\n"
    "f = add_team_metrics(f, 'team_id')\n"
    "f = add_sentiment_scores(f, 'feedback')\n"
    "f = add_emotion_features(f, 'feedback')\n"
    "f = add_text_statistics(f, 'feedback')\n\n"
    "DROP = ['employee_id', 'churned', 'feedback', 'gender', 'department',\n"
    "        'hire_date', 'last_promotion_date', 'team_id', 'tenure_band',\n"
    "        'emotion_dominant']\n"
    "X = f.drop(columns=DROP).select_dtypes(include=['number', 'bool'])\n"
    "y = f['churned']\n"
    "print('feature matrix:', X.shape)"
)


def md(text: str):
    return new_markdown_cell(text)


def code(src: str):
    return new_code_cell(src)


# --------------------------------------------------------------------------- #
# Notebook 02 — Exploratory Data Analysis
# --------------------------------------------------------------------------- #
EDA = [
    md(
        "# 02 · Exploratory Data Analysis\n\n"
        "This notebook explores the synthetic employee dataset to understand "
        "**what drives churn** before any modeling. The data is generated with a "
        "fixed seed, so every figure below is reproducible. The headline question: "
        "*do structured HR attributes or the free-text feedback carry the churn "
        "signal?*"
    ),
    code(BOOTSTRAP),
    code(
        "from employee_churn.data import make_synthetic_employee_data\n"
        "df = make_synthetic_employee_data(n=1500, seed=42)\n"
        "print('rows, cols:', df.shape)\n"
        "print('overall churn rate:', round(df['churned'].mean(), 3))\n"
        "df.head()"
    ),
    md(
        "## Dataset shape and class balance\n\n"
        "The sample has **1,500 employees** and **14 raw columns** spanning "
        "demographics, tenure/promotion dates, compensation, satisfaction and "
        "performance scores, and a free-text `feedback` field. The target "
        "`churned` is **~45% positive** — close to balanced, so accuracy would be "
        "a misleading metric and we will lean on ROC-AUC, F1, and precision@k "
        "instead."
    ),
    code(
        "summary = df.describe(include='number').T[['mean', 'std', 'min', 'max']]\n"
        "display(summary.round(2))\n\n"
        "fig, ax = plt.subplots(figsize=(4, 3))\n"
        "df['churned'].value_counts().sort_index().plot.bar(ax=ax, color=['#4c72b0', '#c44e52'])\n"
        "ax.set_xticklabels(['retained (0)', 'churned (1)'], rotation=0)\n"
        "ax.set_title('Class balance'); ax.set_ylabel('employees')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## Churn by department and gender\n\n"
        "Breaking churn down by categorical segments reveals only **mild** "
        "structural variation."
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))\n"
        "(df.groupby('department')['churned'].mean().sort_values(ascending=False)\n"
        "   .plot.bar(ax=axes[0], color='#4c72b0'))\n"
        "axes[0].axhline(df['churned'].mean(), ls='--', c='k', lw=1, label='overall')\n"
        "axes[0].set_title('Churn rate by department'); axes[0].set_ylabel('rate')\n"
        "axes[0].legend(); axes[0].tick_params(axis='x', rotation=30)\n"
        "(df.groupby('gender')['churned'].mean()\n"
        "   .plot.bar(ax=axes[1], color='#55a868'))\n"
        "axes[1].axhline(df['churned'].mean(), ls='--', c='k', lw=1)\n"
        "axes[1].set_title('Churn rate by gender'); axes[1].set_ylabel('rate')\n"
        "axes[1].tick_params(axis='x', rotation=0)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**Reading the charts.** Marketing churns the most (~52%) and operations "
        "the least (~40%), but every department sits within ~6 points of the "
        "overall 45% line — department alone is a weak predictor. The gender gap is "
        "also small in absolute terms (male ~48% vs nonbinary ~43%). Keep that "
        "gender spread in mind: it is small here, yet we will see in notebook 04 "
        "that the *model* amplifies it into a four-fifths-rule failure."
    ),
    code(
        "num = df.select_dtypes('number').drop(columns=['employee_id', 'team_id'])\n"
        "corr = num.corrwith(df['churned']).drop('churned').sort_values()\n"
        "fig, ax = plt.subplots(figsize=(6, 3.4))\n"
        "corr.plot.barh(ax=ax, color=np.where(corr < 0, '#4c72b0', '#c44e52'))\n"
        "ax.set_title('Correlation of structured features with churn')\n"
        "ax.axvline(0, c='k', lw=0.8)\n"
        "plt.tight_layout(); plt.show()\n"
        "corr.round(3)"
    ),
    md(
        "## Structured features carry a weak signal\n\n"
        "The strongest structured correlate of churn is **`satisfaction_score` at "
        "only −0.17**, followed by `performance_score` (−0.07). Compensation, age, "
        "and overtime are essentially flat (|r| < 0.05). No single structured "
        "feature is decisive — a linear model on structured data alone would "
        "struggle. This is the central motivation for mining the feedback text."
    ),
    code(
        "from employee_churn.nlp.sentiment import add_sentiment_scores\n"
        "from employee_churn.nlp.emotion import add_emotion_features\n"
        "t = add_sentiment_scores(df, 'feedback')\n"
        "t = add_emotion_features(t, 'feedback')\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))\n"
        "for label, color in [(0, '#4c72b0'), (1, '#c44e52')]:\n"
        "    sns.kdeplot(t.loc[t['churned'] == label, 'sentiment'], ax=axes[0],\n"
        "                fill=True, alpha=0.4, color=color, label=f'churned={label}')\n"
        "axes[0].set_title('VADER sentiment by churn'); axes[0].legend()\n"
        "sns.boxplot(data=t, x='churned', y='emotion_polarity', ax=axes[1],\n"
        "            palette=['#4c72b0', '#c44e52'])\n"
        "axes[1].set_title('Emotion polarity by churn (higher = more negative)')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "print('mean sentiment   ', t.groupby('churned')['sentiment'].mean().round(3).to_dict())\n"
        "print('mean emo polarity', t.groupby('churned')['emotion_polarity'].mean().round(3).to_dict())"
    ),
    md(
        "## The text tells the story\n\n"
        "Now the signal is obvious. **Retained employees average a clearly "
        "positive sentiment (~+0.27) while churners average negative (~−0.12)** — "
        "the two distributions visibly separate. The lexicon-based `emotion_polarity` "
        "(negative-emotion hits minus positive ones) flips sign across the two "
        "groups: churners skew positive (more fear/anger/sadness words), retained "
        "employees skew negative (more joy/trust words).\n\n"
        "Even with the lightweight VADER + lexicon approach, the feedback text "
        "separates the classes far better than any structured column. That is the "
        "thesis the modeling notebook will confirm quantitatively."
    ),
    md(
        "## Takeaways\n\n"
        "1. The target is ~balanced (45% churn) — use AUC/F1/precision@k, not "
        "accuracy.\n"
        "2. Structured features are individually weak (best |r| ≈ 0.17, "
        "satisfaction).\n"
        "3. **Text-derived sentiment and emotion are the strongest separators** — "
        "the modeling notebook should weight them heavily.\n"
        "4. A small raw gender gap exists; we must audit whether the model "
        "amplifies it (notebook 04)."
    ),
]


# --------------------------------------------------------------------------- #
# Notebook 03 — Modeling & Evaluation
# --------------------------------------------------------------------------- #
MODELING = [
    md(
        "# 03 · Modeling & Evaluation\n\n"
        "Building on the EDA, this notebook trains the full model zoo, compares "
        "them with cross-validation, tunes the strongest tree model, inspects "
        "calibration, and reads feature importances. Every step uses the same "
        "reproducible feature matrix."
    ),
    code(BOOTSTRAP),
    code(FEATURE_BLOCK),
    md(
        "## Feature matrix\n\n"
        "The engineered matrix has **31 numeric/boolean features** per employee: "
        "structured signals (tenure, promotion velocity, peer-relative "
        "compensation, team size) plus the text-derived sentiment, emotion, and "
        "text-statistics columns. Identifiers, raw dates, and the categorical "
        "`tenure_band`/`emotion_dominant` labels are dropped."
    ),
    code(
        "from employee_churn.models.train import build_model_zoo, cross_validate_models\n"
        "zoo = build_model_zoo(random_state=0)\n"
        "cv = cross_validate_models(zoo, X, y, cv=5)\n"
        "cv_df = pd.DataFrame(cv).T.sort_values('roc_auc_mean', ascending=False)\n"
        "display(cv_df.round(3))\n\n"
        "fig, ax = plt.subplots(figsize=(6.5, 3.4))\n"
        "ax.barh(cv_df.index, cv_df['roc_auc_mean'], xerr=cv_df['roc_auc_std'],\n"
        "        color='#4c72b0', capsize=4)\n"
        "ax.set_xlim(0.5, 0.8); ax.invert_yaxis()\n"
        "ax.set_title('5-fold CV ROC-AUC (mean ± std)'); ax.set_xlabel('ROC-AUC')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## Model comparison\n\n"
        "All four models land in a **narrow 0.66–0.71 ROC-AUC band**, with "
        "**logistic regression marginally ahead (~0.70)** of random forest and "
        "gradient boosting. The error bars overlap, so no model is decisively "
        "best — and the simplest, most interpretable one is competitive. The "
        "modest ceiling is expected: the synthetic target deliberately carries "
        "substantial noise, so this is a realistic 'no free lunch' result rather "
        "than a leaderboard-topping score. We carry the random forest forward "
        "because it gives us SHAP-friendly feature importances for notebook 04."
    ),
    code(
        "from sklearn.model_selection import train_test_split\n"
        "from employee_churn.models.train import tune_hyperparameters\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(\n"
        "    X, y, test_size=0.25, random_state=0, stratify=y)\n"
        "# Tune the random forest and carry this single model through the rest of\n"
        "# the notebook (and the fairness notebook) for a consistent story.\n"
        "best, params, cv_auc = tune_hyperparameters(\n"
        "    zoo['random_forest'], X_tr, y_tr, model_name='random_forest', n_iter=5)\n"
        "best.fit(X_tr, y_tr)\n"
        "print('tuned params:', params, '| CV AUC:', round(cv_auc, 3))"
    ),
    md(
        "## Tuning the carried-forward model\n\n"
        "Randomized search nudges the random forest toward a shallower, more "
        "regularized configuration (`max_depth=4`, `min_samples_leaf=4`, "
        "`n_estimators=200`, CV AUC ≈ 0.70), which guards against overfitting on "
        "1,125 training rows. Everything below evaluates this tuned model."
    ),
    code(
        "from sklearn.metrics import (roc_curve, auc, precision_recall_curve,\n"
        "                             ConfusionMatrixDisplay)\n"
        "proba = best.predict_proba(X_te)[:, 1]\n\n"
        "fpr, tpr, _ = roc_curve(y_te, proba)\n"
        "prec, rec, _ = precision_recall_curve(y_te, proba)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "axes[0].plot(fpr, tpr, color='#c44e52', lw=2, label=f'AUC = {auc(fpr, tpr):.3f}')\n"
        "axes[0].plot([0, 1], [0, 1], ls='--', c='gray')\n"
        "axes[0].set_xlabel('False positive rate'); axes[0].set_ylabel('True positive rate')\n"
        "axes[0].set_title('ROC curve (held-out)'); axes[0].legend()\n"
        "axes[1].plot(rec, prec, color='#4c72b0', lw=2)\n"
        "axes[1].axhline(y_te.mean(), ls='--', c='gray', label=f'baseline = {y_te.mean():.2f}')\n"
        "axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')\n"
        "axes[1].set_title('Precision–Recall curve'); axes[1].legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## Held-out discrimination\n\n"
        "On the 25% hold-out the random forest reaches **ROC-AUC ≈ 0.74** — a bit "
        "above its cross-validated mean, within normal split-to-split variance. "
        "The precision–recall curve sits comfortably above the 0.45 base rate, "
        "confirming the model ranks at-risk employees better than chance across "
        "the whole threshold range."
    ),
    code(
        "from employee_churn.models.evaluate import evaluate_model\n"
        "report = evaluate_model(best, X_te, y_te, top_k=20)\n"
        "print('discrimination:', {k: round(v, 3) for k, v in report['discrimination'].items()})\n"
        "print('calibration   :', {k: round(v, 3) for k, v in report['calibration'].items()})\n\n"
        "fig, ax = plt.subplots(figsize=(3.6, 3.2))\n"
        "ConfusionMatrixDisplay.from_predictions(\n"
        "    y_te, (proba >= 0.5).astype(int), ax=ax, colorbar=False, cmap='Blues')\n"
        "ax.set_title('Confusion matrix @0.5'); plt.tight_layout(); plt.show()"
    ),
    md(
        "## Precision@k for targeted outreach\n\n"
        "HR can only act on a limited number of people per cycle, so "
        "**precision@k** is the operationally honest metric. Among the model's "
        "20 highest-risk employees, **~70% truly churned** — a strong shortlist "
        "for retention conversations even though the global AUC is modest."
    ),
    code(
        "from employee_churn.models.calibrate import calibration_improvement, reliability_curve\n"
        "calib = calibration_improvement(best, X_tr, y_tr, X_te, y_te)\n"
        "base_curve = reliability_curve(y_te, best.predict_proba(X_te)[:, 1])\n"
        "cal_model = calib['calibrated_model']\n"
        "cal_curve = reliability_curve(y_te, cal_model.predict_proba(X_te)[:, 1])\n\n"
        "fig, ax = plt.subplots(figsize=(5, 5))\n"
        "ax.plot([0, 1], [0, 1], ls='--', c='gray', label='perfect')\n"
        "ax.plot(base_curve['mean_predicted'], base_curve['fraction_positive'],\n"
        "        'o-', color='#c44e52', label=f\"raw (ECE={calib['baseline']['expected_calibration_error']:.3f})\")\n"
        "ax.plot(cal_curve['mean_predicted'], cal_curve['fraction_positive'],\n"
        "        's-', color='#4c72b0', label=f\"isotonic (ECE={calib['calibrated']['expected_calibration_error']:.3f})\")\n"
        "ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed frequency')\n"
        "ax.set_title('Reliability curve'); ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## Calibration — an honest result\n\n"
        "**Isotonic calibration does "
        "*not* improve ECE here** (it edges from ~0.063 to ~0.065). On a dataset "
        "this size the isotonic fit adds variance that outweighs the small bias it "
        "removes — a real reminder that calibration is an empirical question, not "
        "a guaranteed win. In production you would prefer Platt/sigmoid scaling on "
        "small samples, or simply keep the raw probabilities and monitor ECE over "
        "time."
    ),
    code(
        "imp = pd.Series(best.feature_importances_, index=X.columns).sort_values()\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "imp.tail(12).plot.barh(ax=ax, color='#55a868')\n"
        "ax.set_title('Top 12 random-forest feature importances')\n"
        "plt.tight_layout(); plt.show()\n"
        "imp.sort_values(ascending=False).head(8).round(3)"
    ),
    md(
        "## What the model actually uses\n\n"
        "The importance ranking validates the EDA decisively: the **top two "
        "features are `emotion_polarity` and `sentiment`** (~0.14 each), with "
        "`text_char_count` and individual emotions (`emotion_joy`, `emotion_anger`, "
        "`emotion_fear`) close behind. `days_since_promotion` is the only "
        "structured feature near the top. **Roughly half of the model's predictive "
        "weight comes from the free-text feedback** — a strong argument for "
        "investing in richer NLP (the transformer sentiment backend on the "
        "roadmap) rather than collecting more structured HR fields."
    ),
    md(
        "## Takeaways\n\n"
        "1. Models cluster around **0.70 CV AUC / 0.74 hold-out**; logistic "
        "regression is competitive with the trees.\n"
        "2. **Precision@20 ≈ 0.70** makes the model useful for targeted outreach "
        "despite the modest global AUC.\n"
        "3. **Calibration did not help** on this sample size — verify, don't "
        "assume.\n"
        "4. **Text features dominate importances**, confirming the EDA thesis."
    ),
]


# --------------------------------------------------------------------------- #
# Notebook 04 — Fairness & Explainability
# --------------------------------------------------------------------------- #
FAIRNESS = [
    md(
        "# 04 · Fairness & Explainability\n\n"
        "A churn model that flags protected groups at different rates is a legal "
        "and ethical liability. This notebook audits group fairness across gender "
        "and uses SHAP to explain *why* the model scores employees as it does — "
        "both at the population level and for individuals."
    ),
    code(BOOTSTRAP),
    code(FEATURE_BLOCK),
    code(
        "from sklearn.model_selection import train_test_split\n"
        "from employee_churn.models.train import build_model_zoo, tune_hyperparameters\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(\n"
        "    X, y, test_size=0.25, random_state=0, stratify=y)\n"
        "# Audit the same tuned random forest carried forward from notebook 03.\n"
        "best, _, _ = tune_hyperparameters(\n"
        "    build_model_zoo(0)['random_forest'], X_tr, y_tr,\n"
        "    model_name='random_forest', n_iter=5)\n"
        "best.fit(X_tr, y_tr)\n"
        "sensitive = f.loc[y_te.index, 'gender']\n"
        "preds = best.predict(X_te)\n"
        "print('test employees:', len(y_te))"
    ),
    md(
        "## Group fairness audit\n\n"
        "We compare, per gender, the **selection rate** (share flagged high-risk), "
        "**true-positive rate** (recall — equal opportunity), and false-positive "
        "rate."
    ),
    code(
        "from employee_churn.models.fairness import group_fairness_report, fairness_summary\n"
        "report = group_fairness_report(y_te.values, preds, sensitive.values)\n"
        "display(report.round(3))\n\n"
        "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
        "report.set_index('group')[['selection_rate', 'true_positive_rate', 'false_positive_rate']].plot.bar(ax=ax)\n"
        "ax.set_title('Fairness metrics by gender'); ax.set_ylabel('rate')\n"
        "ax.tick_params(axis='x', rotation=0); ax.legend(loc='upper right', fontsize=8)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "summary = fairness_summary(report)\n"
        "{k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}"
    ),
    md(
        "## The model fails the four-fifths rule\n\n"
        "This is the most important finding in the whole analysis. The "
        "**disparate-impact ratio is ≈ 0.64**, well below the **0.80 four-fifths "
        "threshold** — so the model `passes_four_fifths = False`. Concretely, the "
        "least-flagged group (nonbinary) is selected at only ~64% the rate of the "
        "most-flagged group (female), and the equal-opportunity gap (difference in "
        "recall across groups) is ~0.18.\n\n"
        "Recall from the EDA that the *raw* gender gap in churn was only a few "
        "points. The model has **amplified** a small population difference into a "
        "materially disparate selection rate. Deploying this as-is could "
        "systematically under- or over-target retention resources by gender.\n\n"
        "**Mitigations to consider:** drop or audit features proxying for gender, "
        "apply group-specific thresholds to equalize selection rates, add a "
        "fairness constraint / post-processing step (e.g. equalized odds), and "
        "always keep a human in the loop. Fairness should be re-checked on every "
        "retrain because, as shown here, it does not follow from a 'fair-looking' "
        "dataset."
    ),
    code(
        "from employee_churn.models.explain import explain_with_shap\n"
        "sample = X_te.head(150)\n"
        "shap_df = explain_with_shap(best, sample)\n"
        "mean_abs = shap_df.abs().mean().sort_values()\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "mean_abs.tail(12).plot.barh(ax=ax, color='#8172b3')\n"
        "ax.set_title('Mean |SHAP value| — global feature impact')\n"
        "ax.set_xlabel('mean |SHAP|'); plt.tight_layout(); plt.show()"
    ),
    md(
        "## Global explanations agree with importances\n\n"
        "Mean absolute SHAP values — a more principled global attribution than "
        "impurity importances — tell the same story: **sentiment and emotion "
        "features carry the most influence on individual predictions**, alongside "
        "`days_since_promotion`. Because SHAP is additive and signed, we can also "
        "open up *individual* predictions."
    ),
    code(
        "# Explain the two highest-risk employees in the sample.\n"
        "risk = pd.Series(best.predict_proba(sample)[:, 1], index=sample.index)\n"
        "top2 = risk.nlargest(2).index\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))\n"
        "for ax, idx in zip(axes, top2):\n"
        "    contrib = shap_df.loc[idx].reindex(mean_abs.index).tail(8)\n"
        "    colors = np.where(contrib >= 0, '#c44e52', '#4c72b0')\n"
        "    contrib.plot.barh(ax=ax, color=colors)\n"
        "    ax.set_title(f'Employee {idx} — risk={risk.loc[idx]:.2f}')\n"
        "    ax.axvline(0, c='k', lw=0.8); ax.set_xlabel('SHAP (→ pushes toward churn)')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## Individual explanations enable action\n\n"
        "For each flagged employee the signed SHAP bars show **which factors push "
        "their risk up (red) versus down (blue)**. This is what makes the score "
        "actionable: rather than an opaque number, a manager sees that a given "
        "person's risk is driven by, say, negative recent feedback and a long gap "
        "since promotion — pointing to concrete retention levers (a career "
        "conversation, a check-in on workload).\n\n"
        "Used responsibly — *with* the fairness caveats above — these explanations "
        "turn the model from a surveillance tool into a support tool."
    ),
    md(
        "## Takeaways\n\n"
        "1. **The model fails the four-fifths rule (DI ≈ 0.64)** and must be "
        "mitigated before any deployment.\n"
        "2. A small dataset gender gap was **amplified** by the model — audit "
        "fairness on every retrain.\n"
        "3. SHAP confirms **text features dominate**, and enables per-employee, "
        "actionable explanations.\n"
        "4. Pair every score with its explanation and a human decision-maker."
    ),
]


def build(name: str, cells: list) -> Path:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    nbformat.write(nb, path)
    return path


def main() -> None:
    """Write all deep-dive notebooks."""
    for name, cells in [
        ("02_exploratory_data_analysis.ipynb", EDA),
        ("03_modeling_and_evaluation.ipynb", MODELING),
        ("04_fairness_and_explainability.ipynb", FAIRNESS),
    ]:
        print("wrote", build(name, cells))


if __name__ == "__main__":
    main()
