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
        "signal?*\n\n"
        "### Contents\n"
        "1. [Dataset overview & class balance](#overview)\n"
        "2. [Data-quality check](#quality)\n"
        "3. [Univariate distributions](#univariate)\n"
        "4. [Churn by segment + significance tests](#segments)\n"
        "5. [Correlation structure](#correlation)\n"
        "6. [Tenure dynamics](#tenure)\n"
        "7. [The text signal: sentiment, emotion & word frequency](#text)\n"
        "8. [Takeaways](#takeaways)"
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
        '<a id="overview"></a>\n'
        "## 1 · Dataset shape and class balance\n\n"
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
        '<a id="quality"></a>\n'
        "## 2 · Data-quality check\n\n"
        "Before trusting any chart, confirm the data is clean: missing values, "
        "duplicates, and obviously invalid ranges."
    ),
    code(
        "quality = pd.DataFrame({\n"
        "    'dtype': df.dtypes.astype(str),\n"
        "    'n_missing': df.isna().sum(),\n"
        "    'pct_missing': (df.isna().mean() * 100).round(2),\n"
        "    'n_unique': df.nunique(),\n"
        "})\n"
        "display(quality)\n"
        "print('duplicate rows:', int(df.duplicated().sum()))\n"
        "print('negative tenure rows:',\n"
        "      int((pd.to_datetime(df['last_promotion_date']) < pd.to_datetime(df['hire_date'])).sum()))"
    ),
    md(
        "**Clean bill of health.** The synthetic generator produces **zero missing "
        "values, zero duplicate rows**, and no promotion-before-hire violations — "
        "so the rest of the analysis needs no imputation. (Real HRIS extracts are "
        "rarely this tidy; the package's `data.clean`/`data.validate` helpers exist "
        "precisely to enforce these checks on live data.)"
    ),
    md(
        '<a id="univariate"></a>\n'
        "## 3 · Univariate distributions\n\n"
        "How is each numeric driver distributed, and does its shape differ between "
        "churners and stayers?"
    ),
    code(
        "num_cols = ['age', 'monthly_salary', 'satisfaction_score',\n"
        "            'performance_score', 'overtime_hours', 'num_promotions']\n"
        "fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))\n"
        "for ax, col in zip(axes.ravel(), num_cols):\n"
        "    for label, color in [(0, '#4c72b0'), (1, '#c44e52')]:\n"
        "        sns.histplot(df.loc[df['churned'] == label, col], ax=ax, color=color,\n"
        "                     alpha=0.5, kde=True, stat='density', label=f'churned={label}')\n"
        "    ax.set_title(col); ax.set_xlabel(''); ax.legend(fontsize=7)\n"
        "fig.suptitle('Numeric feature distributions by churn outcome', y=1.02)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**Reading the grid.** Most distributions overlap heavily between the two "
        "classes — `age`, `monthly_salary`, `num_promotions`, and `overtime_hours` "
        "look nearly identical regardless of outcome. The clearest visible shift is "
        "in **`satisfaction_score`**, where churners lean lower, and a milder shift "
        "in `performance_score`. We quantify these impressions with hypothesis "
        "tests below rather than eyeballing them."
    ),
    md(
        '<a id="segments"></a>\n'
        "## 4 · Churn by department and gender\n\n"
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
    md(
        "### Are these differences statistically significant?\n\n"
        "Eyeballing rates is not enough. We run **Mann–Whitney U** tests for each "
        "numeric feature (churned vs retained distributions) and **chi-square** "
        "tests of independence for the categorical segments."
    ),
    code(
        "from scipy import stats\n"
        "num_cols = ['satisfaction_score', 'performance_score', 'overtime_hours',\n"
        "            'num_promotions', 'age', 'monthly_salary']\n"
        "ch, rt = df[df['churned'] == 1], df[df['churned'] == 0]\n"
        "rows = []\n"
        "for c in num_cols:\n"
        "    u, p = stats.mannwhitneyu(ch[c], rt[c])\n"
        "    rows.append({'feature': c, 'p_value': p,\n"
        "                 'significant_(p<0.05)': p < 0.05})\n"
        "mw = pd.DataFrame(rows).sort_values('p_value').reset_index(drop=True)\n"
        "display(mw)\n"
        "for cat in ['department', 'gender']:\n"
        "    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[cat], df['churned']))\n"
        "    print(f'chi-square {cat:11s}: p = {p:.4f}'\n"
        "          f\"  ({'significant' if p < 0.05 else 'NOT significant'})\")"
    ),
    md(
        "**This is a pivotal result.** Only **`satisfaction_score` (p ≈ 0) and "
        "`performance_score` (p ≈ 0.007)** differ significantly between churners "
        "and stayers; overtime is borderline (p ≈ 0.08) and salary/age/promotions "
        "are not significant. Crucially, **neither `department` (p ≈ 0.08) nor "
        "`gender` (p ≈ 0.21) is significantly associated with churn.**\n\n"
        "Hold onto the gender result: in the data, gender carries *no* statistically "
        "detectable churn signal — yet notebook 04 shows the trained model still "
        "flags genders at materially different rates. That gap between *data* and "
        "*model* behaviour is exactly why a dedicated fairness audit is "
        "non-negotiable."
    ),
    md('<a id="correlation"></a>\n' "## 5 · Correlation structure"),
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
        "fig, ax = plt.subplots(figsize=(7.5, 6))\n"
        "sns.heatmap(num.assign(churned=df['churned']).corr(), annot=True, fmt='.2f',\n"
        "            cmap='RdBu_r', center=0, square=True, cbar_kws={'shrink': 0.8},\n"
        "            annot_kws={'size': 7}, ax=ax)\n"
        "ax.set_title('Correlation matrix (structured features + churn)')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**No multicollinearity surprises.** The off-diagonal correlations are all "
        "near zero, so the structured features are largely independent — there is "
        "no redundant pair to prune, but also no structured feature combination "
        "that obviously interacts. The `churned` row/column is pale across the "
        "board, visually reinforcing that the structured block is a weak predictor "
        "set on its own."
    ),
    md(
        '<a id="tenure"></a>\n'
        "## 6 · Tenure dynamics\n\n"
        "Churn risk is often highly non-linear in tenure, which is why the package "
        "buckets it into ordinal bands. Does risk vary across the career stage?"
    ),
    code(
        "from employee_churn.features.engineer_structured import (\n"
        "    add_career_progression_features, add_tenure_bands)\n"
        "tf = add_career_progression_features(df, 'hire_date', 'last_promotion_date')\n"
        "tf = add_tenure_bands(tf)\n"
        "band = tf.groupby('tenure_band', observed=True)['churned'].agg(['mean', 'size'])\n"
        "display(band.round(3))\n\n"
        "fig, ax = plt.subplots(figsize=(6, 3.4))\n"
        "band['mean'].plot.bar(ax=ax, color='#dd8452')\n"
        "ax.axhline(df['churned'].mean(), ls='--', c='k', lw=1, label='overall')\n"
        "ax.set_title('Churn rate by tenure band'); ax.set_ylabel('churn rate')\n"
        "ax.tick_params(axis='x', rotation=0); ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**A gentle senior-tenure uptick.** Risk is essentially flat for the 1–3y "
        "and 3–7y bands (~0.44) and rises modestly for **7y+ employees (~0.49)** — "
        "consistent with long-tenure staff who have stagnated (note `days_since_"
        "promotion` was the strongest *structured* model feature in notebook 03). "
        "The effect is real but small; tenure alone will not carry a model."
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
        '<a id="text"></a>\n'
        "## 7 · The text tells the story\n\n"
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
        "### Which words distinguish the two groups?\n\n"
        "Using the package's `nlp.preprocessing` pipeline (clean → tokenize → "
        "stopword removal), we count the most frequent content words in each "
        "group's feedback."
    ),
    code(
        "from collections import Counter\n"
        "from employee_churn.nlp.preprocessing import preprocess\n\n"
        "def top_words(texts, n=10):\n"
        "    counter = Counter()\n"
        "    for txt in texts:\n"
        "        counter.update(preprocess(txt))\n"
        "    return pd.Series(dict(counter.most_common(n)))\n\n"
        "churn_words = top_words(df.loc[df['churned'] == 1, 'feedback'])\n"
        "stay_words = top_words(df.loc[df['churned'] == 0, 'feedback'])\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "churn_words.sort_values().plot.barh(ax=axes[0], color='#c44e52')\n"
        "axes[0].set_title('Top words — churned')\n"
        "stay_words.sort_values().plot.barh(ax=axes[1], color='#4c72b0')\n"
        "axes[1].set_title('Top words — retained')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**The vocabulary splits cleanly.** Churners' feedback is dominated by "
        "`workload`, hedging/neutral language (`no`, `strong`, `feelings`, "
        "`either`, `way` — the 'no strong feelings either way' register of "
        "disengagement), while retained employees write about `recognition`, "
        "`trust`, `leadership`, `team`, and `happy`. The presence of explicitly "
        "positive relational words on the retained side — and their near-absence "
        "on the churn side — is exactly the signal VADER and the emotion lexicon "
        "pick up. A future transformer-based encoder (on the roadmap) would capture "
        "this even more richly."
    ),
    md(
        '<a id="takeaways"></a>\n'
        "## Takeaways\n\n"
        "1. **Data is clean** (no missing/duplicate/invalid rows) and the target is "
        "~balanced (45% churn) — use AUC/F1/precision@k, not accuracy.\n"
        "2. Only **`satisfaction_score` and `performance_score`** are statistically "
        "significant structured drivers; salary, age, promotions, department, and "
        "**gender are not** (chi-square p ≈ 0.21).\n"
        "3. Structured features are individually weak (best |r| ≈ 0.17) and "
        "mutually uncorrelated — no single one is decisive.\n"
        "4. **Text-derived sentiment, emotion, and vocabulary are the strongest "
        "separators** — the modeling notebook should (and does) weight them "
        "heavily.\n"
        "5. Gender shows **no significant raw association** with churn, so any "
        "model-level gender disparity is model-induced — audited in notebook 04."
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
        "reproducible feature matrix.\n\n"
        "### Contents\n"
        "1. [Feature matrix](#matrix)\n"
        "2. [Model zoo + cross-validation](#zoo)\n"
        "3. [Tuning the carried-forward model](#tuning)\n"
        "4. [Discrimination: ROC & PR](#roc)\n"
        "5. [Operating threshold selection](#threshold)\n"
        "6. [Cumulative gains & lift (business view)](#lift)\n"
        "7. [Calibration](#calibration)\n"
        "8. [Feature importance: impurity vs permutation](#importance)\n"
        "9. [Takeaways](#takeaways)"
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
    md(
        '<a id="threshold"></a>\n'
        "## 5 · Choosing an operating threshold\n\n"
        "The default 0.5 cutoff is rarely optimal. Sweeping the threshold shows the "
        "precision/recall trade-off and the F1-maximizing operating point."
    ),
    code(
        "from sklearn.metrics import precision_score, recall_score, f1_score\n"
        "ths = np.linspace(0.1, 0.9, 33)\n"
        "prec = [precision_score(y_te, proba >= t, zero_division=0) for t in ths]\n"
        "rec = [recall_score(y_te, proba >= t) for t in ths]\n"
        "f1s = [f1_score(y_te, proba >= t) for t in ths]\n"
        "best_t = ths[int(np.argmax(f1s))]\n\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.plot(ths, prec, label='precision', color='#4c72b0')\n"
        "ax.plot(ths, rec, label='recall', color='#dd8452')\n"
        "ax.plot(ths, f1s, label='F1', color='#c44e52', lw=2.5)\n"
        "ax.axvline(best_t, ls='--', c='k', lw=1, label=f'best F1 @ {best_t:.2f}')\n"
        "ax.set_xlabel('decision threshold'); ax.set_ylabel('score')\n"
        "ax.set_title('Metrics vs decision threshold'); ax.legend()\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'F1 at 0.50 = {f1_score(y_te, proba >= 0.5):.3f} | '\n"
        "      f'best F1 = {max(f1s):.3f} at threshold {best_t:.2f}')"
    ),
    md(
        "**Lowering the bar pays off.** F1 peaks at **~0.71 at a threshold of "
        "0.35**, well above the ~0.56 F1 the default 0.5 cutoff delivers. Because a "
        "missed departure is costly, HR would deliberately run the model at a "
        "*lower* threshold to trade some precision for materially higher recall — "
        "the exact point is a business call about the cost of a missed departure "
        "versus an unnecessary retention conversation."
    ),
    md(
        '<a id="lift"></a>\n'
        "## 6 · Cumulative gains & lift\n\n"
        "For a fixed outreach budget, the operational question is: *if we contact "
        "the top X% of the risk ranking, what share of real churners do we catch, "
        "and how much better is that than random?*"
    ),
    code(
        "order = np.argsort(-proba)\n"
        "y_sorted = y_te.values[order]\n"
        "pcts = np.arange(1, len(y_sorted) + 1) / len(y_sorted)\n"
        "gains = np.cumsum(y_sorted) / y_sorted.sum()\n"
        "base = y_te.mean()\n"
        "decile_capture = y_sorted[: int(0.1 * len(y_sorted))].mean()\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "axes[0].plot(pcts, gains, color='#c44e52', lw=2, label='model')\n"
        "axes[0].plot([0, 1], [0, 1], ls='--', c='gray', label='random')\n"
        "axes[0].set_xlabel('fraction contacted (high to low risk)')\n"
        "axes[0].set_ylabel('fraction of churners captured')\n"
        "axes[0].set_title('Cumulative gains'); axes[0].legend()\n"
        "lift = (np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)) / base\n"
        "axes[1].plot(pcts, lift, color='#4c72b0', lw=2)\n"
        "axes[1].axhline(1.0, ls='--', c='gray', label='random (lift=1)')\n"
        "axes[1].set_xlabel('fraction contacted')\n"
        "axes[1].set_ylabel('lift over base rate'); axes[1].set_title('Lift curve')\n"
        "axes[1].legend(); plt.tight_layout(); plt.show()\n"
        "print(f'Top 10% capture rate = {decile_capture:.1%} '\n"
        "      f'(lift {decile_capture / base:.2f}x over the {base:.1%} base rate)')\n"
        "print(f'Top 30% captures {gains[int(0.3*len(y_sorted)) - 1]:.1%} of all churners')"
    ),
    md(
        "**Strong targeting value.** The top risk decile churns at **~76% — a "
        "~1.7x lift** over the 45% base rate — and contacting the **top 30% of the "
        "ranking captures ~47% of all churners**. The gains curve bows well above "
        "the diagonal across the whole range. So even with a middling AUC, the "
        "model is genuinely useful as a *prioritization* tool, which is how "
        "retention programs actually run (finite outreach capacity, ranked "
        "worklist)."
    ),
    md('<a id="calibration"></a>'),
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
    md('<a id="importance"></a>'),
    code(
        "imp = pd.Series(best.feature_importances_, index=X.columns).sort_values()\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "imp.tail(12).plot.barh(ax=ax, color='#55a868')\n"
        "ax.set_title('Top 12 random-forest impurity importances')\n"
        "plt.tight_layout(); plt.show()\n"
        "imp.sort_values(ascending=False).head(8).round(3)"
    ),
    md(
        "## 8 · What the model actually uses\n\n"
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
        "### Cross-checking with permutation importance\n\n"
        "Impurity importances are biased toward high-cardinality features, so we "
        "confirm the ranking with **permutation importance** — how much held-out "
        "ROC-AUC drops when each feature is shuffled. This is model-agnostic and "
        "measured on data the model never trained on."
    ),
    code(
        "from sklearn.inspection import permutation_importance\n"
        "perm = permutation_importance(\n"
        "    best, X_te, y_te, n_repeats=10, random_state=0, scoring='roc_auc')\n"
        "perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values()\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "perm_imp.tail(10).plot.barh(\n"
        "    ax=ax, color='#8172b3',\n"
        "    xerr=perm.importances_std[perm_imp.tail(10).index.map(list(X.columns).index)])\n"
        "ax.set_title('Top 10 permutation importances (ROC-AUC drop)')\n"
        "ax.set_xlabel('mean AUC decrease when shuffled')\n"
        "plt.tight_layout(); plt.show()\n"
        "perm_imp.sort_values(ascending=False).head(6).round(4)"
    ),
    md(
        "**The text signal survives a stricter test.** Permutation importance "
        "again puts **`emotion_polarity` and `sentiment` on top** (AUC drops of "
        "~0.011 and ~0.008 when shuffled). It also re-ranks a couple of structured "
        "features upward relative to impurity importance — `performance_score` and "
        "`salary_peer_zscore` now appear in the top six — a useful reminder that "
        "the two methods disagree at the margins and that **peer-relative pay** "
        "carries more genuine signal than raw salary did in the EDA. The headline "
        "is unchanged: text features are the backbone of the model."
    ),
    md(
        '<a id="takeaways"></a>\n'
        "## Takeaways\n\n"
        "1. Models cluster around **0.70 CV AUC / 0.74 hold-out**; logistic "
        "regression is competitive with the trees.\n"
        "2. **Precision@20 ≈ 0.70** and a **~1.7x top-decile lift** make the model "
        "useful for targeted outreach despite the modest global AUC.\n"
        "3. The **F1-optimal threshold is ~0.35**, not 0.5 — tune the operating "
        "point to the cost of a missed departure.\n"
        "4. **Calibration did not help** on this sample size — verify, don't "
        "assume.\n"
        "5. **Text features dominate** both impurity and permutation importance, "
        "confirming the EDA thesis."
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
        "both at the population level and for individuals.\n\n"
        "### Contents\n"
        "1. [Group fairness audit](#audit)\n"
        "2. [Four-fifths-rule verdict](#verdict)\n"
        "3. [Mitigation: group-aware thresholds](#mitigation)\n"
        "4. [Global explanations (SHAP)](#global)\n"
        "5. [Individual explanations](#individual)\n"
        "6. [Takeaways](#takeaways)"
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
        '<a id="audit"></a>\n'
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
        '<a id="verdict"></a>\n'
        "## The model fails the four-fifths rule\n\n"
        "This is the most important finding in the whole analysis. The "
        "**disparate-impact ratio is ≈ 0.64**, well below the **0.80 four-fifths "
        "threshold** — so the model `passes_four_fifths = False`. Concretely, the "
        "least-flagged group (nonbinary) is selected at only ~64% the rate of the "
        "most-flagged group (female), and the equal-opportunity gap (difference in "
        "recall across groups) is ~0.18.\n\n"
        "Recall from the EDA that the *raw* gender association with churn was **not "
        "even statistically significant** (chi-square p ≈ 0.21). The model has "
        "**manufactured** a materially disparate selection rate from a signal that "
        "is not reliably in the data — almost certainly by leaning on features that "
        "correlate with gender. Deploying this as-is could systematically under- or "
        "over-target retention resources by gender."
    ),
    md(
        '<a id="mitigation"></a>\n'
        "## Mitigation: group-aware thresholds\n\n"
        "One of the simplest, most transparent post-processing fixes is to choose a "
        "**separate decision threshold per group** so that every group is flagged "
        "at the same rate (demographic-parity post-processing). We target the "
        "overall selection rate and re-audit."
    ),
    code(
        "target = (best.predict_proba(X_te)[:, 1] >= 0.5).mean()\n"
        "proba_te = best.predict_proba(X_te)[:, 1]\n"
        "adj = np.zeros(len(y_te), dtype=int)\n"
        "thresholds = {}\n"
        "for g in np.unique(sensitive.values):\n"
        "    m = sensitive.values == g\n"
        "    thr = np.quantile(proba_te[m], 1 - target)\n"
        "    thresholds[g] = round(float(thr), 3)\n"
        "    adj[m] = (proba_te[m] >= thr).astype(int)\n"
        "print('per-group thresholds:', thresholds)\n\n"
        "adj_report = group_fairness_report(y_te.values, adj, sensitive.values)\n"
        "before = fairness_summary(report)\n"
        "after = fairness_summary(adj_report)\n"
        "compare = pd.DataFrame({\n"
        "    'before (single 0.5)': [before['disparate_impact_ratio'],\n"
        "                            before['equal_opportunity_difference'],\n"
        "                            before['passes_four_fifths']],\n"
        "    'after (group thresholds)': [after['disparate_impact_ratio'],\n"
        "                                 after['equal_opportunity_difference'],\n"
        "                                 after['passes_four_fifths']],\n"
        "}, index=['disparate_impact_ratio', 'equal_opportunity_difference',\n"
        "          'passes_four_fifths'])\n"
        "display(compare)\n\n"
        "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
        "x = np.arange(len(adj_report))\n"
        "ax.bar(x - 0.2, report.set_index('group').loc[adj_report['group'], 'selection_rate'],\n"
        "       0.4, label='before', color='#c44e52')\n"
        "ax.bar(x + 0.2, adj_report['selection_rate'], 0.4, label='after', color='#55a868')\n"
        "ax.set_xticks(x); ax.set_xticklabels(adj_report['group'])\n"
        "ax.set_title('Selection rate by gender — before vs after mitigation')\n"
        "ax.set_ylabel('selection rate'); ax.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**The fix works — with a caveat.** Group-aware thresholds (female ≈ 0.59, "
        "male ≈ 0.53, nonbinary ≈ 0.47) equalize selection rates near the ~28% "
        "target, lifting the disparate-impact ratio from **0.64 → ~0.97 (now "
        "passes)** and collapsing the equal-opportunity gap from **0.18 → ~0.02**. "
        "The caveat is ethical and legal, not technical: applying different "
        "thresholds by a protected attribute is itself a regulated decision and "
        "may not be permissible in every jurisdiction. Treat this as one tool among "
        "several (alongside feature auditing and reweighing) and involve legal/HR "
        "stakeholders before adopting it."
    ),
    md('<a id="global"></a>'),
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
        '<a id="individual"></a>\n'
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
        '<a id="takeaways"></a>\n'
        "## Takeaways\n\n"
        "1. **The model fails the four-fifths rule (DI ≈ 0.64)** and must be "
        "mitigated before any deployment.\n"
        "2. Gender was **not** a significant churn driver in the data, yet the "
        "model manufactured a disparity — audit fairness on every retrain.\n"
        "3. **Group-aware thresholds restore parity** (DI 0.64 → ~0.97), but using "
        "a protected attribute in the decision is itself legally sensitive.\n"
        "4. SHAP confirms **text features dominate** and enables per-employee, "
        "actionable explanations.\n"
        "5. Pair every score with its explanation and a human decision-maker."
    ),
]


# --------------------------------------------------------------------------- #
# Notebook 05 — Business Impact & Retention ROI
# --------------------------------------------------------------------------- #
BUSINESS_SETUP = (
    "from employee_churn.data import make_synthetic_employee_data\n"
    "from employee_churn.features.engineer_structured import (\n"
    "    add_career_progression_features, add_tenure_bands,\n"
    "    add_promotion_velocity, add_compensation_features, add_team_metrics,\n"
    ")\n"
    "from employee_churn.features.engineer_text import add_text_statistics\n"
    "from employee_churn.nlp.sentiment import add_sentiment_scores\n"
    "from employee_churn.nlp.emotion import add_emotion_features\n"
    "from employee_churn.models.train import build_model_zoo, tune_hyperparameters\n"
    "from sklearn.model_selection import train_test_split\n\n"
    "df = make_synthetic_employee_data(n=1500, seed=42)\n"
    "f = add_career_progression_features(df, 'hire_date', 'last_promotion_date')\n"
    "f = add_tenure_bands(f)\n"
    "f = add_promotion_velocity(f, 'num_promotions')\n"
    "f = add_compensation_features(f, 'monthly_salary', 'department')\n"
    "f = add_team_metrics(f, 'team_id')\n"
    "f = add_sentiment_scores(f, 'feedback')\n"
    "f = add_emotion_features(f, 'feedback')\n"
    "f = add_text_statistics(f, 'feedback')\n"
    "DROP = ['employee_id', 'churned', 'feedback', 'gender', 'department',\n"
    "        'hire_date', 'last_promotion_date', 'team_id', 'tenure_band',\n"
    "        'emotion_dominant']\n"
    "X = f.drop(columns=DROP).select_dtypes(include=['number', 'bool'])\n"
    "y = f['churned']\n"
    "idx = np.arange(len(f))\n"
    "X_tr, X_te, y_tr, y_te, _, i_te = train_test_split(\n"
    "    X, y, idx, test_size=0.25, random_state=0, stratify=y)\n"
    "best, _, _ = tune_hyperparameters(\n"
    "    build_model_zoo(0)['random_forest'], X_tr, y_tr,\n"
    "    model_name='random_forest', n_iter=5)\n"
    "best.fit(X_tr, y_tr)\n"
    "proba = best.predict_proba(X_te)[:, 1]\n"
    "y_true = y_te.values\n"
    "base_rate = y_true.mean()\n"
    "print('held-out employees:', len(y_te), '| churn rate:', round(base_rate, 3))"
)

BUSINESS = [
    md(
        "# 05 · Business Impact & Retention ROI\n\n"
        "A model is only worth deploying if acting on it **saves more than it "
        "costs**. This notebook translates the churn scores from notebook 03 into "
        "dollars: how much a targeted retention program saves, how many employees "
        "to contact, and how sensitive the answer is to the (uncertain) business "
        "assumptions.\n\n"
        "### Contents\n"
        "1. [Cost model & assumptions](#assumptions)\n"
        "2. [Expected-value framework & break-even](#breakeven)\n"
        "3. [How many to contact? Net-savings curve](#curve)\n"
        "4. [The realistic case: capacity-constrained targeting](#capacity)\n"
        "5. [Sensitivity analysis](#sensitivity)\n"
        "6. [Recommendations](#recommendations)"
    ),
    code(BOOTSTRAP),
    code(BUSINESS_SETUP),
    md(
        '<a id="assumptions"></a>\n'
        "## 1 · Cost model & assumptions\n\n"
        "We attach a dollar value to each decision with three transparent "
        "assumptions (every organization should plug in its own):\n\n"
        "| Quantity | Assumption | Rationale |\n"
        "|---|---|---|\n"
        "| **Replacement cost** | 50% of annual salary | Industry estimates span "
        "0.5–2x salary (recruiting, onboarding, lost productivity); we take the "
        "conservative end. |\n"
        "| **Intervention cost** | \\$4,000 per employee | Manager time, a "
        "retention package or development budget. |\n"
        "| **Effectiveness** | 25% | Share of would-be churners actually retained "
        "by the intervention. |\n\n"
        "Replacement cost is **salary-linked per employee**, so losing a "
        "higher-paid person is correctly treated as more expensive."
    ),
    code(
        "REPLACEMENT_FRAC = 0.5        # of annual salary\n"
        "INTERVENTION_COST = 4_000.0   # $ per contacted employee\n"
        "EFFECTIVENESS = 0.25          # P(retain | would-be churner, contacted)\n\n"
        "annual_salary = f.iloc[i_te]['monthly_salary'].values * 12\n"
        "replacement_cost = REPLACEMENT_FRAC * annual_salary\n"
        "print(f'avg annual salary    : ${annual_salary.mean():,.0f}')\n"
        "print(f'avg replacement cost : ${replacement_cost.mean():,.0f}')\n"
        "print(f'intervention cost    : ${INTERVENTION_COST:,.0f}')\n"
        "print(f'effectiveness        : {EFFECTIVENESS:.0%}')"
    ),
    md(
        '<a id="breakeven"></a>\n'
        "## 2 · Expected-value framework\n\n"
        "Contacting an employee with churn probability *p* has expected value\n\n"
        "$$ \\mathbb{E}[\\text{value}] = p \\cdot \\text{effectiveness} \\cdot "
        "\\text{replacement\\_cost} - \\text{intervention\\_cost}. $$\n\n"
        "Setting this to zero gives a **break-even churn probability** below which "
        "intervening loses money on average."
    ),
    code(
        "p_star = INTERVENTION_COST / (EFFECTIVENESS * replacement_cost.mean())\n"
        "print(f'break-even churn probability p* = {p_star:.3f}')\n"
        "print(f'employees scored at or above p*: {(proba >= p_star).sum()} of {len(proba)}')"
    ),
    md(
        "**Break-even sits at p\\* ≈ 0.40.** With the average replacement cost "
        "(~\\$40k), an intervention pays for itself once a person's churn "
        "probability clears ~40%. About **250 of the 375 held-out employees** "
        "score above that line — already a hint that a *broad* program is "
        "justified here, but the ranking tells us whom to prioritize when capacity "
        "is limited."
    ),
    md(
        '<a id="curve"></a>\n'
        "## 3 · How many to contact?\n\n"
        "Walking down the risk ranking, we accumulate the **realized** net savings "
        "(back-tested against the true churn labels): each contacted employee who "
        "*actually* churned returns `effectiveness × replacement_cost`; everyone "
        "contacted costs `intervention_cost`."
    ),
    code(
        "order = np.argsort(-proba)\n"
        "benefit = np.where(y_true[order] == 1, EFFECTIVENESS * replacement_cost[order], 0.0)\n"
        "net_model = np.cumsum(benefit - INTERVENTION_COST)\n"
        "k = np.arange(1, len(y_true) + 1)\n"
        "exp_random = k * (base_rate * EFFECTIVENESS * replacement_cost.mean()\n"
        "                  - INTERVENTION_COST)\n"
        "k_star = int(np.argmax(net_model)) + 1\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "ax.plot(k, net_model / 1000, color='#c44e52', lw=2, label='model-ranked')\n"
        "ax.plot(k, exp_random / 1000, color='#4c72b0', lw=2, ls='--', label='random order')\n"
        "ax.axvline(k_star, color='k', lw=1, ls=':', label=f'optimal k = {k_star}')\n"
        "ax.axhline(net_model[-1] / 1000, color='gray', lw=1, ls='--',\n"
        "           label=f'contact everyone = ${net_model[-1]/1000:,.0f}k')\n"
        "ax.set_xlabel('employees contacted (high to low risk)')\n"
        "ax.set_ylabel('cumulative net savings ($k)')\n"
        "ax.set_title('Net savings vs number contacted'); ax.legend(fontsize=8)\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'optimal: contact top {k_star} -> net ${net_model[k_star-1]:,.0f}')"
    ),
    md(
        "**Two lessons from the curve.** First, the model-ranked curve rises far "
        "faster than the random-order line — front-loading true churners is exactly "
        "what a good ranking does. Second, because the intervention here is cheap "
        "relative to the ~\\$40k replacement cost, the *unconstrained* optimum is "
        "broad: net savings peak at **~\\$520k around the top ~250 employees**, and "
        "even contacting everyone stays net-positive. In other words, with these "
        "assumptions the binding question is not *whether* to act but *whom to "
        "prioritize first* — which is where a finite budget comes in."
    ),
    md(
        '<a id="capacity"></a>\n'
        "## 4 · The realistic case: limited capacity\n\n"
        "HR rarely has the bandwidth to run a meaningful intervention for hundreds "
        "of people at once. Suppose the budget covers the **top 20%** (75 of 375). "
        "Where you spend that fixed budget is precisely what the model decides."
    ),
    code(
        "K = int(0.20 * len(y_true))\n"
        "model_net = net_model[K - 1]\n"
        "random_net = K * (base_rate * EFFECTIVENESS * replacement_cost.mean()\n"
        "                  - INTERVENTION_COST)\n"
        "blanket_net = net_model[-1]\n"
        "gross = benefit[:K].sum()\n"
        "roi = gross / (K * INTERVENTION_COST)\n\n"
        "strategies = pd.Series({\n"
        "    'model — top 20%': model_net,\n"
        "    'random — 20%': random_net,\n"
        "    'blanket — all 100%': blanket_net,\n"
        "})\n"
        "fig, ax = plt.subplots(figsize=(7, 3.6))\n"
        "(strategies / 1000).plot.bar(\n"
        "    ax=ax, color=['#55a868', '#4c72b0', '#dd8452'])\n"
        "ax.set_ylabel('net savings ($k)'); ax.set_title('Strategy comparison')\n"
        "ax.tick_params(axis='x', rotation=15)\n"
        "plt.tight_layout(); plt.show()\n"
        "print(f'model top-20%  net = ${model_net:,.0f}  (ROI {roi:.2f}x on spend)')\n"
        "print(f'random 20%     net = ${random_net:,.0f}')\n"
        "print(f'blanket 100%   net = ${blanket_net:,.0f}')\n"
        "print(f'model vs random uplift = ${model_net - random_net:,.0f} '\n"
        "      f'({model_net / max(random_net, 1):.1f}x)')"
    ),
    md(
        "**This is the model's dollar value.** Given a fixed budget for 75 "
        "interventions, **model targeting nets ~\\$276k versus ~\\$42k for random "
        "selection — a 6.6x uplift** — and every \\$1 spent returns ~\\$1.9 in "
        "avoided replacement cost. Strikingly, the model's top-20% program even "
        "**beats blanket outreach to all 375 employees (~\\$234k) while contacting "
        "one-fifth as many people** and burning one-fifth of the manager time. "
        "Targeting is not just cheaper — under realistic capacity limits it is the "
        "*better-performing* strategy."
    ),
    md(
        '<a id="sensitivity"></a>\n'
        "## 5 · Sensitivity analysis\n\n"
        "The dollar figures hinge on two genuinely uncertain inputs — intervention "
        "effectiveness and cost. We recompute the optimal net savings across a grid "
        "to see where the program stops paying off."
    ),
    code(
        "effects = [0.15, 0.25, 0.35]\n"
        "costs = [2_000, 4_000, 8_000]\n"
        "grid = np.zeros((len(effects), len(costs)))\n"
        "for r, e in enumerate(effects):\n"
        "    for c, ic in enumerate(costs):\n"
        "        ben = np.where(y_true[order] == 1, e * replacement_cost[order], 0.0)\n"
        "        grid[r, c] = np.cumsum(ben - ic).max()\n\n"
        "fig, ax = plt.subplots(figsize=(6.5, 4))\n"
        "sns.heatmap(grid / 1000, annot=True, fmt=',.0f', cmap='RdYlGn', center=0,\n"
        "            xticklabels=[f'${c//1000}k' for c in costs],\n"
        "            yticklabels=[f'{int(e*100)}%' for e in effects], ax=ax,\n"
        "            cbar_kws={'label': 'optimal net savings ($k)'})\n"
        "ax.set_xlabel('intervention cost'); ax.set_ylabel('effectiveness')\n"
        "ax.set_title('Optimal net savings under different assumptions')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**The program is robust except in the worst corner.** Across most of the "
        "grid the optimal policy still saves six figures, but the value swings "
        "widely: from **~\\$1.7M** (35% effectiveness, \\$2k intervention) down to "
        "**roughly break-even or negative** when effectiveness is low (15%) and "
        "interventions are expensive (\\$8k). The practical implication is that "
        "**measuring intervention effectiveness is as important as improving the "
        "model** — a small, well-run pilot to estimate that 15–35% number would "
        "de-risk the whole program."
    ),
    md(
        '<a id="recommendations"></a>\n'
        "## 6 · Recommendations\n\n"
        "1. **Deploy as a ranked worklist, not a blanket alarm.** Under realistic "
        "capacity limits, model-targeted outreach delivers a ~6.6x uplift over "
        "random and even beats contacting everyone.\n"
        "2. **Set the cutoff from economics, not 0.5.** Intervene above the "
        "break-even probability (~0.40 here), adjusted for available budget.\n"
        "3. **Instrument effectiveness.** The ROI is most sensitive to how often "
        "interventions actually work — run a holdout/pilot to measure it.\n"
        "4. **Pair with the fairness controls from notebook 04** so targeting "
        "savings are not achieved through disparate treatment.\n"
        "5. **Revisit the cost assumptions per role/region** — replacement cost is "
        "salary-linked, so the ranking and the budget should be too."
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
        ("05_business_impact_and_roi.ipynb", BUSINESS),
    ]:
        print("wrote", build(name, cells))


if __name__ == "__main__":
    main()
