"""Synthetic employee dataset generation.

Produces a realistic, fully reproducible HR + feedback dataset with a known
churn signal. It is used by the example notebooks, documentation snippets, and
the test suite so that every workflow can run end-to-end without requiring
access to sensitive real employee records.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEPARTMENTS = ("engineering", "sales", "support", "marketing", "operations")
GENDERS = ("female", "male", "nonbinary")

# Feedback snippets keyed by sentiment so the generated text carries a signal
# that is correlated with churn (negative feedback -> higher churn risk).
_POSITIVE_FEEDBACK = (
    "I feel happy and supported by my manager and proud of our work.",
    "Great team, I trust leadership and feel confident about my growth.",
    "Delighted with the new projects, plenty of learning opportunities.",
    "Pleased with the recognition I received this quarter.",
)
_NEUTRAL_FEEDBACK = (
    "The quarter was fine, nothing remarkable to report.",
    "Workload was steady and the tooling is acceptable.",
    "Meetings could be shorter but overall it was an average period.",
    "No strong feelings either way about the recent changes.",
)
_NEGATIVE_FEEDBACK = (
    "I am frustrated and afraid about the constant reorgs and unclear goals.",
    "Annoyed by the lack of recognition, I feel undervalued and tired.",
    "Scared about layoffs, morale is low and I am angry about the workload.",
    "Disappointed and exhausted, I have lost trust in the leadership team.",
)


def _pick(rng: np.random.Generator, options, size: int) -> np.ndarray:
    """Sample ``size`` values uniformly from ``options``."""
    return rng.choice(np.array(options, dtype=object), size=size)


def make_synthetic_employee_data(
    n: int = 500,
    seed: int = 42,
    reference_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate a synthetic employee dataset with a latent churn signal.

    The churn label is produced from a logistic function of several drivers so
    that downstream models have a learnable (but noisy) target:

    * shorter tenure, longer time since promotion -> higher risk
    * lower satisfaction and performance -> higher risk
    * negative written feedback -> higher risk
    * higher overtime hours -> higher risk

    Args:
        n: Number of employees to generate.
        seed: Seed controlling all randomness for full reproducibility.
        reference_date: Snapshot date used for tenure/date columns.

    Returns:
        A DataFrame with structured HR columns, a free-text ``feedback`` column,
        and a binary ``churned`` target.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    ref = pd.Timestamp(reference_date)

    age = rng.integers(22, 60, size=n)
    tenure_years = np.clip(rng.gamma(shape=2.0, scale=2.0, size=n), 0.1, 30)
    hire_date = ref - pd.to_timedelta((tenure_years * 365).astype(int), unit="D")

    # Time since promotion is bounded by tenure.
    months_since_promo = np.minimum(
        rng.integers(0, 48, size=n), (tenure_years * 12).astype(int)
    )
    last_promotion_date = ref - pd.to_timedelta(
        (months_since_promo * 30).astype(int), unit="D"
    )

    department = _pick(rng, DEPARTMENTS, n)
    gender = _pick(rng, GENDERS, n)
    team_id = rng.integers(1, 1 + max(2, n // 25), size=n)

    satisfaction = np.clip(rng.normal(0.65, 0.18, size=n), 0.0, 1.0)
    performance = np.clip(rng.normal(0.7, 0.15, size=n), 0.0, 1.0)
    overtime_hours = np.clip(rng.normal(6, 4, size=n), 0, 40)
    num_promotions = rng.poisson(np.clip(tenure_years / 3, 0, None)).astype(int)
    monthly_salary = np.round(
        rng.normal(5500, 1500, size=n) + performance * 2000, 2
    ).clip(2500, None)

    # Latent risk drives both the written feedback tone and the churn label.
    logit = (
        1.4
        - 0.9 * satisfaction * 2
        - 0.7 * performance * 2
        + 0.6 * (months_since_promo / 12)
        - 0.5 * np.log1p(tenure_years)
        + 0.04 * overtime_hours
        + rng.normal(0, 0.5, size=n)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.random(n) < churn_prob).astype(int)

    feedback = np.empty(n, dtype=object)
    for i in range(n):
        if churn_prob[i] > 0.6:
            pool = _NEGATIVE_FEEDBACK
        elif churn_prob[i] < 0.35:
            pool = _POSITIVE_FEEDBACK
        else:
            pool = _NEUTRAL_FEEDBACK
        feedback[i] = pool[rng.integers(0, len(pool))]

    frame = pd.DataFrame(
        {
            "employee_id": np.arange(1, n + 1),
            "age": age,
            "gender": gender,
            "department": department,
            "team_id": team_id,
            "hire_date": hire_date.normalize(),
            "last_promotion_date": last_promotion_date.normalize(),
            "num_promotions": num_promotions,
            "monthly_salary": monthly_salary,
            "satisfaction_score": np.round(satisfaction, 3),
            "performance_score": np.round(performance, 3),
            "overtime_hours": np.round(overtime_hours, 1),
            "feedback": feedback,
            "churned": churned,
        }
    )
    return frame
