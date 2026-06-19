"""Tests for the richer structured feature helpers."""

import pandas as pd
import pytest

from employee_churn.features.engineer_structured import (
    add_compensation_features,
    add_promotion_velocity,
    add_tenure_bands,
)


def test_add_tenure_bands() -> None:
    df = pd.DataFrame({"tenure_days": [100, 600, 2000, 4000]})
    out = add_tenure_bands(df)
    assert list(out["tenure_band"]) == ["0-1y", "1-3y", "3-7y", "7y+"]
    assert out["tenure_band"].cat.ordered


def test_add_tenure_bands_label_mismatch_raises() -> None:
    df = pd.DataFrame({"tenure_days": [100]})
    with pytest.raises(ValueError):
        add_tenure_bands(df, labels=("only-one",))


def test_add_promotion_velocity() -> None:
    df = pd.DataFrame({"num_promotions": [2, 0], "tenure_days": [730, 0]})
    out = add_promotion_velocity(df, "num_promotions")
    assert out.loc[0, "promotions_per_year"] == pytest.approx(1.0)
    # Zero tenure avoids division blow-up.
    assert out.loc[1, "promotions_per_year"] == 0.0


def test_add_compensation_features() -> None:
    df = pd.DataFrame(
        {
            "department": ["eng", "eng", "eng"],
            "monthly_salary": [4000.0, 6000.0, 8000.0],
        }
    )
    out = add_compensation_features(df, "monthly_salary", "department")
    # Median is 6000, so the middle employee has ratio 1.0.
    assert out.loc[1, "salary_to_peer_ratio"] == pytest.approx(1.0)
    assert out.loc[0, "salary_to_peer_ratio"] < 1.0
    assert out.loc[2, "salary_to_peer_ratio"] > 1.0
    assert out.loc[1, "salary_peer_zscore"] == pytest.approx(0.0)


def test_compensation_single_member_group_no_nan() -> None:
    df = pd.DataFrame({"department": ["eng"], "monthly_salary": [5000.0]})
    out = add_compensation_features(df, "monthly_salary", "department")
    # std is undefined for a single member -> z-score falls back to 0.0.
    assert out.loc[0, "salary_peer_zscore"] == 0.0
