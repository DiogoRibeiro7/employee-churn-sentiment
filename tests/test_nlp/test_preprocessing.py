"""Tests for the text preprocessing pipeline."""

import pandas as pd

from employee_churn.nlp.preprocessing import (
    add_clean_text,
    clean_text,
    preprocess,
    remove_stopwords,
    tokenize,
)


def test_clean_text_removes_urls_emails_punct_and_lowercases() -> None:
    raw = "Visit https://x.com or mail me@x.com NOW!! Café déjà-vu."
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "!" not in cleaned
    assert cleaned == cleaned.lower()
    assert "cafe" in cleaned and "deja" in cleaned  # accents folded


def test_clean_text_optional_numbers() -> None:
    assert clean_text("room 101", remove_numbers=True) == "room"
    assert "101" in clean_text("room 101", remove_numbers=False)


def test_tokenize_keeps_apostrophes() -> None:
    assert tokenize("I can't go") == ["I", "can't", "go"]


def test_remove_stopwords_default_and_custom() -> None:
    tokens = ["the", "team", "is", "great"]
    assert remove_stopwords(tokens) == ["team", "great"]
    assert remove_stopwords(tokens, stopwords={"team"}) == ["the", "is", "great"]


def test_preprocess_pipeline() -> None:
    out = preprocess("The team is NOT happy!!")
    assert "team" in out and "happy" in out
    assert "the" not in out  # stopword removed


def test_add_clean_text_column() -> None:
    df = pd.DataFrame({"feedback": ["Great JOB!!!", "Bad, sad."]})
    out = add_clean_text(df, "feedback")
    assert list(out["clean_text"]) == ["great job", "bad sad"]
    assert "clean_text" not in df.columns  # input not mutated
