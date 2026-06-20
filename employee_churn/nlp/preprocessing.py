"""Text preprocessing: cleaning, tokenization, and normalization.

A dependency-free, deterministic preprocessing pipeline for the short free-text
feedback this project works with. It is intentionally lightweight (regex + a
small stopword set) so it runs offline with no model downloads; pass a custom
stopword set or token pattern to adapt it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional

import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Small, general-purpose English stopword set. Deliberately compact; supply your
# own via the ``stopwords`` argument for domain-specific filtering.
DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "without",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "about",
        "into",
        "over",
        "than",
        "too",
        "very",
        "just",
        "also",
    }
)


def clean_text(
    text: str,
    lowercase: bool = True,
    strip_accents: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = False,
    remove_punctuation: bool = True,
) -> str:
    """Normalize a single piece of text.

    Args:
        text: Input string (non-strings are coerced via ``str``).
        lowercase: Lowercase the output.
        strip_accents: Fold accented characters to ASCII (e.g. ``é`` -> ``e``).
        remove_urls: Strip URLs.
        remove_emails: Strip email addresses.
        remove_numbers: Strip digit sequences.
        remove_punctuation: Keep only word characters and whitespace.

    Returns:
        The cleaned, whitespace-normalized string.
    """
    result = "" if text is None else str(text)
    if remove_urls:
        result = _URL_RE.sub(" ", result)
    if remove_emails:
        result = _EMAIL_RE.sub(" ", result)
    if strip_accents:
        result = (
            unicodedata.normalize("NFKD", result)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    if lowercase:
        result = result.lower()
    if remove_numbers:
        result = re.sub(r"\d+", " ", result)
    if remove_punctuation:
        result = re.sub(r"[^\w\s]", " ", result)
    return _WHITESPACE_RE.sub(" ", result).strip()


def tokenize(text: str, pattern: re.Pattern[str] = _TOKEN_RE) -> List[str]:
    """Tokenize text into word tokens using a regex pattern.

    Args:
        text: Input string.
        pattern: Compiled token pattern (defaults to alphabetic words with
            optional internal apostrophes).

    Returns:
        List of token strings in order of appearance.
    """
    return pattern.findall("" if text is None else str(text))


def remove_stopwords(
    tokens: Iterable[str], stopwords: Optional[Iterable[str]] = None
) -> List[str]:
    """Drop stopwords from a token sequence (case-insensitive).

    Args:
        tokens: Iterable of tokens.
        stopwords: Stopword collection. Defaults to :data:`DEFAULT_STOPWORDS`.

    Returns:
        Filtered list of tokens.
    """
    stops = DEFAULT_STOPWORDS if stopwords is None else frozenset(stopwords)
    return [tok for tok in tokens if tok.lower() not in stops]


def preprocess(
    text: str,
    stopwords: Optional[Iterable[str]] = None,
    drop_stopwords: bool = True,
    **clean_kwargs: bool,
) -> List[str]:
    """Run the full clean -> tokenize -> (optional) stopword-removal pipeline.

    Args:
        text: Input string.
        stopwords: Optional custom stopword set.
        drop_stopwords: Whether to remove stopwords.
        **clean_kwargs: Forwarded to :func:`clean_text`.

    Returns:
        List of preprocessed tokens.
    """
    tokens = tokenize(clean_text(text, **clean_kwargs))
    if drop_stopwords:
        tokens = remove_stopwords(tokens, stopwords)
    return tokens


def add_clean_text(
    df: pd.DataFrame,
    text_column: str,
    output_column: str = "clean_text",
    **clean_kwargs: bool,
) -> pd.DataFrame:
    """Add a cleaned-text column to a DataFrame.

    Args:
        df: DataFrame containing a text column.
        text_column: Name of the source text column.
        output_column: Name of the cleaned-text column to add.
        **clean_kwargs: Forwarded to :func:`clean_text`.

    Returns:
        DataFrame with the added ``output_column``.
    """
    out = df.copy()
    out[output_column] = out[text_column].map(lambda t: clean_text(t, **clean_kwargs))
    return out
