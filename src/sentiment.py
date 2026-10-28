"""VADER sentiment analysis (rule-based, needs no training).

IMPORTANT: sentiment describes the TONE of the writing (positive/negative
emotion). It is NOT evidence that a story is true or false - accurate
reporting can be negative (e.g. a disaster) and fabricated stories can be
cheerful. In this project sentiment is reported separately and is also tested
as an extra feature to see whether it helps at all.
"""
from typing import Dict, Iterable

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.config import (NEGATIVE_THRESHOLD, POSITIVE_THRESHOLD,
                        SENTIMENT_COLUMNS)
from src.preprocessing import ensure_nltk_resources

_ANALYZER = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        ensure_nltk_resources()
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def label_from_compound(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze_sentiment(text) -> Dict[str, float]:
    """Return label, compound score (-1..1) and pos/neu/neg proportions."""
    if not isinstance(text, str) or not text.strip():
        return {"label": "Neutral", "compound": 0.0,
                "pos": 0.0, "neu": 1.0, "neg": 0.0}
    s = _get_analyzer().polarity_scores(text)
    return {"label": label_from_compound(s["compound"]),
            "compound": s["compound"], "pos": s["pos"],
            "neu": s["neu"], "neg": s["neg"]}


def sentiment_features(texts: Iterable[str]) -> pd.DataFrame:
    """Sentiment scores for many texts as a DataFrame (used in training)."""
    rows = []
    for t in texts:
        s = analyze_sentiment(t)
        rows.append([s["compound"], s["pos"], s["neu"], s["neg"]])
    return pd.DataFrame(rows, columns=SENTIMENT_COLUMNS)
