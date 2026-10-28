"""Test cases defined ONCE, used by both pytest and the report generator.

Each case has an `expected` description (written BEFORE running) and an
`observe()` function that actually runs the code. `check()` decides pass/fail.
"""
import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src import config as C
from src.data_loader import clean_dataset, load_raw
from src.predict import (FakeNewsPredictor, InvalidInputError,
                         ModelNotFoundError)
from src.preprocessing import preprocess_text
from src.sentiment import analyze_sentiment

_predictor = None


def predictor() -> FakeNewsPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FakeNewsPredictor()
    return _predictor


def attempt(fn: Callable[[], Any]) -> dict:
    """Run fn and capture either its value or the exception it raised."""
    try:
        return {"ok": True, "value": fn()}
    except Exception as exc:  # noqa: BLE001 - we want to record any failure
        return {"ok": False, "error": type(exc).__name__, "msg": str(exc)}


@dataclass
class Case:
    id: str
    area: str
    description: str
    expected: str
    observe: Callable[[], dict]
    check: Callable[[dict], bool]
    show: Callable[[dict], str]


def _show_pred(o):
    if not o["ok"]:
        return f"{o['error']}: {o['msg']}"
    v = o["value"]
    return (f"{v['label']} ({v['confidence']:.1%}); sentiment {v['sentiment']['label']} "
            f"({v['sentiment']['compound']:+.3f}); warnings={len(v['warnings'])}")


def _raises(exc_name):
    return lambda o: (not o["ok"]) and o["error"] == exc_name


def _predicts(o):
    return o["ok"] and o["value"]["label"] in ("FAKE", "REAL")


NORMAL_ARTICLE = (
    "The city council voted on Tuesday to approve a new budget that increases "
    "funding for public transport and road repairs. Officials said the plan "
    "would be paid for through a small rise in local fees. Several residents "
    "spoke during the meeting, with some welcoming the investment and others "
    "questioning the cost. The mayor said the changes would take effect next "
    "year and promised a public report on spending every six months. "
) * 2

LONG_ARTICLE = NORMAL_ARTICLE * 40   # several thousand words


def _sent(text):
    return lambda: attempt(lambda: analyze_sentiment(text))


def _sent_check(label):
    return lambda o: o["ok"] and o["value"]["label"] == label


def _sent_show(o):
    return (f"{o['value']['label']} (compound {o['value']['compound']:+.3f})"
            if o["ok"] else f"{o['error']}: {o['msg']}")


def _dirty_df():
    return pd.DataFrame({
        "id": range(8),
        "title": ["A", None, "C", "D", "E", "F", "G", "H"],
        "text": ["good article one", "has missing title", None, "   ",
                 "duplicate body", "duplicate body", "conflict body", "conflict body"],
        "label": ["REAL", "FAKE", "REAL", "FAKE", "REAL", "REAL", "REAL", "FAKE"],
    })


def _clean_dirty():
    df, log = clean_dataset(_dirty_df())
    return {"rows": len(df), "log": log}


def _end_to_end():
    """Run the SAVED model over the whole held-out test set through the
    public predictor and compare with the accuracy recorded in metrics.json."""
    raw = load_raw()
    df, _ = clean_dataset(raw)
    ids = pd.read_csv(C.TEST_IDS_PATH)["id"]
    test = df[df["id"].isin(ids)]
    correct = total = 0
    for content, y in zip(test["content"], test["y"]):
        try:
            res = predictor().predict(content)
        except InvalidInputError:
            continue
        total += 1
        correct += int(C.LABEL_TO_INT[res["label"]] == y)
    saved = json.loads(C.METRICS_PATH.read_text())["results"][
        "Logistic Regression (TF-IDF)"]["accuracy"]
    return {"n": total, "accuracy": correct / total, "saved_accuracy": saved}


CASES = [
    Case("T01", "Empty input", "Empty string",
         "InvalidInputError raised (no crash, clear message)",
         lambda: attempt(lambda: predictor().predict("")),
         _raises("InvalidInputError"), _show_pred),
    Case("T02", "Empty input", "Whitespace only",
         "InvalidInputError raised",
         lambda: attempt(lambda: predictor().predict("   \n\t  ")),
         _raises("InvalidInputError"), _show_pred),
    Case("T03", "Invalid input", "None instead of text",
         "InvalidInputError raised",
         lambda: attempt(lambda: predictor().predict(None)),
         _raises("InvalidInputError"), _show_pred),
    Case("T04", "Invalid input", "Number instead of text (12345)",
         "InvalidInputError raised",
         lambda: attempt(lambda: predictor().predict(12345)),
         _raises("InvalidInputError"), _show_pred),
    Case("T05", "Invalid input", "Only symbols: '!!!! ???? ####'",
         "InvalidInputError raised (no meaningful words)",
         lambda: attempt(lambda: predictor().predict("!!!! ???? ####")),
         _raises("InvalidInputError"), _show_pred),
    Case("T06", "Very short text", "Two words: 'Breaking news'",
         f"InvalidInputError raised (fewer than {C.MIN_WORDS_HARD} words)",
         lambda: attempt(lambda: predictor().predict("Breaking news")),
         _raises("InvalidInputError"), _show_pred),
    Case("T07", "Very short text", "Headline-length text (9 words)",
         "A FAKE/REAL prediction is returned WITH a short-text warning",
         lambda: attempt(lambda: predictor().predict(
             "City council approves new budget for road repairs")),
         lambda o: _predicts(o) and len(o["value"]["warnings"]) >= 1, _show_pred),
    Case("T08", "Normal prediction", "Article-length paragraph (~90 words)",
         "Valid prediction; probabilities sum to 1; confidence in [0.5, 1]",
         lambda: attempt(lambda: predictor().predict(NORMAL_ARTICLE)),
         lambda o: _predicts(o)
         and abs(sum(o["value"]["probabilities"].values()) - 1) < 1e-6
         and 0.5 <= o["value"]["confidence"] <= 1.0, _show_pred),
    Case("T09", "Long article", "~7,000-word text",
         "Prediction returned without error",
         lambda: attempt(lambda: predictor().predict(LONG_ARTICLE)),
         _predicts, _show_pred),
    Case("T10", "Sentiment", "Clearly positive text",
         "Label Positive, compound >= 0.05", _sent(
             "I absolutely love this wonderful, amazing and fantastic result!"),
         _sent_check("Positive"), _sent_show),
    Case("T11", "Sentiment", "Clearly negative text",
         "Label Negative, compound <= -0.05", _sent(
             "This is a terrible, horrible and disgusting disaster."),
         _sent_check("Negative"), _sent_show),
    Case("T12", "Sentiment", "Neutral factual text",
         "Label Neutral", _sent("The committee will meet on Tuesday at noon."),
         _sent_check("Neutral"), _sent_show),
    Case("T13", "Sentiment", "Sentiment of empty text",
         "Neutral with compound 0.0 (no crash)", _sent(""),
         lambda o: o["ok"] and o["value"]["label"] == "Neutral"
         and o["value"]["compound"] == 0.0, _sent_show),
    Case("T14", "Missing dataset values",
         "8-row dirty DataFrame: missing title/text, blank text, duplicate, "
         "conflicting-label duplicate",
         "Rows kept = 3 ('good article one', 'has missing title', and ONE copy of "
         "'duplicate body'); missing/blank text and the conflicting-label pair removed",
         lambda: attempt(_clean_dirty),
         lambda o: o["ok"] and o["value"]["rows"] == 3,
         lambda o: (f"rows kept = {o['value']['rows']}; log = {o['value']['log']}"
                    if o["ok"] else f"{o['error']}: {o['msg']}")),
    Case("T15", "Missing dataset values", "DataFrame without a 'text' column",
         "ValueError raised naming the missing column",
         lambda: attempt(lambda: clean_dataset(pd.DataFrame(
             {"title": ["a"], "label": ["REAL"]}))),
         lambda o: (not o["ok"]) and o["error"] == "ValueError"
         and "text" in o["msg"],
         lambda o: f"{o.get('error')}: {o.get('msg')}" if not o["ok"] else "no error"),
    Case("T16", "Preprocessing", "URL/HTML removed, negation + year kept",
         "'not' and '2016' kept; 'http', 'b' and stop-word 'the' removed",
         lambda: attempt(lambda: preprocess_text(
             "The claim is <b>not</b> true, see https://example.com in 2016")),
         lambda o: o["ok"] and {"not", "2016"} <= set(o["value"].split())
         and not ({"the", "http", "https", "b"} & set(o["value"].split())),
         lambda o: repr(o["value"]) if o["ok"] else o["msg"]),
    Case("T17", "Preprocessing", "Missing/invalid values (None, NaN, 5)",
         "All three return '' (no crash)",
         lambda: attempt(lambda: [preprocess_text(None),
                                  preprocess_text(float("nan")),
                                  preprocess_text(5)]),
         lambda o: o["ok"] and o["value"] == ["", "", ""],
         lambda o: repr(o["value"]) if o["ok"] else o["msg"]),
    Case("T18", "Model loading", "Models directory does not exist",
         "ModelNotFoundError raised with instruction to train",
         lambda: attempt(lambda: FakeNewsPredictor("/nonexistent/folder")),
         _raises("ModelNotFoundError"),
         lambda o: f"{o.get('error')}: {o.get('msg','')[:60]}"),
    Case("T19", "End-to-end consistency",
         "Saved model + vectorizer run over the full held-out test set via "
         "the public predictor",
         "Accuracy equals the accuracy stored in reports/metrics.json "
         "(proves training and prediction preprocessing match)",
         lambda: attempt(_end_to_end),
         lambda o: o["ok"] and abs(o["value"]["accuracy"]
                                   - o["value"]["saved_accuracy"]) < 1e-9,
         lambda o: (f"n={o['value']['n']}, accuracy={o['value']['accuracy']:.4f}, "
                    f"saved={o['value']['saved_accuracy']:.4f}"
                    if o["ok"] else f"{o['error']}: {o['msg']}")),
]
