"""Prediction on new text using the SAVED vectorizer and model.

Nothing is retrained here. The same `preprocess_text` used in training is
applied, so training and prediction stay consistent.

Quick demo from a terminal:
    python -m src.predict "Paste a news article or headline here ..."
"""
import json
import sys
from pathlib import Path
from typing import Optional

import joblib

from src import config as C
from src.preprocessing import preprocess_text
from src.sentiment import analyze_sentiment


class InvalidInputError(ValueError):
    """The submitted text cannot be analysed (empty, wrong type, too short)."""


class ModelNotFoundError(FileNotFoundError):
    """Trained model files are missing - run `python -m src.train` first."""


def validate_text(text) -> str:
    """Return the stripped text or raise InvalidInputError with a clear reason."""
    if not isinstance(text, str):
        raise InvalidInputError("Input must be text (a string).")
    text = text.strip()
    if not text:
        raise InvalidInputError("Input is empty. Please paste a headline or article.")
    if len(text) > C.MAX_CHARS:
        raise InvalidInputError(
            f"Input is too long ({len(text):,} characters; limit {C.MAX_CHARS:,}).")
    if len(text.split()) < C.MIN_WORDS_HARD:
        raise InvalidInputError(
            f"Input is too short. Please enter at least {C.MIN_WORDS_HARD} words.")
    return text


class FakeNewsPredictor:
    def __init__(self, models_dir: Optional[Path] = None):
        models_dir = Path(models_dir) if models_dir else C.MODELS_DIR
        vec_path = models_dir / C.VECTORIZER_PATH.name
        model_path = models_dir / C.MODEL_PATH.name
        if not vec_path.exists() or not model_path.exists():
            raise ModelNotFoundError(
                f"Trained model files not found in {models_dir}.\n"
                "Run:  python -m src.train")
        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)

    def predict(self, text) -> dict:
        text = validate_text(text)
        clean = preprocess_text(text)
        if not clean:
            raise InvalidInputError(
                "The text has no meaningful words after cleaning "
                "(only symbols, numbers or common filler words).")

        features = self.vectorizer.transform([clean])
        proba = self.model.predict_proba(features)[0]          # [REAL, FAKE]
        probs = {C.INT_TO_LABEL[i]: float(p)
                 for i, p in zip(self.model.classes_, proba)}
        label = max(probs, key=probs.get)

        n_words = len(text.split())
        warnings = []
        if n_words < C.MIN_WORDS_WARN:
            warnings.append(
                f"Short text ({n_words} words). The model was trained on full "
                "articles (median ~600 words), so results for headlines or short "
                "snippets are much less reliable.")
        if probs[label] < C.LOW_CONFIDENCE:
            warnings.append("Low confidence: the model is close to undecided.")

        return {
            "label": label,
            "confidence": probs[label],
            "probabilities": probs,
            "sentiment": analyze_sentiment(text),   # on RAW text, not cleaned
            "word_count": n_words,
            "warnings": warnings,
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.predict "your text here"')
    try:
        print(json.dumps(FakeNewsPredictor().predict(" ".join(sys.argv[1:])), indent=2))
    except (InvalidInputError, ModelNotFoundError) as e:
        raise SystemExit(f"Error: {e}")
