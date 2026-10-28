"""Text preprocessing shared by TRAINING and PREDICTION.

Both phases import `preprocess_text` from this one file, so they can never
drift apart (a classic source of silent bugs).

Choices (and why):
  * lowercase                 -> "Trump" and "trump" become one feature
  * remove URLs / HTML tags   -> noise, not language
  * tokenise with a regex     -> simple, no extra downloads, keeps digits
  * keep digits               -> numbers/years can carry information
  * remove stop-words         -> common words ("the", "and") add little signal,
                                 BUT negations (not, no, n't words) are KEPT
                                 because they flip meaning
  * NO stemming/lemmatising   -> keeps the pipeline simple and interpretable;
                                 TF-IDF with bigrams already works well
  * drop 1-character tokens   -> mostly leftovers like "s" or "t"

Sentiment is NOT computed from this cleaned text: VADER relies on
capitalisation and punctuation ("GREAT!!!"), which we strip here.
"""
import re
from functools import lru_cache
from typing import List

import nltk
from nltk.tokenize import RegexpTokenizer

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
POSSESSIVE_RE = re.compile(r"'s\b")
_TOKENIZER = RegexpTokenizer(r"[a-z0-9]+(?:'[a-z]+)?")


def ensure_nltk_resources() -> None:
    """Download required NLTK data once (no-op if already present)."""
    for resource, path in [("stopwords", "corpora/stopwords"),
                           ("vader_lexicon", "sentiment/vader_lexicon.zip")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    ensure_nltk_resources()
    from nltk.corpus import stopwords
    sw = set(stopwords.words("english"))
    negations = {"no", "nor", "not"} | {w for w in sw if w.endswith("n't")}
    return frozenset(sw - negations)


def tokenize(text: str) -> List[str]:
    """Lowercase, strip URLs/HTML/possessives, and split into tokens."""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = POSSESSIVE_RE.sub("", text)
    return _TOKENIZER.tokenize(text)


def preprocess_text(text) -> str:
    """Full cleaning: returns a space-joined string of useful tokens.

    Missing / non-string input returns "" instead of crashing.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    sw = get_stopwords()
    tokens = [t for t in tokenize(text) if len(t) > 1 and t not in sw]
    return " ".join(tokens)
