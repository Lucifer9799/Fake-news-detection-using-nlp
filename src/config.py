"""Central configuration: paths, constants, and hyper-parameters.

Keeping every setting here means training, prediction, tests and the
Streamlit app all agree with each other (no "magic numbers" scattered around).
"""
from pathlib import Path

# ---- Paths (all relative to the project root, wherever it is on your PC) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "fake_or_real_news.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "logreg_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
TEST_IDS_PATH = REPORTS_DIR / "test_ids.csv"

# ---- Dataset ----
DATASET_URL = (
    "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/"
    "fake_or_real_news.csv"
)
LABEL_TO_INT = {"REAL": 0, "FAKE": 1}      # FAKE is the "positive" class
INT_TO_LABEL = {0: "REAL", 1: "FAKE"}

# ---- Reproducibility / splitting ----
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---- TF-IDF settings ----
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),   # single words and word pairs
    min_df=3,             # ignore terms in fewer than 3 documents
    max_df=0.90,          # ignore terms in more than 90% of documents
    max_features=30000,
    sublinear_tf=True,    # dampen very frequent terms
)

# ---- Sentiment ----
SENTIMENT_COLUMNS = ["vader_compound", "vader_pos", "vader_neu", "vader_neg"]
POSITIVE_THRESHOLD = 0.05    # standard VADER cut-offs
NEGATIVE_THRESHOLD = -0.05

# ---- Input validation (used by the predictor / app) ----
MIN_WORDS_HARD = 3       # fewer words than this -> rejected
MIN_WORDS_WARN = 30      # fewer words than this -> allowed, with a warning
LOW_CONFIDENCE = 0.60    # below this -> "low confidence" warning
MAX_CHARS = 200_000      # protects the app from absurdly large pastes
