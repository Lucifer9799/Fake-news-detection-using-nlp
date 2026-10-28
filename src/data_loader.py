"""Dataset loading, inspection and cleaning.

Dataset: "fake_or_real_news.csv" (6,335 articles; columns id, title, text, label).
Label assumption: the `label` column (FAKE / REAL) is taken as ground truth.
"""
from pathlib import Path

import pandas as pd

from src.config import LABEL_TO_INT, RAW_DATA_PATH

REQUIRED_COLUMNS = ["title", "text", "label"]


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the CSV with pandas, with a helpful error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run:  python -m src.download_data"
        )
    return pd.read_csv(path)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return basic facts about the data (size, missing, duplicates, balance)."""
    text = df["text"].fillna("") if "text" in df else pd.Series(dtype=str)
    words = text.str.split().str.len()
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "blank_text_rows": int((text.str.strip() == "").sum()),
        "duplicate_text_rows": int(df.duplicated(subset=["text"]).sum())
        if "text" in df else None,
        "duplicate_title_rows": int(df.duplicated(subset=["title"]).sum())
        if "title" in df else None,
        "class_distribution": {str(k): int(v) for k, v in
                               df["label"].value_counts(dropna=False).items()}
        if "label" in df else None,
        "words_per_article": {
            "mean": round(float(words.mean()), 1),
            "median": float(words.median()),
            "min": int(words.min()),
            "max": int(words.max()),
        },
    }


def clean_dataset(df: pd.DataFrame):
    """Clean the raw DataFrame.

    Steps (each logged so it can be reported in the capstone document):
      1. Check required columns exist.
      2. Keep only rows whose label is FAKE or REAL.
      3. Drop rows with missing / blank article text.
      4. Remove texts that appear with CONFLICTING labels (cannot be trusted).
      5. Remove exact duplicate texts (prevents the same article landing in
         both the training and test sets = data leakage).
      6. Build `content` = title + text and the integer target `y`.

    Returns (clean_df, log_dict).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    log = {"rows_initial": int(len(df))}
    df = df.copy()

    df["label"] = df["label"].astype("string").str.strip().str.upper()
    before = len(df)
    df = df[df["label"].isin(LABEL_TO_INT)]
    log["dropped_missing_or_unknown_label"] = int(before - len(df))

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    before = len(df)
    df = df[df["text"] != ""]
    log["dropped_blank_text"] = int(before - len(df))

    before = len(df)
    n_labels = df.groupby("text")["label"].transform("nunique")
    df = df[n_labels == 1]
    log["dropped_conflicting_label_duplicates"] = int(before - len(df))

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    log["dropped_duplicate_text"] = int(before - len(df))

    df["content"] = (df["title"] + ". " + df["text"]).str.strip()
    df["y"] = df["label"].map(LABEL_TO_INT).astype(int)
    df = df.reset_index(drop=True)
    log["rows_final"] = int(len(df))
    return df, log
