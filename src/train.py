"""Training + evaluation pipeline.  Run with:   python -m src.train

What it does (in order):
  1. Load + inspect + clean the dataset
  2. Compute VADER sentiment (on RAW text) and preprocess text
  3. Split into train/test (stratified, fixed seed) BEFORE fitting anything
  4. Fit three models on the TRAIN set only:
        A) Naive Bayes + TF-IDF          (baseline)
        B) Logistic Regression + TF-IDF  (primary model)
        C) Logistic Regression + TF-IDF + VADER features (sentiment experiment)
  5. Evaluate on the untouched TEST set (+ 5-fold CV on train for stability)
  6. Save figures, metrics.json, reports, and the deployed model artifacts
"""
import json
import platform
import sys
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.data_loader import clean_dataset, inspect_dataset, load_raw
from src.preprocessing import preprocess_text
from src.sentiment import sentiment_features

TARGET_NAMES = ["REAL", "FAKE"]  # index 0 = REAL, 1 = FAKE


# ----------------------------------------------------------------- models
def build_models() -> dict:
    """Three pipelines. Feature extraction lives INSIDE each pipeline, so it
    is only ever fitted on training data (also inside each CV fold)."""
    def tfidf():
        return TfidfVectorizer(**C.TFIDF_PARAMS)

    text_only = lambda: ColumnTransformer([("tfidf", tfidf(), "clean_text")])
    text_plus_sent = lambda: ColumnTransformer([
        ("tfidf", tfidf(), "clean_text"),
        ("sent", StandardScaler(), C.SENTIMENT_COLUMNS),
    ])
    lr = lambda: LogisticRegression(max_iter=1000, random_state=C.RANDOM_SEED)
    return {
        "Naive Bayes (TF-IDF)": Pipeline([("features", text_only()),
                                          ("clf", MultinomialNB())]),
        "Logistic Regression (TF-IDF)": Pipeline([("features", text_only()),
                                                  ("clf", lr())]),
        "Logistic Regression (TF-IDF + sentiment)": Pipeline(
            [("features", text_plus_sent()), ("clf", lr())]),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, pos_label=1),
        "recall_fake": recall_score(y_true, y_pred, pos_label=1),
        "f1_fake": f1_score(y_true, y_pred, pos_label=1),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------- figures
def plot_class_distribution(raw_counts, clean_counts, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, counts, title in zip(axes, [raw_counts, clean_counts],
                                 ["Raw dataset", "After cleaning"]):
        labels = ["REAL", "FAKE"]
        vals = [counts.get(l, 0) for l in labels]
        bars = ax.bar(labels, vals, color=["#2a9d8f", "#e76f51"])
        ax.set_title(title)
        ax.bar_label(bars)
    axes[0].set_ylabel("Number of articles")
    fig.suptitle("Class distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(np.array(cm), display_labels=TARGET_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(results, path):
    metrics = ["accuracy", "precision_fake", "recall_fake", "f1_fake"]
    names = list(results)
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, n in enumerate(names):
        vals = [results[n][m] for m in metrics]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=n)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision (FAKE)", "Recall (FAKE)",
                        "F1 (FAKE)"])
    lo = min(results[n][m] for n in names for m in metrics)
    ax.set_ylim(max(0, lo - 0.1), 1.02)
    ax.set_title("Model comparison on the held-out test set", pad=28)
    ax.set_ylabel(f"Score (note: axis starts at {max(0, lo - 0.1):.2f})")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sentiment_by_class(df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab, color in [("REAL", "#2a9d8f"), ("FAKE", "#e76f51")]:
        ax.hist(df.loc[df["label"] == lab, "vader_compound"], bins=40,
                alpha=0.6, label=lab, color=color)
    ax.set_xlabel("VADER compound score")
    ax.set_ylabel("Articles")
    ax.set_title("Sentiment distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main() -> None:
    for d in (C.MODELS_DIR, C.REPORTS_DIR, C.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load / inspect / clean ------------------------------------------
    raw = load_raw()
    inspection = inspect_dataset(raw)
    print("RAW DATA:", json.dumps(inspection, indent=2))
    df, cleaning_log = clean_dataset(raw)
    print("CLEANING LOG:", json.dumps(cleaning_log, indent=2))

    # 2. Features that do not depend on the split -------------------------
    # VADER is rule-based (nothing is "learned"), and preprocessing is a pure
    # function, so computing them before the split does not leak information.
    print("Computing VADER sentiment ...")
    sent = sentiment_features(df["content"])
    df = pd.concat([df, sent], axis=1)
    print("Preprocessing text ...")
    df["clean_text"] = df["content"].map(preprocess_text)
    before = len(df)
    df = df[df["clean_text"].str.split().str.len() > 0].reset_index(drop=True)
    cleaning_log["dropped_empty_after_preprocessing"] = int(before - len(df))
    cleaning_log["rows_final"] = int(len(df))

    # 3. Split BEFORE fitting anything -------------------------------------
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=C.TEST_SIZE,
        random_state=C.RANDOM_SEED, stratify=df["y"])
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    feature_cols = ["clean_text"] + C.SENTIMENT_COLUMNS
    X_train, y_train = train[feature_cols], train["y"]
    X_test, y_test = test[feature_cols], test["y"]
    print(f"Train: {len(train)}  Test: {len(test)}")
    test[["id"]].to_csv(C.TEST_IDS_PATH, index=False)

    # 4/5. Fit, cross-validate, evaluate -----------------------------------
    results, reports, fitted = {}, {}, {}
    for name, pipe in build_models().items():
        print(f"\n=== {name} ===")
        cv = cross_validate(clone(pipe), X_train, y_train, cv=C.CV_FOLDS,
                            scoring=["accuracy", "f1"], n_jobs=1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        res = evaluate(y_test, pred)
        res["cv_accuracy_mean"] = float(cv["test_accuracy"].mean())
        res["cv_accuracy_std"] = float(cv["test_accuracy"].std())
        res["cv_f1_mean"] = float(cv["test_f1"].mean())
        res["train_accuracy"] = float(accuracy_score(y_train, pipe.predict(X_train)))
        results[name] = res
        reports[name] = classification_report(y_test, pred,
                                              target_names=TARGET_NAMES, digits=4)
        fitted[name] = pipe
        print(reports[name])
        print(f"Accuracy {res['accuracy']:.4f} | CV acc {res['cv_accuracy_mean']:.4f}"
              f" +/- {res['cv_accuracy_std']:.4f} | train acc {res['train_accuracy']:.4f}")

    # 6a. Extra analysis on the primary model --------------------------------
    primary_name = "Logistic Regression (TF-IDF)"
    primary = fitted[primary_name]
    vec = primary.named_steps["features"].named_transformers_["tfidf"]
    clf = primary.named_steps["clf"]

    # Headline-only stress test: the model was trained on full articles, so
    # how does it cope when given only the headline?
    titles = test["title"].map(preprocess_text)
    keep = titles.str.len() > 0
    head_pred = clf.predict(vec.transform(titles[keep]))
    headline_acc = float(accuracy_score(y_test[keep], head_pred))

    # Most influential words (checks for suspicious dataset artefacts).
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    top = pd.DataFrame({
        "top_REAL_terms": terms[order[:25]],
        "top_REAL_coef": clf.coef_[0][order[:25]],
        "top_FAKE_terms": terms[order[::-1][:25]],
        "top_FAKE_coef": clf.coef_[0][order[::-1][:25]],
    })
    top.to_csv(C.REPORTS_DIR / "top_features.csv", index=False)

    sent_stats = df.groupby("label")["vader_compound"].agg(["mean", "std"]).round(4)

    # 6b. Figures --------------------------------------------------------------
    plot_class_distribution(inspection["class_distribution"],
                            df["label"].value_counts().to_dict(),
                            C.FIGURES_DIR / "class_distribution.png")
    for name, res in results.items():
        slug = (name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                .replace("+", "plus").replace("-", "_"))
        plot_confusion(res["confusion_matrix"], name,
                       C.FIGURES_DIR / f"confusion_matrix_{slug}.png")
    plot_comparison(results, C.FIGURES_DIR / "model_comparison.png")
    plot_sentiment_by_class(df, C.FIGURES_DIR / "sentiment_by_class.png")

    # 6c. Save artefacts ---------------------------------------------------------
    joblib.dump(vec, C.VECTORIZER_PATH)
    joblib.dump(clf, C.MODEL_PATH)
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deployed_model": primary_name,
        "random_seed": C.RANDOM_SEED,
        "versions": {"python": platform.python_version(),
                     "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    C.METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    metrics = {
        **metadata,
        "dataset_inspection_raw": inspection,
        "cleaning_log": cleaning_log,
        "split": {"train": len(train), "test": len(test),
                  "test_size": C.TEST_SIZE, "stratified": True},
        "tfidf_params": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in C.TFIDF_PARAMS.items()},
        "tfidf_vocabulary_size": int(len(terms)),
        "results": results,
        "headline_only_accuracy_primary_model": headline_acc,
        "headline_only_n": int(keep.sum()),
        "sentiment_by_class": sent_stats.to_dict(orient="index"),
    }
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    with open(C.REPORTS_DIR / "classification_reports.txt", "w") as f:
        for name, rep in reports.items():
            f.write(f"=== {name} ===\n{rep}\n")

    print("\nHeadline-only accuracy (primary model):", round(headline_acc, 4))
    print("Sentiment by class:\n", sent_stats)
    print("\nTop REAL terms:", list(top["top_REAL_terms"][:15]))
    print("Top FAKE terms:", list(top["top_FAKE_terms"][:15]))
    print(f"\nSaved model  -> {C.MODEL_PATH}\nSaved vectorizer -> {C.VECTORIZER_PATH}"
          f"\nSaved metrics -> {C.METRICS_PATH}")


if __name__ == "__main__":
    sys.exit(main())
