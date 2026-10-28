# Capstone Progress & Evidence Document

Use this to prepare the **Mid-Term Progress** and **Phase III** evaluations. Every number below was produced by code
run on 2026-10-27 (Python 3.12.3, scikit-learn 1.8.0) in a development sandbox,
**not yet on your own machine**. Re-run `python -m src.train` on your PC and use *your* output in the final report
(values should match closely if library versions are similar, but confirm).

## A. Status of work

### Completed and verified (code executed, output observed)
- Dataset downloaded, loaded and inspected (`src/download_data.py`, `src/data_loader.py`).
- Cleaning pipeline: removed 36 blank-text rows and 241 duplicates → 6,058 articles.
- Preprocessing module shared by training and prediction (`src/preprocessing.py`).
- VADER sentiment module (`src/sentiment.py`).
- Training/evaluation pipeline: Naive Bayes, Logistic Regression, Logistic Regression + sentiment (`src/train.py`).
- Saved artefacts: `models/tfidf_vectorizer.joblib`, `models/logreg_model.joblib`, `models/model_metadata.json`.
- Prediction module with input validation (`src/predict.py`).
- Streamlit app (`app/streamlit_app.py`) — exercised with Streamlit's headless test harness for empty input, too-short input and a normal article.
  **Not yet viewed in a real browser by me** — you must do this and capture screenshots.
- Test suite: 19 cases, 19 passed (`docs/TEST_REPORT.md`). One initial failure (T14) was a mistake in my *expected* value, corrected and documented.
- Figures in `reports/figures/`; architecture diagram `docs/architecture.png`.

### Ongoing / needs you
- Run the project on your own machine; capture screenshots (home, result, error) into `docs/screenshots/`.
- Verify the dataset's original source/author/license and cite it.
- Review `reports/top_features.csv` and discuss dataset bias in your own words.

### Planned (NOT implemented)
- Second-dataset evaluation (e.g. LIAR) to measure generalisation.
- Hyper-parameter tuning, probability calibration, model explanations (LIME/SHAP), transformer model, deployment.

## B. Report sections (draft content — edit in your own words)

### 1. System architecture and design
See `docs/architecture.png` and README §3. Two pipelines: an offline training pipeline and an online prediction pipeline that
loads the saved TF-IDF vectorizer and classifier once. Sentiment is a separate module run on raw text and not used to decide FAKE/REAL.
Configuration is centralised in `src/config.py`.

### 2. Methodology and implementation
See README §6. Key design choices: split before fitting; duplicates removed to prevent leakage; negations kept when removing
stop-words; no stemming; unigrams+bigrams TF-IDF; fixed seed 42; feature extraction inside sklearn pipelines so it is fitted only on training folds.

### 3. Prototype evidence
Terminal output of `python -m src.train` (also saved in `reports/train_log.txt`); `python -m src.predict "..."`; Streamlit screenshots (**you to capture**).

### 4. Dataset and experimental setup
6,335 raw articles (REAL 3,171 / FAKE 3,164) →
6,058 after cleaning; stratified 80/20 split → 4,846 train / 1,212 test; seed 42;
TF-IDF vocabulary 30,000 terms; 5-fold CV on the training set.

### 5. Testing and validation
19 test cases covering empty, whitespace, None, numeric and symbol-only input, very short text, headline-length text, normal and
~7,000-word text, three sentiment cases, missing dataset values, preprocessing, model-loading failure and an end-to-end consistency
check (predictor accuracy over the full test set equals the stored metric). Expected and observed results are in `docs/TEST_REPORT.md`.

### 6. Quantitative results
| Model | Accuracy | Precision (FAKE) | Recall (FAKE) | F1 (FAKE) | CV accuracy |
|---|---|---|---|---|---|
| Naive Bayes | 0.9290 | 0.9415 | 0.9169 | 0.9290 | 0.9109 ± 0.0064 |
| Logistic Regression | 0.9447 | 0.9334 | 0.9593 | 0.9462 | 0.9216 ± 0.0072 |
| LR + sentiment | 0.9464 | 0.9309 | 0.9658 | 0.9480 | 0.9185 ± 0.0053 |

Headline-only accuracy of the primary model: 60.4% (n=1,212).
Interpretation and caveats: README §10–11. Do **not** claim sentiment improves classification — the evidence does not support it.

### 7. Module integration
`config` → used by all modules; `data_loader` + `preprocessing` + `sentiment` → `train` → saved artefacts → `predict` → `streamlit_app`.
Integration is evidenced by test T19 (saved model through the public predictor reproduces the stored test accuracy) and the app run.

### 8. Technical challenges and limitations
- Duplicate articles risked train/test leakage → removed before splitting.
- Dataset shortcuts (dates, topics, site boilerplate) inflate accuracy → documented via top features.
- Headlines are out-of-distribution for a model trained on articles (60.4%).
- Sentiment gave no reliable gain.
- My initial T14 expectation was wrong (kept one copy of duplicates) — corrected.

## C. Evidence checklist
| Item | File / command | Status |
|---|---|---|
| Dataset loaded & inspected | `reports/train_log.txt`, `reports/metrics.json` | Done (sandbox) — re-run on your PC |
| Cleaning log | `reports/metrics.json` → `cleaning_log` | Done |
| Trained model + vectorizer | `models/*.joblib` | Done |
| Metrics, classification reports | `reports/metrics.json`, `reports/classification_reports.txt` | Done |
| Confusion matrices, class chart, comparison chart | `reports/figures/` | Done |
| Architecture diagram | `docs/architecture.png` | Done |
| Test report (expected vs observed) | `docs/TEST_REPORT.md` | Done — 19/19 |
| README | `README.md` | Done (screenshots section pending) |
| App screenshots | `docs/screenshots/` | **Pending — you** |
| Dataset provenance/licence verified | — | **Pending — you** |
| Second-dataset generalisation test | — | Planned |
