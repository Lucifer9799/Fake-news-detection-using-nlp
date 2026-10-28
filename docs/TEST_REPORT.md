# Test Report

Generated: 2026-10-27 17:06 UTC by `python -m tests.generate_test_report`

**Result: 19 / 19 test cases passed.**

| ID | Area | Test | Expected result | Observed result | Status |
|----|------|------|-----------------|-----------------|--------|
| T01 | Empty input | Empty string | InvalidInputError raised (no crash, clear message) | InvalidInputError: Input is empty. Please paste a headline or article. | PASS |
| T02 | Empty input | Whitespace only | InvalidInputError raised | InvalidInputError: Input is empty. Please paste a headline or article. | PASS |
| T03 | Invalid input | None instead of text | InvalidInputError raised | InvalidInputError: Input must be text (a string). | PASS |
| T04 | Invalid input | Number instead of text (12345) | InvalidInputError raised | InvalidInputError: Input must be text (a string). | PASS |
| T05 | Invalid input | Only symbols: '!!!! ???? ####' | InvalidInputError raised (no meaningful words) | InvalidInputError: The text has no meaningful words after cleaning (only symbols, numbers or common filler words). | PASS |
| T06 | Very short text | Two words: 'Breaking news' | InvalidInputError raised (fewer than 3 words) | InvalidInputError: Input is too short. Please enter at least 3 words. | PASS |
| T07 | Very short text | Headline-length text (9 words) | A FAKE/REAL prediction is returned WITH a short-text warning | FAKE (73.7%); sentiment Positive (+0.402); warnings=1 | PASS |
| T08 | Normal prediction | Article-length paragraph (~90 words) | Valid prediction; probabilities sum to 1; confidence in [0.5, 1] | REAL (59.6%); sentiment Positive (+0.840); warnings=1 | PASS |
| T09 | Long article | ~7,000-word text | Prediction returned without error | REAL (57.9%); sentiment Positive (+1.000); warnings=1 | PASS |
| T10 | Sentiment | Clearly positive text | Label Positive, compound >= 0.05 | Positive (compound +0.958) | PASS |
| T11 | Sentiment | Clearly negative text | Label Negative, compound <= -0.05 | Negative (compound -0.934) | PASS |
| T12 | Sentiment | Neutral factual text | Label Neutral | Neutral (compound +0.000) | PASS |
| T13 | Sentiment | Sentiment of empty text | Neutral with compound 0.0 (no crash) | Neutral (compound +0.000) | PASS |
| T14 | Missing dataset values | 8-row dirty DataFrame: missing title/text, blank text, duplicate, conflicting-label duplicate | Rows kept = 3 ('good article one', 'has missing title', and ONE copy of 'duplicate body'); missing/blank text and the conflicting-label pair removed | rows kept = 3; log = {'rows_initial': 8, 'dropped_missing_or_unknown_label': 0, 'dropped_blank_text': 2, 'dropped_conflicting_label_duplicates': 2, 'dropped_duplicate_text': 1, 'rows_final': 3} | PASS |
| T15 | Missing dataset values | DataFrame without a 'text' column | ValueError raised naming the missing column | ValueError: Dataset is missing required column(s): ['text'] | PASS |
| T16 | Preprocessing | URL/HTML removed, negation + year kept | 'not' and '2016' kept; 'http', 'b' and stop-word 'the' removed | 'claim not true see 2016' | PASS |
| T17 | Preprocessing | Missing/invalid values (None, NaN, 5) | All three return '' (no crash) | ['', '', ''] | PASS |
| T18 | Model loading | Models directory does not exist | ModelNotFoundError raised with instruction to train | ModelNotFoundError: Trained model files not found in /nonexistent/folder. Run:   | PASS |
| T19 | End-to-end consistency | Saved model + vectorizer run over the full held-out test set via the public predictor | Accuracy equals the accuracy stored in reports/metrics.json (proves training and prediction preprocessing match) | n=1212, accuracy=0.9447, saved=0.9447 | PASS |

> Note: for hand-written sample text (T07-T09) the test checks that the system behaves correctly (valid output, warnings), NOT that the predicted label is 'right' - there is no ground truth for text I wrote myself. Correctness of predictions is measured on the held-out dataset split (see T19 and reports/metrics.json).
