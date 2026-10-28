"""Streamlit UI.   Launch with:   streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

# Make `src` importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src import config as C
from src.predict import FakeNewsPredictor, InvalidInputError, ModelNotFoundError

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource(show_spinner="Loading trained model ...")
def load_predictor() -> FakeNewsPredictor:
    """Load the saved model ONCE (not on every click)."""
    return FakeNewsPredictor()


def load_metrics():
    try:
        return json.loads(C.METRICS_PATH.read_text())
    except Exception:
        return None


st.title("📰 Fake News Detector")
st.caption("NLP + machine learning (TF-IDF + Logistic Regression) with VADER sentiment analysis")

st.info(
    "**Please note:** this tool gives a *statistical estimate* based on writing "
    "patterns in a 2016-era dataset. It is **not a fact-check** and can be wrong. "
    "Always verify important claims with trusted sources."
)

try:
    predictor = load_predictor()
except ModelNotFoundError as exc:
    st.error(f"Model not found. {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

text = st.text_area("Paste a news headline or full article:", height=250,
                    placeholder="Paste the text here. Full articles work best ...",
                    max_chars=C.MAX_CHARS)

if st.button("Analyze", type="primary"):
    try:
        result = predictor.predict(text)
    except InvalidInputError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Something went wrong while analysing the text: {exc}")
    else:
        for w in result["warnings"]:
            st.warning(w)

        st.subheader("Result")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Fake-news prediction**")
            if result["label"] == "FAKE":
                st.error(f"Likely FAKE ({result['confidence']:.1%} model confidence)")
            else:
                st.success(f"Likely REAL ({result['confidence']:.1%} model confidence)")
            st.progress(result["probabilities"]["FAKE"],
                        text=f"P(FAKE) = {result['probabilities']['FAKE']:.1%}")
        with col2:
            s = result["sentiment"]
            st.markdown("**Sentiment (VADER)**")
            st.metric(s["label"], f"{s['compound']:+.3f}",
                      help="Compound score from -1 (very negative) to +1 (very positive)")
            st.caption(f"pos {s['pos']:.2f} · neu {s['neu']:.2f} · neg {s['neg']:.2f}")

        st.caption(
            "Sentiment describes the *tone* of the writing, not its truthfulness. "
            "It is shown separately and does not decide the prediction. "
            "Confidence is the model's own probability, not a measure of factual accuracy."
        )

with st.expander("About the model & its limitations"):
    m = load_metrics()
    if m:
        r = m["results"]["Logistic Regression (TF-IDF)"]
        st.write(
            f"Held-out test accuracy: **{r['accuracy']:.1%}**, F1 (FAKE): "
            f"**{r['f1_fake']:.3f}** ({m['split']['test']} test articles). "
            f"Accuracy on headlines alone dropped to "
            f"**{m['headline_only_accuracy_primary_model']:.1%}**."
        )
    st.markdown(
        "- Trained on ~6,000 news articles, mostly US politics from 2015-2016.\n"
        "- The model may learn topic, style and time-period cues rather than truth.\n"
        "- Performance on other topics, newer events or other languages is unknown."
    )
