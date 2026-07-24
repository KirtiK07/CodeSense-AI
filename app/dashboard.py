"""
dashboard.py
-------------
The CodeSense AI Streamlit app. Wires together:
    upload -> detect language -> static_metrics -> ML risk prediction
                                                  -> LLM review (independent)
                                -> combined report

Run from the project root:
    streamlit run app/dashboard.py
"""

import os
import sys
import tempfile

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make sibling packages importable (analysis/, training/, ai_review/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai_review"))

from static_metrics import extract_features, StaticAnalysisError  # noqa: E402
from language_detect import detect_language, is_supported  # noqa: E402
from train_model import probability_to_risk          # noqa: E402
from llm_review import get_review                    # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "defect_model.pkl")

RISK_COLORS = {"Low": "#66BB6A", "Medium": "#FFCA28", "High": "#EF5350"}


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def run_pipeline(file_path: str, bundle: dict) -> dict:
    """
    Runs the full ML + LLM pipeline on one file and returns everything the
    UI needs. Kept separate from the Streamlit rendering code so it can be
    tested/called independently of the UI layer.
    """
    result = extract_features(file_path, bundle)
    features, language, coverage = result["features"], result["language"], result["coverage"]

    row = pd.DataFrame([features])[bundle["feature_names"]]
    probability = float(bundle["model"].predict_proba(row)[0][1])
    risk = probability_to_risk(probability)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    measured_count = sum(1 for v in coverage.values() if v == "measured")
    review = get_review(code, language, risk, probability, measured_count)

    importances = dict(zip(bundle["feature_names"], bundle["model"].feature_importances_))

    return {
        "language": language,
        "features": features,
        "coverage": coverage,
        "measured_count": measured_count,
        "probability": probability,
        "risk": risk,
        "review": review,
        "feature_importances": importances,
        "code": code,
    }


def render_risk_gauge(probability: float, risk: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": RISK_COLORS[risk]},
            "steps": [
                {"range": [0, 33], "color": "#e8f5e9"},
                {"range": [33, 66], "color": "#fff8e1"},
                {"range": [66, 100], "color": "#ffebee"},
            ],
        },
        title={"text": "Predicted Defect Probability"},
    ))
    fig.update_layout(
        height=250, margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(importances: dict, coverage: dict, top_n: int = 8):
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [n for n, _ in top]
    values = [v for _, v in top]
    colors = ["#4FC3F7" if coverage.get(n) == "measured" else "#6b7280" for n in names]

    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors))
    fig.update_layout(
        title="Top Feature Importances (blue = measured, grey = estimated)",
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_review(review: dict):
    if review.get("error"):
        st.error(f"AI review unavailable: {review['error']}")
        return

    st.markdown("**Summary**")
    st.write(review["summary"] or "_No summary returned._")

    def bullet_section(title, items):
        st.markdown(f"**{title}**")
        if items:
            for item in items:
                st.markdown(f"- {item}")
        else:
            st.caption("None found.")

    bullet_section("🐛 Bugs", review["bugs"])
    bullet_section("🔒 Security Issues", review["security_issues"])
    bullet_section("💡 Suggestions", review["suggestions"])
    bullet_section("✅ Best Practices", review["best_practices"])

    if review["refactored_code"]:
        st.markdown("**Refactored Code**")
        st.code(review["refactored_code"], language="python")


def main():
    st.set_page_config(page_title="CodeSense AI", layout="wide")
    st.title("🔍 CodeSense AI")
    st.caption("ML-based defect risk prediction + independent AI code review")

    bundle = load_model_bundle()
    if bundle is None:
        st.error(
            f"No trained model found at `{MODEL_PATH}`. "
            f"Run `python training/train_model.py` first."
        )
        return

    uploaded = st.file_uploader(
        "Upload a source code file",
        type=["py", "java", "js", "jsx", "ts", "tsx", "c", "h", "cpp", "cc", "hpp"],
    )

    if uploaded is None:
        st.info("Upload a file to get started.")
        return

    language_preview = detect_language(uploaded.name)
    if not is_supported(language_preview):
        st.error(f"Unsupported file type: {uploaded.name}")
        return

    st.write(f"**Detected language:** {language_preview}")

    MAX_FILE_SIZE_KB = 500
    if uploaded.size > MAX_FILE_SIZE_KB * 1024:
        st.error(
            f"File is {uploaded.size // 1024} KB, which exceeds the "
            f"{MAX_FILE_SIZE_KB} KB limit for this demo. Try a smaller file "
            f"or a single module rather than a whole bundled/minified file."
        )
        return

    if st.button("Run Analysis", type="primary"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Running static analysis, ML prediction, and AI review..."):
                result = run_pipeline(tmp_path, bundle)
        except StaticAnalysisError as e:
            st.error(f"Couldn't analyze this file: {e}")
            return
        finally:
            os.unlink(tmp_path)

        col_ml, col_ai = st.columns(2)

        with col_ml:
            st.subheader("📊 ML Quality / Risk Prediction")
            risk = result["risk"]
            st.markdown(
                f"### Risk Level: <span style='color:{RISK_COLORS[risk]}'>{risk}</span>",
                unsafe_allow_html=True,
            )
            render_risk_gauge(result["probability"], risk)
            st.caption(
                f"{result['measured_count']}/21 features measured directly from code; "
                f"the rest were estimated from training-set medians "
                f"({'full parity for Python' if result['language'] == 'python' else 'partial for ' + result['language']})."
            )
            render_feature_importance(result["feature_importances"], result["coverage"])
            with st.expander("View extracted feature values"):
                st.json(result["features"])

        with col_ai:
            st.subheader("🤖 AI Code Review")
            render_review(result["review"])

        with st.expander("View uploaded code"):
            st.code(result["code"], language=result["language"])


if __name__ == "__main__":
    main()