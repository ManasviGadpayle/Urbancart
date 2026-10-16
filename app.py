from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REGRESSION_MODEL_PATH = ROOT / "urbankart_best_regression_model.joblib"
REVIEWS_DATASET_PATH = ROOT / "UrbanKart_HuggingFace_NLP_Customer_Reviews.csv"
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_TEXT_LENGTH = 512

# Configure the Streamlit app only once, at the entry point.
st.set_page_config(
    page_title="UrbanKart Customer Intelligence Hub",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #f7f9fc; }
[data-testid="stMetric"] {
    background: #ffffff; padding: 15px; border: 1px solid #e7ebf2;
    border-radius: 12px;
}
.urban-header {
    padding: 1.2rem 1.5rem; border-radius: 16px; margin-bottom: 1rem;
    background: linear-gradient(120deg, #14213d, #245b86); color: white;
}
.urban-header p { color: #eaf2fa; margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)


def render_header(subtitle: str) -> None:
    st.markdown(
        f'<div class="urban-header"><h1>🛍️ UrbanKart Customer Intelligence Hub</h1>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# ---------------- Shared, cached model/data loaders ----------------
@st.cache_resource(show_spinner=False)
def load_regression_artifact(model_path: str, modified_time: float):
    """Load the saved regression artifact once per process."""
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact or "features" not in artifact:
        raise ValueError(
            "The regression artifact must be a dictionary containing 'model' and 'features'."
        )
    return artifact


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline(model_id: str):
    """Load the Hugging Face pipeline once and reuse it across page reruns."""
    from transformers import pipeline
    return pipeline(
        task="sentiment-analysis",
        model=model_id,
        tokenizer=model_id,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
    )


@st.cache_data(show_spinner=False)
def load_reviews_dataset(path_string: str, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(path_string)


# ---------------- Revenue Prediction page ----------------
def revenue_prediction_page() -> None:
    render_header("Revenue Prediction · Estimate annual customer revenue")
    st.subheader("Customer Revenue Prediction")
    st.write("Enter a customer's profile and engagement details to estimate annual revenue.")

    if not REGRESSION_MODEL_PATH.exists():
        st.error(f"Regression model file not found: {REGRESSION_MODEL_PATH.name}")
        st.info("Place urbankart_best_regression_model.joblib in the project root beside app.py.")
        return

    try:
        artifact = load_regression_artifact(
            str(REGRESSION_MODEL_PATH), REGRESSION_MODEL_PATH.stat().st_mtime
        )
        model = artifact["model"]
        features = list(artifact["features"])
        model_name = artifact.get("model_name", type(model).__name__)
    except Exception as exc:
        st.error("Could not load the revenue prediction model.")
        st.exception(exc)
        return

def customer_feedback_page() -> None:
    render_header("Customer Feedback · Sentiment and Examples")
    st.subheader("Customer Feedback Analysis")
    st.write("Upload or use the sample dataset to inspect reviews and run sentiment analysis.")

    if REVIEWS_DATASET_PATH.exists():
        try:
            df = load_reviews_dataset(str(REVIEWS_DATASET_PATH), REVIEWS_DATASET_PATH.stat().st_mtime)
            st.markdown("**Dataset preview:**")
            st.dataframe(df.head(10))
        except Exception as exc:
            st.error("Could not load the reviews dataset.")
            st.exception(exc)
            return
    else:
        st.info(f"Place {REVIEWS_DATASET_PATH.name} in the project root beside app.py or upload a file.")

    # Run sentiment pipeline on demand (loads model lazily)
    if st.button("Run sentiment on first 10 reviews"):
        if not REVIEWS_DATASET_PATH.exists():
            st.error("No reviews dataset available to analyze.")
        else:
            with st.spinner("Loading sentiment model and running analysis..."):
                try:
                    pipe = load_sentiment_pipeline(SENTIMENT_MODEL_ID)
                    texts = df.iloc[:10]["review_text"].astype(str).tolist() if "review_text" in df.columns else df.iloc[:10].astype(str).values.flatten().tolist()
                    results = pipe(texts)
                    st.write(results)
                except Exception as exc:
                    st.error("Sentiment pipeline failed.")
                    st.exception(exc)

def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select page", ["Revenue Prediction", "Customer Feedback"])
    if page == "Revenue Prediction":
        revenue_prediction_page()
    else:
        customer_feedback_page()

if __name__ == "__main__":
    main()
