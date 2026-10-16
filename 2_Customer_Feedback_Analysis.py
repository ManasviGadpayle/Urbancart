
from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline

# ---------------- CONFIGURATION ----------------

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

DATASET_PATH = (
    Path(__file__).parent
    / "UrbanKart_HuggingFace_NLP_Customer_Reviews.csv"
)

st.set_page_config(
    page_title="UrbanKart Customer Intelligence Hub",
    page_icon="🛍️",
    layout="wide",
)

# ---------------- PROFESSIONAL UI ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.header {
    padding: 25px;
    border-radius: 14px;
    background: linear-gradient(110deg, #14213d, #286090);
    color: white;
    margin-bottom: 20px;
}
.header p {
    color: #e5edf8;
}
[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e0e5ed;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>UrbanKart Customer Intelligence Hub</h1>
    <p>Customer feedback and sentiment analysis powered by AI</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Analysis Settings")
st.sidebar.write("Model:")
st.sidebar.code(MODEL_NAME)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
    )


# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_dataset(file_path, modified_time):
    return pd.read_csv(file_path)


# ---------------- DETECT REVIEW COLUMN ----------------

def detect_review_column(df):
    preferred = [
        "review_text",
        "review",
        "customer_review",
        "feedback",
        "feedback_text",
        "text",
        "comment",
        "comments",
    ]

    columns = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for name in preferred:
        if name in columns:
            return columns[name]

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if not text_columns:
        return None

    return max(
        text_columns,
        key=lambda col: df[col].dropna().astype(str).str.len().mean()
        if df[col].notna().any() else 0,
    )


# ---------------- SENTIMENT LABELS ----------------

def format_sentiment(label):
    label = str(label).lower()

    if "negative" in label or label == "label_0":
        return "Negative"

    if "neutral" in label or label == "label_1":
        return "Neutral"

    if "positive" in label or label == "label_2":
        return "Positive"

    return label.title()


# ---------------- ANALYZE REVIEWS ----------------

def analyze_reviews(classifier, reviews):
    predictions = []

    batch_size = 32

    for start in range(0, len(reviews), batch_size):
        batch = reviews[start:start + batch_size]

        result = classifier(
            batch,
            batch_size=batch_size,
            truncation=True,
            max_length=512,
        )

        predictions.extend(result)

    return pd.DataFrame({
        "Predicted Sentiment": [
            format_sentiment(item["label"])
            for item in predictions
        ],
        "Confidence Score": [
            round(float(item["score"]), 4)
            for item in predictions
        ],
    })


# ---------------- APPLICATION TABS ----------------

single_tab, batch_tab = st.tabs([
    "Single Review Analysis",
    "Batch Analysis",
])

# ==================================================
# SINGLE REVIEW ANALYSIS
# ==================================================

with single_tab:
    st.subheader("Analyze Individual Customer Feedback")

    review = st.text_area(
        "Enter customer review",
        placeholder=(
            "Example: The product quality is excellent "
            "and delivery was fast."
        ),
        height=130,
    )

    if st.button("Analyze Review", type="primary"):
        if not review.strip():
            st.warning("Please enter a customer review.")

        else:
            try:
                with st.spinner("Analyzing customer sentiment..."):
                    classifier = load_model()

                    result = classifier(
                        review.strip(),
                        truncation=True,
                        max_length=512,
                    )[0]

                sentiment = format_sentiment(result["label"])
                confidence = float(result["score"])

                col1, col2 = st.columns(2)

                col1.metric("Predicted Sentiment", sentiment)
                col2.metric("Confidence Score", f"{confidence:.2%}")

                st.progress(min(max(confidence, 0), 1))

            except Exception as error:
                st.error("Unable to analyze the review.")
                st.exception(error)


# ==================================================
# BATCH ANALYSIS
# ==================================================

with batch_tab:
    st.subheader("Batch Customer Feedback Analysis")

    uploaded_file = st.file_uploader(
        "Upload a customer reviews CSV (optional)",
        type=["csv"],
    )

    df = None
    dataset_name = ""

    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name

        elif DATASET_PATH.exists():
            df = load_dataset(
                str(DATASET_PATH),
                DATASET_PATH.stat().st_mtime,
            )
            dataset_name = DATASET_PATH.name

    except Exception as error:
        st.error("Unable to load the CSV file.")
        st.exception(error)

    if df is None:
        st.info(
            "Place UrbanKart_HuggingFace_NLP_Customer_Reviews.csv "
            "in the same folder as this Python file, or upload a CSV."
        )

    elif df.empty:
        st.warning("The dataset is empty.")

    else:
        review_column = detect_review_column(df)

        st.write(f"Dataset: **{dataset_name}**")
        st.write(f"Total rows: **{len(df):,}**")

        if review_column is None:
            st.error("Could not identify a review-text column.")

        else:
            st.success(
                f"Detected review text column: `{review_column}`"
            )

            st.dataframe(
                df.head(),
                use_container_width=True,
                hide_index=True,
            )

            valid_mask = (
                df[review_column].notna()
                & df[review_column].astype(str).str.strip().ne("")
            )

            st.write(
                f"Reviews available for analysis: "
                f"**{int(valid_mask.sum()):,}**"
            )

            if st.button("Run Batch Analysis", type="primary"):
                if not valid_mask.any():
                    st.warning("No valid reviews found.")

                else:
                    try:
                        classifier = load_model()

                        reviews = (
                            df.loc[valid_mask, review_column]
                            .astype(str)
                            .tolist()
                        )

                        progress = st.progress(0)
                        predictions = []

                        batch_size = 32

                        for start in range(
                            0, len(reviews), batch_size
                        ):
                            batch = reviews[
                                start:start + batch_size
                            ]

                            output = analyze_reviews(
                                classifier,
                                batch,
                            )

                            predictions.extend(
                                output.to_dict("records")
                            )

                            progress.progress(
                                min(
                                    (start + len(batch)) / len(reviews),
                                    1.0,
                                )
                            )

                        result_df = df.copy()

                        result_df["Predicted Sentiment"] = ""
                        result_df["Confidence Score"] = pd.NA

                        result_df.loc[
                            valid_mask,
                            "Predicted Sentiment",
                        ] = [
                            item["Predicted Sentiment"]
                            for item in predictions
                        ]

                        result_df.loc[
                            valid_mask,
                            "Confidence Score",
                        ] = [
                            item["Confidence Score"]
                            for item in predictions
                        ]

                        st.session_state["analysis_results"] = result_df
                        st.session_state["analysis_source"] = dataset_name

                        st.success("Batch analysis completed.")

                    except Exception as error:
                        st.error("Batch analysis failed.")
                        st.exception(error)

            result_df = st.session_state.get("analysis_results")

            if (
                result_df is not None
                and st.session_state.get("analysis_source") == dataset_name
            ):
                analyzed = result_df[
                    result_df["Predicted Sentiment"].astype(str).ne("")
                ]

                counts = (
                    analyzed["Predicted Sentiment"]
                    .value_counts()
                    .reindex(
                        ["Positive", "Neutral", "Negative"],
                        fill_value=0,
                    )
                )

                st.divider()
                st.subheader("Sentiment Distribution")

                col1, col2, col3 = st.columns(3)

                col1.metric("Positive", int(counts["Positive"]))
                col2.metric("Neutral", int(counts["Neutral"]))
                col3.metric("Negative", int(counts["Negative"]))

                chart_df = counts.rename_axis(
                    "Sentiment"
                ).reset_index(name="Reviews")

                st.bar_chart(
                    chart_df.set_index("Sentiment")
                )

                st.subheader("Analysis Results")

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                )

                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "Download Results as CSV",
                    data=csv_data,
                    file_name="UrbanKart_Sentiment_Analysis.csv",
                    mime="text/csv",
                    type="primary",
                )

st.divider()
st.caption(
    "UrbanKart Customer Intelligence Hub | "
    "AI-generated sentiment is an estimate and should be reviewed "
    "for important customer decisions."
)
