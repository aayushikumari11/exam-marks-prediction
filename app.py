"""Single-page Exam Marks Prediction application."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pkl"
DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "student_exam_data.csv"

st.set_page_config(
    page_title="Exam Marks Prediction",
    page_icon="📘",
    layout="wide",
)


# Keep the existing artifact format. This app loads the trained model only.
@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


def get_estimator(model):
    """Get the final estimator when the saved model is a scikit-learn Pipeline."""
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def get_model_name(artifact, estimator):
    """Use the stored name when present; otherwise show the estimator class."""
    return artifact.get("model_name") or estimator.__class__.__name__


def performance_category(marks):
    if marks < 40:
        return "Needs Improvement"
    if marks < 60:
        return "Average"
    if marks < 75:
        return "Good"
    if marks < 90:
        return "Very Good"
    return "Excellent"


def make_recommendations(values):
    recommendations = []

    if values["Study_Hours"] < 3:
        recommendations.append(
            "Your study time is below 3 hours per day. Consider adding a "
            "short, focused study session to your routine."
        )

    if values["Attendance"] < 75:
        recommendations.append(
            "Your attendance is below 75%. Improving attendance can help you "
            "keep up with class lessons."
        )

    

    if values["Sleep_Hours"] < 6:
        recommendations.append(
            "Your sleep is below 6 hours per night. Consider maintaining a "
            "more consistent sleep schedule."
        )

    

    if values["Study_Days_Per_Week"] < 3:
        recommendations.append(
            "You study fewer than 3 days per week. Try spreading study "
            "sessions across more days."
        )

    return recommendations


def show_analytics(features=None):
    st.markdown('<div id="analytics"></div>', unsafe_allow_html=True)
    st.header("📊 Analytics")

    if not DATASET_PATH.exists():
        st.info(
            "The dataset file was not found. The prediction section can still "
            "be used with the saved model."
        )
        return

    try:
        data = load_dataset()
    except Exception as error:
        st.info(f"The dataset could not be loaded: {error}")
        return

    target_column = "Final_Exam_Marks"
    identifier_columns = {
        column
        for column in data.columns
        if column.lower() in {"id", "student_id"}
    }

    if features:
        dataset_features = [
            feature for feature in features if feature in data.columns
        ]
    else:
        dataset_features = [
            column
            for column in data.columns
            if column != target_column and column not in identifier_columns
        ]

    record_column, feature_column = st.columns(2)
    record_column.metric("Records", f"{len(data):,}")
    feature_column.metric("Features", len(dataset_features))

    st.subheader("Dataset preview")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

    st.subheader("Basic statistics")
    numeric_data = data.select_dtypes(include="number")
    if numeric_data.empty:
        st.info("No numeric columns are available for summary statistics.")
    else:
        st.dataframe(
            numeric_data.describe().T.round(2),
            use_container_width=True,
        )

    