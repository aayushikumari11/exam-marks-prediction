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

    if values["Practice_Tests"] < 5:
        recommendations.append(
            "You have attempted fewer than 5 practice tests. Try more practice "
            "questions or mock tests."
        )

    if values["Sleep_Hours"] < 6:
        recommendations.append(
            "Your sleep is below 6 hours per night. Consider maintaining a "
            "more consistent sleep schedule."
        )

    if values["Class_Participation"] <= 2:
        recommendations.append(
            "Your class participation rating is low. Try asking or answering "
            "a question during class."
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

    # Use the project's actual dataset column names.
    chart_specs = [
        ("Study_Hours", target_column, "Study Hours vs Marks"),
        ("Attendance", target_column, "Attendance vs Marks"),
        ("Previous_Exam_Marks", target_column, "Previous Marks vs Marks"),
    ]
    available_charts = [
        spec
        for spec in chart_specs
        if spec[0] in data.columns and spec[1] in data.columns
    ]

    st.subheader("Academic factor charts")
    if not available_charts:
        st.info(
            "The dataset does not contain the columns needed for the study "
            "hours, attendance, or previous marks charts."
        )
    else:
        chart_columns = st.columns(len(available_charts))
        for container, (x_column, y_column, title) in zip(
            chart_columns, available_charts
        ):
            with container:
                figure, axis = plt.subplots(figsize=(5, 4))
                sns.scatterplot(
                    data=data,
                    x=x_column,
                    y=y_column,
                    alpha=0.65,
                    color="#2878A5",
                    ax=axis,
                )
                axis.set_title(title)
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

    st.subheader("Correlation heatmap")
    if numeric_data.shape[1] < 2:
        st.info("At least two numeric columns are needed for a heatmap.")
    else:
        figure, axis = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            numeric_data.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=axis,
        )
        axis.set_title("Correlation between numeric dataset columns")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)


def show_feature_importance(model, features):
    estimator = get_estimator(model)
    values = None
    label = None

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).reshape(-1)
        label = "Feature importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        if coefficients.ndim > 1:
            values = np.mean(np.abs(coefficients), axis=0)
        else:
            values = np.abs(coefficients.reshape(-1))
        label = "Absolute coefficient"

    if values is None or len(values) != len(features):
        st.info("Feature importance is not available for the current model.")
        return

    importance_data = pd.DataFrame(
        {"Feature": features, "Value": values}
    ).sort_values("Value", ascending=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=importance_data,
        x="Value",
        y="Feature",
        color="#2878A5",
        ax=axis,
    )
    axis.set_title(label)
    axis.set_xlabel(label)
    axis.set_ylabel("Feature")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

   


# Keep the page on one screen and use in-page links for navigation.
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stAppDeployButton"] {
        display: none;
    }

    .top-navigation {
        display: flex;
        gap: 0.55rem;
        overflow-x: auto;
        padding: 0.65rem;
        margin: 0.5rem 0 1.5rem 0;
        border: 1px solid #dce5ed;
        border-radius: 14px;
        background: #f7fafc;
        white-space: nowrap;
    }

    .top-navigation a {
        display: inline-block;
        padding: 0.6rem 0.9rem;
        border-radius: 10px;
        color: #17324d;
        text-decoration: none;
        font-weight: 600;
    }

    .top-navigation a:hover {
        background: #e3eef6;
        color: #0c5c85;
    }

    [data-testid="stMetric"] {
        padding: 1rem;
        border: 1px solid #dce5ed;
        border-radius: 14px;
        background: #ffffff;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .top-navigation a {
            padding: 0.5rem 0.7rem;
        }
    }
    </style>

    <nav class="top-navigation">
        <a href="#home">🏠 Home</a>
        <a href="#prediction">🎯 Prediction</a>
        <a href="#analytics">📊 Analytics</a>
        <a href="#model">🤖 Model</a>
        <a href="#recommendations">💡 Recommendations</a>
        <a href="#about">ℹ️ About</a>
    </nav>
    """,
    unsafe_allow_html=True,
)


# Load the existing saved artifact. No training happens in this app.
artifact = None
model_error = None

if MODEL_PATH.exists():
    try:
        artifact = load_artifact()
    except Exception as error:
        model_error = str(error)


st.markdown('<div id="home"></div>', unsafe_allow_html=True)
st.title("📘 Exam Marks Prediction")
st.write("Predict exam performance using academic and study-related factors.")


if model_error:
    st.warning(f"The saved model could not be loaded: {model_error}")
elif artifact is None:
    st.warning("The saved model was not found at `models/model.pkl`.")


# Prediction form and result
st.markdown('<div id="prediction"></div>', unsafe_allow_html=True)
st.header("🎯 Prediction")

if artifact is not None:
    model, features = artifact["model"], artifact["features"]

    with st.form("prediction_form"):
        st.subheader("Enter academic and study information")

        first_column, second_column, third_column = st.columns(3)

        with first_column:
            study_hours = st.number_input(
                "Study hours per day",
                min_value=0.0,
                max_value=16.0,
                value=4.0,
                step=0.5,
            )
            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
            )
            previous_marks = st.number_input(
                "Previous exam marks",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
            )
            assignment_marks = st.number_input(
                "Assignment marks",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
            )

        with second_column:
            quiz_marks = st.number_input(
                "Quiz marks",
                min_value=0.0,
                max_value=100.0,
                value=68.0,
                step=1.0,
            )
            internal_marks = st.number_input(
                "Internal marks",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0,
            )
            practice_tests = st.number_input(
                "Practice tests attempted",
                min_value=0,
                max_value=30,
                value=5,
                step=1,
            )
            sleep_hours = st.number_input(
                "Sleep hours per night",
                min_value=0.0,
                max_value=16.0,
                value=7.0,
                step=0.5,
            )

        with third_column:
            gpa = st.number_input(
                "Previous semester GPA (0–10)",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1,
            )
            participation = st.slider(
                "Class participation (1 = low, 5 = high)",
                min_value=1,
                max_value=5,
                value=3,
            )
            study_days = st.number_input(
                "Study days per week",
                min_value=0,
                max_value=7,
                value=5,
                step=1,
            )

        submitted = st.form_submit_button(
            "Predict Exam Marks",
            type="primary",
        )

    if submitted:
        entered_values = {
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Exam_Marks": previous_marks,
            "Assignment_Marks": assignment_marks,
            "Quiz_Marks": quiz_marks,
            "Internal_Marks": internal_marks,
            "Practice_Tests": practice_tests,
            "Sleep_Hours": sleep_hours,
            "Previous_Semester_GPA": gpa,
            "Class_Participation": participation,
            "Study_Days_Per_Week": study_days,
        }

        missing_inputs = [
            feature for feature in features if feature not in entered_values
        ]

        if missing_inputs:
            st.error(
                "The saved model expects inputs that are not present in the "
                f"form: {', '.join(missing_inputs)}"
            )
        else:
            input_frame = pd.DataFrame(
                [[entered_values[feature] for feature in features]],
                columns=features,
            )

            try:
                raw_prediction = model.predict(input_frame)
                predicted_marks = float(np.asarray(raw_prediction).reshape(-1)[0])
                predicted_marks = max(0.0, min(100.0, predicted_marks))

                st.session_state["exam_prediction"] = {
                    "marks": predicted_marks,
                    "inputs": entered_values,
                }
            except Exception as error:
                st.error(f"Could not generate a prediction: {error}")

    result = st.session_state.get("exam_prediction")

    if result:
        marks = result["marks"]
        values = result["inputs"]
        category = performance_category(marks)

        st.subheader("Prediction result")
        marks_column, category_column = st.columns(2)
        marks_column.metric("Predicted marks", f"{marks:.1f} / 100")
        category_column.metric("Performance category", category)

        st.progress(int(round(marks)))
        st.caption(f"Estimated score: {marks:.1f}%")


        st.markdown('<div id="recommendations"></div>', unsafe_allow_html=True)
        st.subheader("💡 Personalized recommendations")

        recommendations = make_recommendations(values)
        if recommendations:
            for item in recommendations:
                st.markdown(f"- {item}")
        else:
            st.success(
                "Your entered values meet the recommendation checks shown "
                "here. Keep up your consistent study habits."
            )

        st.subheader("Student input summary")
        summary_labels = {
            "Study_Hours": "Study hours per day",
            "Attendance": "Attendance (%)",
            "Previous_Exam_Marks": "Previous exam marks",
            "Assignment_Marks": "Assignment marks",
            "Quiz_Marks": "Quiz marks",
            "Internal_Marks": "Internal marks",
            "Practice_Tests": "Practice tests attempted",
            "Sleep_Hours": "Sleep hours per night",
            "Previous_Semester_GPA": "Previous semester GPA",
            "Class_Participation": "Class participation",
            "Study_Days_Per_Week": "Study days per week",
        }

        summary_table = pd.DataFrame(
            [
                {"Input": summary_labels[name], "Entered value": value}
                for name, value in values.items()
            ]
        )
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
else:
    st.info("Add the existing `models/model.pkl` file to enable predictions.")


# Analytics uses the actual CSV column names.
show_analytics(
    features=artifact.get("features") if artifact is not None else None
)


# Model details and importance
st.markdown('<div id="model"></div>', unsafe_allow_html=True)
st.header("🤖 Model")

if artifact is None:
    st.info("Model information is unavailable until `models/model.pkl` is present.")
else:
    model = artifact["model"]
    features = artifact["features"]
    estimator = get_estimator(model)
    model_name = get_model_name(artifact, estimator)
    target = artifact.get("target", "Final_Exam_Marks")

    st.subheader("Model information")
    model_column, target_column = st.columns(2)
    model_column.metric("Model", model_name)
    target_column.metric("Predicts", target)

    st.markdown("**Features used**")
    st.write(", ".join(features))

    

    st.subheader("Feature Importance")
    show_feature_importance(model, features)


