"""Streamlit interface for Exam Marks Prediction Using Machine Learning."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# Keep the existing model path and artifact structure.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pkl"
DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "student_exam_data.csv"

st.set_page_config(
    page_title="Exam Marks Prediction",
    page_icon="📘",
    layout="wide",
)


@st.cache_resource
def load_artifact():
    """Load the previously trained model. This app does not train or modify it."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    """Load the project dataset when it exists."""
    return pd.read_csv(DATASET_PATH)


def get_estimator(model):
    """Return the final estimator if the saved model is a scikit-learn Pipeline."""
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def get_model_name(artifact, estimator):
    """Use the saved model name when available; otherwise show the real class name."""
    saved_name = artifact.get("model_name")
    if saved_name:
        return saved_name

    return estimator.__class__.__name__


def get_performance_category(marks):
    """Map a predicted mark to the requested performance category."""
    if marks < 40:
        return "Needs Improvement"
    if marks < 60:
        return "Average"
    if marks < 75:
        return "Good"
    if marks < 90:
        return "Very Good"
    return "Excellent"


def get_recommendations(values):
    """Create recommendations based on this student's entered values."""
    recommendations = []

    if values["Study_Hours"] < 3:
        recommendations.append(
            "Your study time is below 3 hours per day. Try adding a short, "
            "focused study session to your routine."
        )

    if values["Attendance"] < 75:
        recommendations.append(
            "Your attendance is below 75%. Attending classes more regularly "
            "can help you keep up with lessons."
        )

    if values["Practice_Tests"] < 5:
        recommendations.append(
            "You have attempted fewer than 5 practice tests. Try attempting "
            "more practice questions or mock tests."
        )

    if values["Sleep_Hours"] < 6:
        recommendations.append(
            "Your reported sleep is below 6 hours per night. A more regular "
            "sleep schedule may support your study routine."
        )

    if values["Class_Participation"] <= 2:
        recommendations.append(
            "Your class participation rating is low. Consider asking or "
            "answering a question during class."
        )

    if values["Study_Days_Per_Week"] < 3:
        recommendations.append(
            "You study fewer than 3 days per week. Spreading study sessions "
            "across more days may help you practise consistently."
        )

    return recommendations


def show_feature_importance(model, features):
    """Display genuine feature importances or coefficients exposed by the model."""
    estimator = get_estimator(model)
    importance_kind = None
    importance_values = None

    if hasattr(estimator, "feature_importances_"):
        importance_values = np.asarray(estimator.feature_importances_).reshape(-1)
        importance_kind = "Feature importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)

        # For a multi-output estimator, show the mean absolute coefficient
        # across outputs. For a single-output model, show absolute coefficients.
        if coefficients.ndim > 1:
            importance_values = np.mean(np.abs(coefficients), axis=0)
        else:
            importance_values = np.abs(coefficients.reshape(-1))

        importance_kind = "Absolute coefficient"

    if importance_values is None or len(importance_values) != len(features):
        st.info("Feature importance is not available for the current model.")
        return

    importance_data = pd.DataFrame(
        {
            "Feature": features,
            "Importance": importance_values,
        }
    ).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=importance_data,
        x="Importance",
        y="Feature",
        color="#3977A8",
        ax=ax,
    )
    ax.set_title(f"Model {importance_kind}")
    ax.set_xlabel(importance_kind)
    ax.set_ylabel("Feature")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if importance_kind == "Absolute coefficient":
        st.caption(
            "The chart uses absolute values of the trained model's coefficients. "
            "They show model weights, not proof that a feature causes a mark to change."
        )
    else:
        st.caption(
            "These values come from the trained model. Feature importance does not "
            "show that a feature causes a mark to change."
        )


def show_analytics():
    """Display data summaries and charts using the actual project dataset columns."""
    st.title("📊 Analytics")

    if not DATASET_PATH.exists():
        st.info(
            "The dataset file was not found. Prediction and the other project "
            "sections can still be used."
        )
        return

    try:
        data = load_dataset()
    except Exception as error:
        st.info(f"The dataset could not be loaded: {error}")
        return

    st.subheader("Dataset overview")
    first, second = st.columns(2)
    first.metric("Rows", f"{data.shape[0]:,}")
    second.metric("Columns", f"{data.shape[1]:,}")

    st.subheader("Dataset preview")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

    st.subheader("Basic statistics")
    numeric_data = data.select_dtypes(include="number")
    if numeric_data.empty:
        st.info("There are no numeric columns available for summary statistics.")
    else:
        st.dataframe(
            numeric_data.describe().T.round(2),
            use_container_width=True,
        )

    # These names are the real columns in this project's dataset.
    chart_pairs = [
        ("Study_Hours", "Final_Exam_Marks", "Study Hours vs Exam Marks"),
        ("Attendance", "Final_Exam_Marks", "Attendance vs Exam Marks"),
        (
            "Previous_Exam_Marks",
            "Final_Exam_Marks",
            "Previous Marks vs Exam Marks",
        ),
    ]

    st.subheader("Academic factor charts")
    available_pairs = [
        pair for pair in chart_pairs
        if pair[0] in data.columns and pair[1] in data.columns
    ]

    if not available_pairs:
        st.info(
            "The dataset does not contain the expected columns for the "
            "study-hours, attendance, or previous-marks charts."
        )
    else:
        chart_columns = st.columns(len(available_pairs))
        for chart_column, (x_column, y_column, title) in zip(
            chart_columns, available_pairs
        ):
            with chart_column:
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.scatterplot(
                    data=data,
                    x=x_column,
                    y=y_column,
                    alpha=0.65,
                    color="#3977A8",
                    ax=ax,
                )
                ax.set_title(title)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("Correlation heatmap")
    if numeric_data.shape[1] < 2:
        st.info("At least two numeric columns are needed to create a heatmap.")
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            numeric_data.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
        )
        ax.set_title("Correlation between numeric dataset columns")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# Load the existing saved model once. A missing model does not prevent viewing
# Analytics or About.
artifact = None
model_error = None

if MODEL_PATH.exists():
    try:
        artifact = load_artifact()
    except Exception as error:
        model_error = str(error)


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "🏠 Home",
        "🎯 Prediction",
        "📊 Analytics",
        "🤖 Model Information",
        "ℹ️ About",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("B.Tech CSE (AIML) Minor Project")


if page == "🏠 Home":
    st.title("📘 Exam Marks Prediction Using Machine Learning")
    st.write(
        "Estimate a student's expected exam marks using academic and "
        "behavioral information."
    )

    st.info(
        "Predictions are machine-learning estimates based on the available "
        "training data. They are not guaranteed marks or official assessments."
    )

    st.subheader("Project sections")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            - **🎯 Prediction:** Enter student information and view an estimate.
            - **📊 Analytics:** Explore the project dataset and its relationships.
            """
        )

    with col2:
        st.markdown(
            """
            - **🤖 Model Information:** See the saved model details and available
              feature importance.
            - **ℹ️ About:** Read the project objective and technology stack.
            """
        )

    if artifact is not None:
        estimator = get_estimator(artifact["model"])
        st.success(f"Saved model loaded: **{get_model_name(artifact, estimator)}**")
    elif model_error:
        st.warning(f"The saved model could not be loaded: {model_error}")
    else:
        st.warning(
            "The saved model file was not found. Add `models/model.pkl` "
            "to enable predictions."
        )


elif page == "🎯 Prediction":
    st.title("🎯 Predict Exam Marks")
    st.write(
        "Enter the student's academic and study information, then select "
        "**Predict Exam Marks**."
    )

    if artifact is None:
        if model_error:
            st.error(f"The saved model could not be loaded: {model_error}")
        else:
            st.error(
                "The saved model file was not found. Ensure "
                "`models/model.pkl` exists."
            )
        st.stop()

    model = artifact["model"]
    features = artifact["features"]

    with st.form("prediction_form"):
        st.subheader("Student information")

        left, middle, right = st.columns(3)

        with left:
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

        with middle:
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

        with right:
            gpa = st.number_input(
                "Previous semester GPA (0–10)",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1,
            )
            class_participation = st.slider(
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
        input_values = {
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Exam_Marks": previous_marks,
            "Assignment_Marks": assignment_marks,
            "Quiz_Marks": quiz_marks,
            "Internal_Marks": internal_marks,
            "Practice_Tests": practice_tests,
            "Sleep_Hours": sleep_hours,
            "Previous_Semester_GPA": gpa,
            "Class_Participation": class_participation,
            "Study_Days_Per_Week": study_days,
        }

        unsupported_features = [
            feature for feature in features if feature not in input_values
        ]

        if unsupported_features:
            st.error(
                "The saved model expects input features that are not available "
                f"in this form: {', '.join(unsupported_features)}"
            )
        else:
            # Preserve the exact feature order expected by the saved model.
            input_row = pd.DataFrame(
                [[input_values[feature] for feature in features]],
                columns=features,
            )

            try:
                prediction = float(np.asarray(model.predict(input_row)).reshape(-1)[0])
                prediction = max(0.0, min(100.0, prediction))

                st.session_state["prediction_result"] = {
                    "marks": prediction,
                    "values": input_values,
                }
            except Exception as error:
                st.error(f"Prediction could not be generated: {error}")

    result = st.session_state.get("prediction_result")

    if result:
        predicted_marks = result["marks"]
        values = result["values"]
        category = get_performance_category(predicted_marks)

        st.divider()
        st.subheader("Prediction result")

        mark_column, category_column = st.columns(2)
        mark_column.metric("Predicted marks", f"{predicted_marks:.1f} / 100")
        category_column.metric("Performance category", category)

        st.progress(int(round(predicted_marks)))
        st.caption(f"Estimated score: {predicted_marks:.1f}%")


        st.subheader("Personalized recommendations")
        recommendations = get_recommendations(values)

        if recommendations:
            for recommendation in recommendations:
                st.write(f"- {recommendation}")
        else:
            st.success(
                "Your entered values meet the recommendation checks in this "
                "demo. Keep up your consistent study habits."
            )

        st.subheader("Your input summary")
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

        summary = pd.DataFrame(
            [
                {
                    "Input": summary_labels[name],
                    "Entered value": value,
                }
                for name, value in values.items()
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


elif page == "📊 Analytics":
    show_analytics()


elif page == "🤖 Model Information":
    st.title("🤖 Model Information")

    if artifact is None:
        if model_error:
            st.error(f"The saved model could not be loaded: {model_error}")
        else:
            st.info(
                "The saved model file was not found. Model details and feature "
                "importance will appear after `models/model.pkl` is available."
            )
    else:
        model = artifact["model"]
        estimator = get_estimator(model)
        features = artifact["features"]
        target = artifact.get("target", "Final_Exam_Marks")
        model_name = get_model_name(artifact, estimator)

        st.subheader("Saved model")
        st.write(f"**Model name:** {model_name}")
        st.write(f"**Model class:** `{estimator.__class__.__name__}`")
        st.write(f"**Predicts:** `{target}`")

        st.subheader("Features used by the model")
        st.write(", ".join(features))

        st.subheader("How the prediction works")
       

        st.subheader("Feature Importance")
        show_feature_importance(model, features)


elif page == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.subheader("Exam Marks Prediction Using Machine Learning")
    st.write(
        "**Objective:** Predict a student's expected exam marks using "
        "academic and behavioral factors."
    )

    st.markdown(
        """
        **Technologies**

        Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit,
        and Joblib.
        """
    )
