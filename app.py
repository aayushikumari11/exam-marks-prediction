"""Simple Streamlit interface for the exam-mark estimate."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.pkl"
st.set_page_config(page_title="Exam Marks Prediction", page_icon="📘", layout="wide")

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

st.title("📘 Exam Marks Prediction Using Machine Learning")
st.write("Estimate a student's final exam marks from study habits and academic indicators.")
st.info("This is an ML-based estimate from a synthetic teaching dataset. It is not a guarantee or an official assessment.")

if not MODEL_PATH.exists():
    st.error("Model file is missing. From the project folder, run: `python train_model.py`")
    st.stop()
artifact = load_artifact()
model, features = artifact["model"], artifact["features"]

with st.form("prediction_form"):
    st.subheader("Student information")
    left, middle, right = st.columns(3)
    with left:
        study_hours = st.number_input("Study hours per day", 0.0, 16.0, 4.0, 0.5)
        attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
        previous = st.number_input("Previous exam marks (0–100)", 0.0, 100.0, 65.0, 1.0)
        assignment = st.number_input("Assignment marks (0–100)", 0.0, 100.0, 70.0, 1.0)
    with middle:
        quiz = st.number_input("Quiz marks (0–100)", 0.0, 100.0, 68.0, 1.0)
        internal = st.number_input("Internal marks (0–100)", 0.0, 100.0, 65.0, 1.0)
        practice = st.number_input("Practice tests attempted", 0, 30, 5, 1)
        sleep = st.number_input("Sleep hours per night", 0.0, 16.0, 7.0, 0.5)
    with right:
        gpa = st.number_input("Previous semester GPA (0–10)", 0.0, 10.0, 7.0, 0.1)
        participation = st.slider("Class participation (1 = low, 5 = high)", 1, 5, 3)
        days = st.number_input("Study days per week", 0, 7, 5, 1)
    submitted = st.form_submit_button("Predict Exam Marks", type="primary")

if submitted:
    values = [study_hours, attendance, previous, assignment, quiz, internal,
              practice, sleep, gpa, participation, days]
    row = pd.DataFrame([dict(zip(features, values))])
    prediction = float(max(0, min(100, model.predict(row)[0])))
    if prediction < 40: category = "Needs Improvement"
    elif prediction < 60: category = "Average"
    elif prediction < 75: category = "Good"
    elif prediction < 90: category = "Very Good"
    else: category = "Excellent"
    st.subheader("Estimated result")
    a, b = st.columns(2)
    a.metric("Predicted marks", f"{prediction:.1f} / 100")
    b.metric("Expected performance category", category)
    st.success(f"Based on the entered academic information, the predicted exam score is {prediction:.1f} marks.")

st.divider()
st.subheader("About the model")
st.write(f"Selected from five regressors using the lowest held-out test RMSE: **{artifact['model_name']}**.")
if artifact.get("metrics"):
    st.dataframe(pd.DataFrame(artifact["metrics"]).round(3), use_container_width=True, hide_index=True)
st.caption("Inputs are indicators, not causes. Real marks also depend on teaching, exam difficulty, health, and many other factors.")
st.caption("B.Tech CSE (AIML) Minor Project • Educational use")
