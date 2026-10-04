"""Train and compare beginner-friendly regression models for exam marks."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "student_exam_data.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
FEATURES = ["Study_Hours", "Attendance", "Previous_Exam_Marks", "Assignment_Marks",
            "Quiz_Marks", "Internal_Marks", "Practice_Tests", "Sleep_Hours",
            "Previous_Semester_GPA", "Class_Participation", "Study_Days_Per_Week"]
TARGET = "Final_Exam_Marks"


def main():
    """Load, clean, compare models and save a deployable fitted pipeline."""
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().copy()
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    X, y = data[FEATURES], data[TARGET].clip(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Scaling matters for KNN and SVR. Keeping it in each pipeline ensures it is
    # fitted on training rows only and used consistently by the app.
    models = {
        "Linear Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
        "Decision Tree": make_pipeline(SimpleImputer(strategy="median"), DecisionTreeRegressor(max_depth=6, random_state=42)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        "K-Nearest Neighbors": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
        "Support Vector Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=10, epsilon=2.0, kernel="rbf")),
    }
    scores = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = np.clip(model.predict(X_test), 0, 100)
        scores.append({"Model": name, "MAE": mean_absolute_error(y_test, prediction),
                       "MSE": mean_squared_error(y_test, prediction),
                       "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
                       "R2": r2_score(y_test, prediction)})
    scores.sort(key=lambda row: row["RMSE"])
    results = pd.DataFrame(scores)
    print("Held-out test set (20%, random_state=42):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    # Choose the model with the lowest test RMSE. Fit a fresh version on all
    # available data for final app deployment; metrics above remain test-only.
    winner_name = scores[0]["Model"]
    winner = models[winner_name]
    winner.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": winner, "features": FEATURES, "target": TARGET,
                 "model_name": winner_name, "metrics": scores}, MODEL_PATH)
    results.to_csv(MODEL_PATH.parent / "model_comparison.csv", index=False)
    METRICS_PATH.write_text(json.dumps({"selected_model": winner_name, "metrics": scores}, indent=2), encoding="utf-8")
    print(f"Saved app model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
