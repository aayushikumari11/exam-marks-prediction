# Exam Marks Prediction Using Machine Learning

A beginner-friendly B.Tech CSE (AIML) minor project that estimates final exam marks from academic and study-habit indicators. The project compares five regression models and provides a small Streamlit prediction interface.

## Project architecture and workflow

```text
CSV dataset → inspect and clean → explore and select features → 80/20 split
           → train five regressors → compare held-out metrics → save winner
           → Streamlit form → estimated marks and category
```

```text
exam marks prediction/
├── dataset/student_exam_data.csv       # 800 synthetic student records
├── notebooks/exam_marks_prediction.ipynb # Guided analysis, charts, models
├── models/model.pkl                    # Saved model pipeline for the app
├── models/model_comparison.csv         # Test-set metrics (created by training)
├── models/metrics.json                 # Test-set metrics and selected model
├── app.py                              # Streamlit interface
├── train_model.py                      # Cleaning, comparison, artifact creation
├── requirements.txt                    # Python packages
├── README.md                           # Setup and project guide
└── report/project_report.md            # Report and viva questions/answers
```

## Problem statement and objectives

Students and instructors may want an early, approximate view of performance based on available academic indicators. The project demonstrates how tabular data can be explored and used to estimate a continuous mark. Objectives are to prepare a transparent dataset, compare common regression algorithms fairly, and make a simple interface that communicates uncertainty.

## Dataset

`dataset/student_exam_data.csv` contains 800 **synthetic** records made for learning. Each row represents one fictional student. `Student_ID` is an identifier and is excluded from training. Inputs are study hours/day, attendance (%), previous exam marks, assignment and quiz marks, internal marks, practice tests attempted, sleep hours/night, previous semester GPA (0–10), class participation (1–5), and study days/week. `Final_Exam_Marks` (0–100) is the target. The generated target combines these signals with noise and is clipped to the mark range. It is not collected from a college and should not be used for real student decisions. See the report for field definitions and limitations.

## Technologies and algorithms

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, Jupyter, and Streamlit. The training script compares Linear Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Regression. Median imputation is inside each model pipeline. Scaling is used for Linear Regression, KNN, and SVR; trees use original units. Model choice is based on lowest test RMSE, with the 20% test split held out during comparison. Run the script to generate the exact comparison for your environment.

## Install and run

From this project directory, use Python 3.10 or newer (a Python version supported by the installed scikit-learn release):

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit. To explore the notebook, run `jupyter notebook` and open `notebooks/exam_marks_prediction.ipynb`. Re-run `python train_model.py` after changing the CSV; it regenerates `models/model.pkl`, `models/model_comparison.csv`, and `models/metrics.json`.

## Model evaluation

The script reports MAE, MSE, RMSE, and R² on the same held-out 20% split for every model and then fits the lowest-RMSE model on all rows for deployment. MAE/RMSE are mark errors (lower is better); MSE squares errors; R² describes variance explained relative to predicting the test mean (closer to 1 is better, and it can be negative). The notebook adds an actual-vs-predicted chart, residual plot, model comparison, correlation heatmap, and tree feature importance. Results measure performance on this synthetic data only.

## Screenshots

Add screenshots after running the app and notebook:

- `screenshots/streamlit-home.png` — input form
- `screenshots/streamlit-prediction.png` — sample estimate
- `screenshots/model-comparison.png` — evaluation results

## Limitations and future scope

The dataset is synthetic, modest in size, and encodes a simplified relationship. Performance can be optimistic because test and training rows share the same generator. The estimate is uncertain and omits factors such as subject difficulty, teaching, health, and personal circumstances. Future work could use consented larger real datasets, personalized study suggestions, an early-warning view, ERP/LMS integration with privacy controls, and explainability methods.

## Files

- `train_model.py`: reproducible split, preprocessing pipelines, five regressors, metrics, and model serialization.
- `app.py`: input form, prediction, category, model metrics, and estimate caveat.
- `notebooks/exam_marks_prediction.ipynb`: step-by-step data exploration, plots, comparison, and evaluation.
- `dataset/student_exam_data.csv`: synthetic sample data.
- `models/`: model and generated score summaries.
- `report/project_report.md`: ten-chapter report, references, and 35 viva Q&As.

## Contributors

Add the student/team names, enrollment numbers, department, college, and guide name here before submission.

## References

- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- pandas documentation: https://pandas.pydata.org/docs/
- Matplotlib documentation: https://matplotlib.org/stable/
- seaborn documentation: https://seaborn.pydata.org/
- Cortes, KDD Cup 1998 Data: https://www.kdd.org/kdd-cup/view/kdd-cup-1998 (background on educational prediction datasets; not the data used here)
