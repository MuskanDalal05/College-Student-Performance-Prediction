import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 College Student Performance Measurement System")
st.write(
    "Enter a student's academic details, choose a machine-learning model, "
    "and predict the student's performance level."
)

DATA_PATH = "college_student_performance_dataset.csv"

FEATURES = [
    "Attendance_Percentage",
    "Study_Hours_Per_Day",
    "Previous_Semester_Marks",
    "Internal_Assessment_Marks",
    "Assignment_Score",
    "Extracurricular_Score_Out_of_10",
    "Class_Participation_Rating_1_to_5",
    "Sleep_Hours_Per_Night",
]
TARGET = "Performance_Level"
CLASSES = ["Low", "Average", "High"]

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = data.dropna(subset=required).copy()
    data[TARGET] = data[TARGET].astype(str)
    return data

@st.cache_resource
def train_models():
    data = load_data()

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = data[TARGET].astype(str)

    # Remove any rows with missing or invalid feature values
    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_rows].astype("float64")
    y = y.loc[valid_rows].astype(str) 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced"
            ))
        ]),
        
        "ANN": Pipeline([
         ("scaler", StandardScaler()),
         ("model", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1500,
        early_stopping=False,
        random_state=42
        ))
       ]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, pred),
            "y_test": y_test,
            "predictions": pred,
            "report": classification_report(
                y_test, pred, labels=CLASSES, output_dict=True, zero_division=0
            ),
            "confusion": confusion_matrix(y_test, pred, labels=CLASSES),
        }
    return data, results

try:
    data, results = train_models()

except Exception:
    import traceback
    st.error("Full error details:")
    st.code(traceback.format_exc())
    st.stop()    

with st.sidebar:
    st.header("Choose model")
    selected_model = st.selectbox("Machine-learning model", ["Decision Tree", "Random Forest", "ANN"])
    st.caption("All models are trained on the same training split for a fair comparison.")
    st.divider()
    st.subheader("Dataset")
    st.write(f"Records: **{len(data)}**")
    st.write(f"Input features: **{len(FEATURES)}**")

st.subheader("1. Enter Student Details")
with st.form("student_form"):
    c1, c2 = st.columns(2)
    with c1:
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=18.0, value=3.0, step=0.5)
        previous_marks = st.number_input("Previous semester marks (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        internal_marks = st.number_input("Internal assessment marks (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    with c2:
        assignment_score = st.number_input("Assignment score (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        extracurricular = st.number_input("Extracurricular score (out of 10)", min_value=0, max_value=10, value=5, step=1)
        participation = st.slider("Class participation (1 = low, 5 = high)", min_value=1, max_value=5, value=3)
        sleep_hours = st.number_input("Sleep hours per night", min_value=0.0, max_value=16.0, value=7.0, step=0.5)

    submitted = st.form_submit_button("Predict Student Performance", type="primary")

if submitted:
    student = pd.DataFrame([{
        "Attendance_Percentage": attendance,
        "Study_Hours_Per_Day": study_hours,
        "Previous_Semester_Marks": previous_marks,
        "Internal_Assessment_Marks": internal_marks,
        "Assignment_Score": assignment_score,
        "Extracurricular_Score_Out_of_10": extracurricular,
        "Class_Participation_Rating_1_to_5": participation,
        "Sleep_Hours_Per_Night": sleep_hours,
    }], columns=FEATURES)

    chosen = results[selected_model]
    prediction = chosen["model"].predict(student)[0]
    st.subheader("2. Prediction Result")
    if prediction == "High":
        st.success(f"Predicted performance level: **{prediction}**")
    elif prediction == "Average":
        st.warning(f"Predicted performance level: **{prediction}**")
    else:
        st.error(f"Predicted performance level: **{prediction}**")

    st.caption(
        "This is an educational prediction based on a synthetic practice dataset. "
        "It is not an official academic assessment and should not be used to make high-stakes decisions."
    )

st.divider()
st.subheader("3. Compare Model Performance")
comparison = pd.DataFrame({
    "Model": list(results.keys()),
    "Test Accuracy (%)": [round(results[n]["accuracy"] * 100, 2) for n in results],
})
st.dataframe(comparison, hide_index=True, use_container_width=True)
st.bar_chart(comparison.set_index("Model"))

st.subheader(f"4. Evaluation Details — {selected_model}")
chosen = results[selected_model]
m1, m2 = st.columns(2)
with m1:
    st.metric("Test accuracy", f"{chosen['accuracy'] * 100:.2f}%")
    report_df = pd.DataFrame(chosen["report"]).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)
with m2:
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=chosen["confusion"], display_labels=CLASSES
    )
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Preview dataset"):
    st.dataframe(data.head(20), use_container_width=True)

with st.expander("How to interpret the models"):
    st.markdown("""
    - **Decision Tree:** learns a sequence of decision rules.
    - **Random Forest:** combines predictions from many decision trees.
    - **ANN:** a neural network that learns patterns through connected layers.

    Accuracy is the percentage of test examples classified correctly. Also review
    precision, recall, F1-score, and the confusion matrix rather than relying only
    on accuracy. The data in this starter project is synthetic, so the metrics are
    only for demonstrating the workflow.
    """)
