import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="KNN Classifier App",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("data/diabetes_prediction_dataset.csv")

# =====================================================
# ENCODE CATEGORICAL COLUMNS
# =====================================================

gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

smoking_map = {
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5
}

df['gender'] = df['gender'].map(gender_map)

df['smoking_history'] = df['smoking_history'].map(smoking_map)

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop('diabetes', axis=1)

y = df['diabetes']

# =====================================================
# LOAD MODEL & SCALER
# =====================================================

model = joblib.load("models/knn_classifier.pkl")

scaler = joblib.load("models/scaler.pkl")

# =====================================================
# TITLE
# =====================================================

st.title("🩺 Diabetes Prediction using KNN Classifier")

st.write("Machine Learning Classification Project")

# =====================================================
# DATASET PREVIEW
# =====================================================

st.subheader("Dataset Preview")

st.dataframe(df.head())

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("Enter Patient Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

age = st.sidebar.slider(
    "Age",
    1,
    100,
    30
)

bmi = st.sidebar.slider(
    "BMI",
    10.0,
    60.0,
    25.0
)

blood_glucose_level = st.sidebar.slider(
    "Blood Glucose Level",
    50,
    300,
    120
)

HbA1c_level = st.sidebar.slider(
    "HbA1c Level",
    3.0,
    15.0,
    5.5
)

# =====================================================
# FIXED VALUES
# =====================================================

hypertension = 0
heart_disease = 0
smoking_history = 0

# =====================================================
# INPUT DATAFRAME
# =====================================================

input_df = pd.DataFrame([{
    'gender': gender_map[gender],
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'smoking_history': smoking_history,
    'bmi': bmi,
    'HbA1c_level': HbA1c_level,
    'blood_glucose_level': blood_glucose_level
}])

# =====================================================
# SCALE INPUT
# =====================================================

input_scaled = scaler.transform(input_df)

# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(input_scaled)[0]

# =====================================================
# OUTPUT
# =====================================================

st.subheader("Prediction Result")

if prediction == 1:
    st.error("⚠️ Patient is likely Diabetic")
else:
    st.success("✅ Patient is Non-Diabetic")

# =====================================================
# MODEL PERFORMANCE
# =====================================================

X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)

accuracy = accuracy_score(y, y_pred)

st.subheader("Model Accuracy")

st.metric("Accuracy", f"{accuracy:.4f}")

# =====================================================
# CONFUSION MATRIX
# =====================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xlabel("Predicted")

ax.set_ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i, j], ha='center', va='center')

st.pyplot(fig)

# =====================================================
# DATASET SHAPE
# =====================================================

st.subheader("Dataset Shape")

st.write(df.shape)

# =====================================================
# STATISTICS
# =====================================================

st.subheader("Dataset Statistics")

st.dataframe(df.describe())