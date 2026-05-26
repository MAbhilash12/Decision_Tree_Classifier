import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

from sklearn.tree import plot_tree

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Decision Tree Classifier",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("🌳 Decision Tree Classifier App")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/heart.csv")

# ==========================================
# LOAD MODEL & SCALER
# ==========================================

model = joblib.load(
    "models/decision_tree_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================================
# DATASET HEAD
# ==========================================

st.subheader("📊 Dataset Head")

st.dataframe(df.head())

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider(
    "Age",
    20,
    80,
    40
)

sex = st.sidebar.selectbox(
    "Sex",
    [0, 1]
)

cp = st.sidebar.slider(
    "Chest Pain Type",
    0,
    3,
    1
)

trestbps = st.sidebar.slider(
    "Resting Blood Pressure",
    80,
    200,
    120
)

chol = st.sidebar.slider(
    "Cholesterol",
    100,
    600,
    200
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

restecg = st.sidebar.slider(
    "Rest ECG",
    0,
    2,
    1
)

thalach = st.sidebar.slider(
    "Max Heart Rate",
    60,
    220,
    150
)

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.sidebar.slider(
    "Old Peak",
    0.0,
    6.0,
    1.0
)

slope = st.sidebar.slider(
    "Slope",
    0,
    2,
    1
)

ca = st.sidebar.slider(
    "CA",
    0,
    4,
    0
)

thal = st.sidebar.slider(
    "Thal",
    0,
    3,
    1
)

# ==========================================
# INPUT ARRAY
# ==========================================

input_data = np.array([[
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]])

# ==========================================
# SCALE INPUT
# ==========================================

input_scaled = scaler.transform(
    input_data
)

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.sidebar.button("Predict"):

    prediction = model.predict(
        input_scaled
    )

    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:

        st.success(
            "Person Has Heart Disease"
        )

    else:

        st.error(
            "Person Does Not Have Heart Disease"
        )

# ==========================================
# MODEL ACCURACY
# ==========================================

X = df.drop("target", axis=1)

y = df["target"]

X_scaled = scaler.transform(X)

preds = model.predict(X_scaled)

accuracy = accuracy_score(
    y,
    preds
)

st.subheader("📈 Model Accuracy")

st.metric(
    label="Accuracy",
    value=f"{accuracy*100:.2f}%"
)

# ==========================================
# TARGET DISTRIBUTION
# ==========================================

st.subheader("📊 Target Distribution")

fig1, ax1 = plt.subplots()

sns.countplot(
    x="target",
    data=df,
    ax=ax1
)

st.pyplot(fig1)

# ==========================================
# CONFUSION MATRIX
# ==========================================

st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(
    y,
    preds
)

fig2, ax2 = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax2
)

ax2.set_xlabel("Predicted")

ax2.set_ylabel("Actual")

st.pyplot(fig2)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.subheader("⭐ Feature Importance")

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

fig3, ax3 = plt.subplots(figsize=(10,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_df,
    ax=ax3
)

st.pyplot(fig3)

# ==========================================
# DECISION TREE VISUALIZATION
# ==========================================

st.subheader("🌳 Decision Tree Visualization")

fig4, ax4 = plt.subplots(
    figsize=(20,10)
)

plot_tree(
    model,
    feature_names=X.columns,
    class_names=[
        "No Disease",
        "Disease"
    ],
    filled=True,
    ax=ax4
)

st.pyplot(fig4)
