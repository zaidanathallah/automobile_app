
import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Config
st.set_page_config(page_title="Automobile Price Prediction 🚗", layout="wide")

# Sidebar
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Automobile_data.csv")

# Main Title
st.title("🚗 Automobile Price Prediction & Explainable AI")

# EDA
if st.checkbox("Show Data"):
    st.write(df.head())

if st.checkbox("Show Missing Values"):
    st.write(df.isnull().sum())

if st.checkbox("Show Correlation Matrix"):
    st.write(sns.heatmap(df.corr(), annot=True))
    st.pyplot()

# Features and Target
target = st.sidebar.selectbox("Select Target Variable", df.columns, index=len(df.columns)-1)
features = st.sidebar.multiselect("Select Feature Variables", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"])

if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Ridge Regression":
    alpha = st.sidebar.slider("Alpha (Ridge)", 0.1, 10.0, 1.0)
    model = Ridge(alpha=alpha)
elif model_choice == "Lasso Regression":
    alpha = st.sidebar.slider("Alpha (Lasso)", 0.1, 10.0, 1.0)
    model = Lasso(alpha=alpha)
else:
    n_estimators = st.sidebar.slider("n_estimators", 100, 1000, 100)
    max_depth = st.sidebar.slider("max_depth", 1, 20, 5)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# Train
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Model Performance")
st.write("R2 Score:", r2_score(y_test, y_pred))
st.write("MSE:", mean_squared_error(y_test, y_pred))

# Save Model
if st.button("Save Model"):
    with open("saved_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model Saved!")

# Feature Importance + SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

st.subheader("Feature Importance (SHAP)")
shap.summary_plot(shap_values, X_train)
st.pyplot(bbox_inches='tight')

# SHAP Dependence Plot
feature_dependence = st.selectbox("Select Feature for Dependence Plot", features)
shap.dependence_plot(feature_dependence, shap_values.values, X_train)
st.pyplot(bbox_inches='tight')

# Manual Predict
st.subheader("Predict Price Manually")
manual_input = {}
for feature in features:
    manual_input[feature] = st.number_input(f"Input {feature}", value=float(X[feature].mean()))

if st.button("Predict Price"):
    manual_df = pd.DataFrame([manual_input])
    manual_pred = model.predict(manual_df)
    st.success(f"Predicted Price: {manual_pred[0]:,.2f}")
