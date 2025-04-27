import joblib
import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
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
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax)
    st.pyplot(fig)

# Features and Target
# Features and Target
target = st.sidebar.selectbox("Select Target Variable", df.columns, index=len(df.columns)-1)
features = st.sidebar.multiselect("Select Feature Variables", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

X = df[features]
y = df[target]

# Convert target to numeric
y = pd.to_numeric(y, errors='coerce')

# Drop missing target
X = X[~y.isna()]
y = y.dropna()

# Handle categorical variables
X = pd.get_dummies(X)

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
    os.makedirs('models', exist_ok=True)
    with open("models/saved_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model Saved to models/ folder!")

# Feature Importance + SHAP
st.subheader("Feature Importance (SHAP)")

# Select SHAP Explainer type
if model_choice == "Random Forest":
    explainer = shap.TreeExplainer(model)
else:
    explainer = shap.Explainer(model.predict, X_train)

shap_values = explainer(X_train)

fig_shap = plt.figure()
shap.summary_plot(shap_values, X_train, show=False)
st.pyplot(fig_shap)

# SHAP Dependence Plot
feature_dependence = st.selectbox("Select Feature for Dependence Plot", X_train.columns)
fig_dep = plt.figure()
shap.dependence_plot(feature_dependence, shap_values.values, X_train, show=False)
st.pyplot(fig_dep)

# Manual Predict
st.subheader("Predict Price Manually")
manual_input = {}
for feature in features:
    manual_input[feature] = st.number_input(f"Input {feature}", value=float(df[feature].mean()))

if st.button("Predict Price"):
    manual_df = pd.DataFrame([manual_input])
    manual_df = pd.get_dummies(manual_df)
    manual_df = manual_df.reindex(columns=X_train.columns, fill_value=0)  # Important!
    manual_pred = model.predict(manual_df)
    st.success(f"Predicted Price: {manual_pred[0]:,.2f}")
