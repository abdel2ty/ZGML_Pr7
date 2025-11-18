import streamlit as st
import pandas as pd
import joblib

st.title("House Price Prediction")

model = joblib.load("linear_regression_pipeline.joblib")

uploaded = st.file_uploader("Upload Kaggle test.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    preds = model.predict(df)
    df["PredictedPrice"] = preds
    st.write(df)