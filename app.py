import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Page config & Title
# -------------------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("House Price Prediction")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("About App")
st.sidebar.info("""
This app predicts house prices based on Kaggle dataset features.
- Upload your test CSV file to predict prices for multiple houses.
- The app will add a 'PredictedPrice' column.
""")

# Download link for test CSV
test_csv_path = "test.csv"
if os.path.exists(test_csv_path):
    st.sidebar.warning("You can download the test CSV template below")
    st.sidebar.download_button(
        label="Download test.csv",
        data=open(test_csv_path, "rb"),
        file_name="test.csv",
        mime="text/csv"
    )
else:
    st.sidebar.warning("test.csv not found on server.")

# -------------------------------
# Load model
# -------------------------------
model_path = "linear_regression_pipeline.joblib"
model = joblib.load(model_path)

# -------------------------------
# CSV Upload for Prediction
# -------------------------------
st.subheader("Upload your test CSV file for prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        if st.button("Predict Prices"):
            predictions = model.predict(df)
            df["PredictedPrice"] = predictions

            # If Id column exists, sort by it
            if "Id" in df.columns:
                df.sort_values("Id", inplace=True)

            st.subheader("Predicted Prices")
            st.dataframe(df[["Id", "PredictedPrice"]] if "Id" in df.columns else df)

            # Download button
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predicted_prices.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")