import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Title & Sidebar
# -------------------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("House Price Prediction")

st.sidebar.header("About App")
st.sidebar.write("""
This app predicts house prices based on your input data.
- Upload a CSV file to predict prices for multiple houses.
- Or enter house features manually to get a single prediction.
""")

# Provide download link for train.csv
train_csv_path = "train.csv"
if os.path.exists(train_csv_path):
    st.sidebar.download_button(
        label="Download train.csv",
        data=open(train_csv_path, "rb"),
        file_name="train.csv",
        mime="text/csv"
    )
else:
    st.sidebar.warning("train.csv not found on server.")

# -------------------------------
# Load Model
# -------------------------------
model_path = "linear_regression_pipeline.joblib"
model = joblib.load(model_path)

# -------------------------------
# Tabs for Input Options
# -------------------------------
tab1, tab2 = st.tabs(["Upload CSV for Prediction", "Manual Input Prediction"])

# -------------------------------
# Tab 1: CSV Upload
# -------------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())

            if st.button("Predict Prices"):
                predictions = model.predict(df)
                df["PredictedPrice"] = predictions
                st.subheader("Predicted Prices")
                st.dataframe(df)

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

# -------------------------------
# Tab 2: Manual Input
# -------------------------------
with tab2:
    st.subheader("Enter House Features Manually")
    
    # Example: numeric features
    MSSubClass = st.number_input("MSSubClass", min_value=20, max_value=190, value=60)
    LotFrontage = st.number_input("LotFrontage", value=70.0)
    LotArea = st.number_input("LotArea", value=8450)
    YearBuilt = st.number_input("YearBuilt", min_value=1872, max_value=2025, value=2008)
    
    # Example: categorical features
    MSZoning = st.selectbox("MSZoning", ["RL", "RM", "FV", "RH", "C (all)"])
    Street = st.selectbox("Street", ["Pave", "Grvl"])
    Alley = st.selectbox("Alley", ["NA", "Grvl", "Pave"])
    
    if st.button("Predict Price"):
        try:
            # Construct a single-row dataframe
            input_data = pd.DataFrame({
                "MSSubClass": [MSSubClass],
                "LotFrontage": [LotFrontage],
                "LotArea": [LotArea],
                "YearBuilt": [YearBuilt],
                "MSZoning": [MSZoning],
                "Street": [Street],
                "Alley": [Alley]
            })
            
            price = model.predict(input_data)[0]
            st.success(f"Predicted House Price: ${price:,.2f}")
        except Exception as e:
            st.error(f"Error predicting price: {e}")