import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_house_prices_pca_model.joblib")

model = load_model()

# ---------------------------------------------------
# Load training data to rebuild preprocessing pipeline
# ---------------------------------------------------
@st.cache_resource
def load_train_info():
    train = pd.read_csv("train.csv")
    
    numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols.remove('SalePrice')

    categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # PCA model
    pca = PCA(n_components=0.95, random_state=42)

    return train, numeric_cols, categorical_cols, preprocessor, pca

train, numeric_cols, categorical_cols, preprocessor, pca = load_train_info()

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    color: #1A5276;
    margin-bottom: 5px;
}
.sub-title {
    font-size: 18px;
    text-align: center;
    color: #555;
    margin-bottom: 40px;
}
.pred-box {
    padding: 25px;
    background: #F8F9F9;
    border-radius: 18px;
    box-shadow: 0 0 12px rgba(0,0,0,0.12);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>House Price Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>XGBoost + PCA + Full Preprocessing Pipeline</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.header("Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ---------------------------------------------------
# Single Prediction Input
# ---------------------------------------------------
st.header("Single House Prediction")

cols = st.columns(3)

input_data = {}

for i, col in enumerate(numeric_cols[:9]):
    with cols[i % 3]:
        input_data[col] = st.number_input(col, value=float(train[col].median()))

# Fill missing numeric columns
for col in numeric_cols:
    if col not in input_data:
        input_data[col] = train[col].median()

# Fill categorical columns with mode
for col in categorical_cols:
    input_data[col] = train[col].mode()[0]

# ---------------------------------------------------
# Single Predict Button
# ---------------------------------------------------
if st.button("Predict Price"):
    df = pd.DataFrame([input_data])

    # Full preprocessing
    X_scaled = preprocessor.fit(train.drop("SalePrice", axis=1)).transform(df)
    X_pca = pca.fit(preprocessor.transform(train.drop("SalePrice", axis=1))).transform(X_scaled)

    prediction = model.predict(X_pca)[0]

    st.markdown(
        f"<div class='pred-box'><h3>Estimated Price: <b>{int(prediction):,}$</b></h3></div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# Batch Prediction
# ---------------------------------------------------
st.header("Batch CSV Prediction")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    if st.button("Predict CSV"):
        X_scaled = preprocessor.fit(train.drop("SalePrice", axis=1)).transform(data)
        X_pca = pca.fit(preprocessor.transform(train.drop("SalePrice", axis=1))).transform(X_scaled)

        preds = model.predict(X_pca)

        result = data.copy()
        result["PredictedPrice"] = preds

        st.write("### Predictions")
        st.dataframe(result.head())

        # Download file
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, "predictions.csv")