import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scaler = joblib.load("scaler.pkl")

# Configure the page
st.set_page_config(page_title="Telco Churn Insight", layout="wide")

# Custom style for a sleek fixed-height layout
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 2rem;
        }
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            padding: 0.6em 1.5em;
            border-radius: 10px;
            font-size: 16px;
        }
        .stMetric label {
            font-size: 1.2rem;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-header'>📊 Telco Churn Insight</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Professional Dashboard for Customer Churn Prediction</div>", unsafe_allow_html=True)

# Full page dashboard layout (single screen view)
input_col, summary_col, result_col = st.columns([2.5, 1.2, 1.3], gap="large")

with input_col:
    st.subheader("🧍 Customer Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    tenure = st.slider("Tenure (in months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    stream_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charge = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charge = st.number_input("Total Charges ($)", 0.0, 10000.0, 2500.0)

user_data = pd.DataFrame([{ 
    'gender': gender,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet,
    'OnlineSecurity': online_sec,
    'OnlineBackup': backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': stream_tv,
    'StreamingMovies': stream_movies,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly_charge,
    'TotalCharges': total_charge
}])

with summary_col:
    st.subheader("🔎 Summary")
    st.dataframe(user_data.T, use_container_width=True, height=500)

with result_col:
    st.subheader("📈 Prediction")
    def preprocess_input(df):
        df_encoded = pd.get_dummies(df)
        for col in set(model_columns) - set(df_encoded.columns):
            df_encoded[col] = 0
        df_encoded = df_encoded[model_columns]
        df_encoded[["MonthlyCharges", "TotalCharges"]] = scaler.transform(df_encoded[["MonthlyCharges", "TotalCharges"]])
        return df_encoded

    if st.button("🔍 Predict Now"):
        processed = preprocess_input(user_data)
        prediction = model.predict(processed)
        prob = model.predict_proba(processed)[0][1]

        if prediction[0] == 1:
            st.error("❌ Customer likely to churn.")
        else:
            st.success("✅ Customer likely to stay.")

        st.metric(label="Churn Probability", value=f"{prob:.2%}")
        st.progress(prob)

        st.markdown("---")
        st.markdown("### 💡 Insight")
        st.info("Adjust profile inputs on the left to explore churn scenarios.")
