import joblib
import matplotlib.pyplot as plt
import streamlit as st
import sys
import os

# Fix module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict

st.title("Customer Churn Prediction System")

st.write("Enter customer information to predict churn risk")

gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider("Tenure (months)", 0, 72)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

total_charges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict Churn"):

    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "InternetService": internet_service,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    result = predict(data)

    if result[0] == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

st.subheader("Model Feature Importance")

model = joblib.load("models/churn_model.pkl")

features = joblib.load("models/feature_columns.pkl")

importances = model.feature_importances_

fig, ax = plt.subplots()

ax.barh(features[:10], importances[:10])

st.pyplot(fig)