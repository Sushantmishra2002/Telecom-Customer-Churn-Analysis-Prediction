from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0"
)

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(data: CustomerData):

    input_data = {
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges
    }

    prediction = predict(input_data)

    if prediction[0] == 1:
        result = "Customer likely to churn"
    else:
        result = "Customer likely to stay"

    return {"prediction": result}