import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Load feature columns used during training
feature_columns = joblib.load("models/feature_columns.pkl")


def predict(data):

    # Convert input data into dataframe
    df = pd.DataFrame([data])

    # Apply one-hot encoding
    df = pd.get_dummies(df)

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data
    df = df[feature_columns]

    # Predict
    prediction = model.predict(df)

    return prediction