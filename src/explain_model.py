import shap
import joblib
import pandas as pd

from data_preprocessing import load_data, clean_data
from feature_engineering import encode_features

# Load model
model = joblib.load("models/churn_model.pkl")

# Load data
df = load_data("data/telco_churn.csv")

df = clean_data(df)

df = encode_features(df)

X = df.drop("Churn_Yes", axis=1)

# Create explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Show summary plot
shap.summary_plot(shap_values, X)