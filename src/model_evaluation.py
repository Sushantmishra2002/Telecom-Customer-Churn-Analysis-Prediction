import joblib
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from data_preprocessing import load_data, clean_data
from feature_engineering import encode_features
from sklearn.model_selection import train_test_split

model = joblib.load("models/churn_model.pkl")

df = load_data("data/telco_churn.csv")

df = clean_data(df)

df = encode_features(df)

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))