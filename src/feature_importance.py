import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/churn_model.pkl")

# Load feature names
features = joblib.load("models/feature_columns.pkl")

# Get feature importance
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(10)

plt.figure(figsize=(10,6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance Score")
plt.title("Top Features Influencing Customer Churn")

plt.gca().invert_yaxis()

plt.show()