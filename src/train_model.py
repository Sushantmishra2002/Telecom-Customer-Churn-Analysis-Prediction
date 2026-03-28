import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from data_preprocessing import load_data, clean_data
from feature_engineering import encode_features


def main():

    print("Loading dataset...")

    df = load_data("data/telco_churn.csv")

    print("Cleaning data...")

    df = clean_data(df)

    print("Encoding features...")

    df = encode_features(df)

    print("Preparing features and target...")

    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]
    joblib.dump(X.columns, "models/feature_columns.pkl")

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Starting hyperparameter tuning...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    print("Best parameters found:", grid_search.best_params_)

    print("Making predictions...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    print("Saving model...")

    joblib.dump(model, "models/churn_model.pkl")

    print("Model saved successfully!")


if __name__ == "__main__":
    main()