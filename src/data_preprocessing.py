import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df = df.drop("customerID", axis=1)

    return df