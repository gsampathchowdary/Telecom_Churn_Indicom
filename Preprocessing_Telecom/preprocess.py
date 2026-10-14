import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base_dir = Path(__file__).resolve().parent.parent
output_dir = base_dir / "output" / "processed"
os.makedirs(output_dir, exist_ok=True)


def load_and_clean_data(file_path=None):
    if file_path is None:
        file_path = base_dir / "data" / "Telco-Customer-Churn.csv"

    df = pd.read_csv(file_path)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    multi_cat_cols = [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    x = df.drop("Churn", axis=1)
    y = df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
    x_test[num_cols] = scaler.transform(x_test[num_cols])

    joblib.dump(scaler, output_dir / "scaler.joblib")
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_and_clean_data()
    print(
        f"preprocessing complete. training rows: {x_train.shape[0]}, testing rows: {x_test.shape[0]}"
    )