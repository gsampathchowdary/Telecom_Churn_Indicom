import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def preprocess_data(input_path="data/Telco-Customer-Churn.csv", output_dir="output/processed"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Clean TotalCharges (some entries have spaces instead of numbers)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop customerID if it exists
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    # Target variable encoding
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Split features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Standard scaling for continuous variables
    scaler = StandardScaler()
    continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    existing_cont_cols = [col for col in continuous_cols if col in X.columns]
    X[existing_cont_cols] = scaler.fit_transform(X[existing_cont_cols])
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    
    # Combine back for processed dataset reference
    processed_df = X.copy()
    processed_df['Churn'] = y
    
    processed_file_path = os.path.join(output_dir, "processed_churn.csv")
    processed_df.to_csv(processed_file_path, index=False)
    print(f"Preprocessing complete. Processed data saved to {processed_file_path}")
    
    return processed_file_path

if __name__ == "__main__":
    preprocess_data()