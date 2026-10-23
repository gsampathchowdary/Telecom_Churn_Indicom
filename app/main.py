import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.stats import ks_2samp

app = FastAPI(title="Telecom Churn Prediction & Drift Monitoring Service", version="1.0.0")

MODEL_PATH = "output/models/xgb_churn_model.joblib"
REFERENCE_DATA_PATH = "output/processed/processed_churn.csv"

class SubscriberFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract_Month_to_month: int = 0
    InternetService_Fiber_optic: int = 0
    TechSupport_No: int = 0

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "telecom-churn-api"}

@app.post("/predict")
def predict_churn(payload: SubscriberFeatures):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(REFERENCE_DATA_PATH):
        raise HTTPException(status_code=500, detail="Model or reference dataset missing. Run main.py training pipeline first.")
    try:
        model = joblib.load(MODEL_PATH)
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        
        # Get the exact feature columns expected by the model (excluding target 'Churn')
        expected_columns = ref_df.drop(columns=["Churn"]).columns
        
        # Create a dataframe with all expected columns initialized to 0
        input_df = pd.DataFrame(0, index=[0], columns=expected_columns)
        
        # Fill in the values provided by the user payload
        payload_dict = payload.dict()
        for key, value in payload_dict.items():
            if key in input_df.columns:
                input_df[key] = value
                
        # Make prediction
        prob = float(model.predict_proba(input_df)[:, 1][0])
        return {
            "churn_probability": round(prob, 4),
            "at_risk_churner": bool(prob >= 0.45),
            "threshold_used": 0.45
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitor/drift")
def check_data_drift(current_batch: list[SubscriberFeatures]):
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise HTTPException(status_code=500, detail="Reference dataset not found.")
    
    ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    curr_df = pd.DataFrame([item.dict() for item in current_batch])

    drift_report = {}
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in ref_df.columns and col in curr_df.columns:
            stat, p_value = ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())
            drift_report[col] = {
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": bool(p_value < 0.05)
            }
    return {"drift_analysis": drift_report}