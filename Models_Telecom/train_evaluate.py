import os
import joblib
import optuna
import shap
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

os.makedirs("output/models", exist_ok=True)
os.makedirs("output/processed", exist_ok=True)

def train_and_evaluate(processed_path="output/processed/processed_churn.csv"):
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found at {processed_path}. Run preprocessing first.")
        
    df = pd.read_csv(processed_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 3.5),
            "random_state": 42,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    print("Running Optuna hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    best_params = study.best_params

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telecom_Churn_Optuna_SHAP")

    with mlflow.start_run(run_name="XGBoost_Optuna_Best"):
        mlflow.log_params(best_params)

        model = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
        model.fit(X_train, y_train)

        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= 0.45).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_probs),
            "recall": recall_score(y_test, y_preds),
            "precision": precision_score(y_test, y_preds),
            "f1_score": f1_score(y_test, y_preds)
        }
        mlflow.log_metrics(metrics)
        print(f"Model Evaluation Metrics: {metrics}")

        # Generate SHAP explanations
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        shap_plot_path = "output/models/shap_summary_plot.png"
        plt.savefig(shap_plot_path, dpi=300)
        plt.close()
        mlflow.log_artifact(shap_plot_path)

        # Save model binary locally
        model_path = "output/models/xgb_churn_model.joblib"
        joblib.dump(model, model_path)
        
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="churn_model",
            registered_model_name="TelecomChurnXGBoost",
        )
        print(f"Model trained, evaluated, and saved successfully to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()