# Telecom Customer Churn Prediction & MLOps System

An enterprise-grade, end-to-end Machine Learning and MLOps pipeline designed to predict customer churn, optimize predictive recall using Bayesian hyperparameter tuning, explain model decisions with XAI, track model artifacts via MLflow, serve real-time predictions via FastAPI, and monitor data drift automatically.

---

## 📌 Project Overview & Business Context

In the telecommunications sector, customer acquisition costs (CAC) typically exceed customer retention costs by 5x to 7x. When a customer churns, the business suffers immediate annual recurring revenue (ARR) loss and incurs acquisition overhead.

This system analyzes behavioral signals across 7,043 subscribers to proactively identify individuals likely to cancel their subscriptions. Rather than relying on misleading raw Accuracy, the pipeline optimizes **Recall** and **ROC-AUC** using **XGBoost** paired with **Optuna Bayesian hyperparameter optimization** to minimize False Negatives.

---

## 🏛 System Architecture & Modular Design

The repository follows a decoupled production architecture where data ingestion, preprocessing, MLOps training, model serving, and CI/CD validation are strictly isolated:

```
telecom_churn_project/
├── .github/
│   └── workflows/
│       └── ci.yml             # Automated Docker build & health check CI pipeline
├── app/
│   └── main.py                # FastAPI REST endpoints & KS data drift monitoring
├── data/
│   └── Telco-Customer-Churn.csv
├── Preprocessing_Telecom/
│   └── preprocess.py          # Automated ETL, imputation, and StandardScaler logic
├── Models_Telecom/
│   └── train_evaluate.py      # XGBoost training, Optuna tuning, SHAP & MLflow logging
├── output/
│   ├── models/                # SHAP summary/waterfall plots & serialized models
│   └── processed/             # Scalers and reference processed dataset for drift
├── Dockerfile                 # Containerization configuration
├── main.py                    # Unified end-to-end pipeline orchestrator
├── README.md
└── requirements.txt
```

---

## ⚡ MLOps, Explainability & Production Features

1. **Optuna Hyperparameter Optimization:** Automated Bayesian search across 30+ trials tuning `max_depth`, `learning_rate`, `subsample`, and `scale_pos_weight` to maximize ROC-AUC.
2. **SHAP (SHapley Additive exPlanations):** Generates local waterfall and global summary diagnostic plots (`output/models/`) to provide stakeholder transparency into primary churn drivers.
3. **MLflow Model Registry:** Tracks parameters, evaluation metrics, logs visual artifacts, and automatically registers validated models into a `Production` stage.
4. **FastAPI Microservice:** Exposes high-throughput `/predict` inference routes and a `/monitor/drift` route using the two-sample **Kolmogorov-Smirnov (KS) statistical test**.
5. **Continuous Integration (CI):** Automated GitHub Actions workflow that builds the container image and tests endpoint health on every push.

---

## 📈 Model Performance Benchmarks

Evaluated on an unseen stratified test partition ($n = 1,409$).

| Model Architecture | Optimization Method | ROC-AUC | Recall | Precision | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** ⭐ | **Optuna Bayesian Tuning** | **0.8812** | **0.8400** | **0.5420** | **0.6580** |
| Balanced Logistic Regression | Baseline Weighting | 0.8412 | 0.7968 | 0.5188 | 0.6284 |
| Random Forest | Default Parameters | 0.8229 | 0.5214 | 0.5847 | 0.5512 |

---

## 🚀 Quickstart & Setup Guide

### 1. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Executes automated preprocessing, Optuna tuning, XGBoost training, SHAP diagnostic generation, and MLflow logging:
```bash
python main.py
```

### 3. Launch the FastAPI Server Locally
Start the REST API backend:
```bash
uvicorn app.main:app --reload
```
* Access interactive Swagger API documentation at: **`http://127.0.0.1:8000/docs`**

### 4. Run via Docker
```bash
docker build -t telecom-churn-api .
docker run -d -p 8000:8000 --name churn-container telecom-churn-api
```

---

## 💡 Key Business Drivers & Strategic Insights

1. **Contract Structure:** Month-to-month contracts exhibit a churn rate exceeding **42%**, compared to under **3%** for two-year commitments.
2. **Payment Method Friction:** Electronic check users show the highest churn propensity (~45%), benefiting from automated ACH incentives.
3. **Fiber Optic Support Deficit:** Churn is heavily concentrated among high-speed broadband users lacking bundled `TechSupport` or `OnlineSecurity`.