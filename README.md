# Telecom Customer Churn Prediction System

An end-to-end Machine Learning pipeline designed to predict customer churn using customer demographics, subscribed services, and account analytics. Built with modular production-grade design principles, reproducible Git workflows, and an emphasis on recall-driven business intervention.

---

## 📌 Project Overview & Business Context

In the telecommunications sector, customer acquisition costs (CAC) typically exceed customer retention costs by a factor of 5× to 7×. When a customer churns, the business suffers immediate annual recurring revenue (ARR) loss and incurs acquisition overhead to replace them.

This system analyzes behavioral signals across 7,043 customers to proactively identify individuals likely to cancel their subscriptions. Rather than maximizing standard Accuracy—which is misleading on imbalanced datasets (~26.5% churn rate)—the pipeline prioritizes **Recall** and **ROC-AUC** to minimize **False Negatives** (customers who leave undetected without an intervention offer).

---

## 🏛 System Architecture & Modular Design

The repository follows a decoupled architecture where data ingestion, preprocessing, visualization, and model lifecycle management are strictly isolated.

```
telecom_churn_project/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                          # Unified orchestrator pipeline
├── data/
│   └── Telco-Customer-Churn.csv
├── src/
│   ├── EDA_Telecom/
│   │   ├── __init__.py
│   │   └── eda.py                   # Univariate/Bivariate visualization scripts
│   ├── Preprocessing_Telecom/
│   │   ├── __init__.py
│   │   └── preprocess.py            # Encoding, imputation & scaling logic
│   └── Models_Telecom/
│       ├── __init__.py
│       └── train_evaluate.py        # Multi-model benchmarking & evaluation
└── output/
    ├── eda/                         # Rendered retention/churn factor charts
    │   ├── categorical_churn_rates.png
    │   ├── churn_distribution.png
    │   └── numeric_drivers.png
    ├── models/                      # Metric tables, ROC plots & serialized models
    │   ├── best_model.joblib
    │   ├── best_model_confusion_matrix.png
    │   ├── evaluation_metrics.csv
    │   └── roc_auc_curve.png
    └── processed/                   # Fitted preprocessing transformers
        └── scaler.joblib
```

---

## 🔀 Git Branching Strategy

Development follows a feature-branch workflow where code isolation mirrors system modules:

```text
master ───────────────●─────────────────────●─────────────────●───────────● (v1.0.0)
                       \                   /                 /           /
feature/eda             ●─────────────────┘                 /           /
                         \                                 /           /
feature/preprocessing     ●───────────────────────────────┘           /
                           \                                         /
feature/models              ●───────────────────────────────────────┘
                             \                                     /
feature/pipeline              ●───────────────────────────────────┘
```

* `feature/eda`: Exploratory analysis, class distribution, churn vs. billing/contract correlations.
* `feature/preprocessing`: Null imputation (`TotalCharges`), one-hot encoding, binary mapping, stratified train-test splitting, standard scaling.
* `feature/models`: Multi-classifier benchmarking, cost-sensitive class weighting, metric serialization, ROC-AUC visualization.
* `feature/pipeline`: Unified entry-point execution (`main.py`) and artifact persistence.

---

## 📊 Dataset & Feature Dictionary

* **Dataset Size:** 7,043 records, 21 initial columns
* **Target Feature:** `Churn` (`1` = Churned within the last month, `0` = Retained)

| Feature Category | Features Included | Preprocessing Transform |
| :--- | :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Binary conversion / Dummy encoding |
| **Services Subscribed** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | One-Hot Encoded (`drop_first=True`) |
| **Account Metadata** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | `StandardScaler` on continuous features; median imputation on `TotalCharges` whitespace entries |

---

## 📈 Metric Benchmarks & Model Comparison

All models were evaluated on an unseen stratified test partition (20% sample, $n = 1,409$).

| Model | Accuracy | Precision | **Recall** | F1-Score | **ROC-AUC** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** ⭐ | **0.7516** | **0.5188** | **0.7968** | **0.6284** | **0.8412** |
| Random Forest (Balanced) | 0.7764 | 0.5847 | 0.5214 | 0.5512 | 0.8229 |
| Decision Tree (Max Depth = 5) | 0.7431 | 0.5098 | 0.6979 | 0.5891 | 0.7934 |
| Naive Bayes (Gaussian) | 0.7048 | 0.4682 | 0.8128 | 0.5941 | 0.8201 |
| K-Nearest Neighbors ($k=5$) | 0.7630 | 0.5601 | 0.5187 | 0.5386 | 0.7758 |

### Decision Logic for Production Deployment
* **Primary Champion:** `Logistic Regression` with balanced class weights.
* **Why not Random Forest or Accuracy-first models?** While Random Forest achieved higher raw accuracy (77.6%), its recall for the churn class was only 52.1%, leaving nearly 48% of churning customers completely undetected.
* **Balanced Logistic Regression** yielded **~79.7% Recall** combined with an industry-leading **0.841 ROC-AUC**, providing a well-calibrated probability score suitable for customer tiering.

---

## 🚀 Quickstart & Setup Guide

### 1. Configure Virtual Environment
```bash

# Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
Create a `requirements.txt` file (or install directly):
```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```
Run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Place Dataset
Place the raw dataset in the `data/` directory:
```bash
Telco-Customer-Churn.csv
```

### 4. Execute the End-to-End Pipeline
```bash
python main.py
```
To run isolated components:
```bash
python src/EDA_Telecom/eda.py
python src/Preprocessing_Telecom/preprocess.py
```

All figures, metrics, and trained model artifacts will automatically populate inside the `output/` folder.

---

## 💡 Key Business Drivers & Strategic Recommendations

Based on bivariate analysis and feature coefficients extracted during model training:

1. **Contract Structure:** Customers on month-to-month contracts churn at a rate exceeding **42%**, compared to under **3%** for two-year contracts.
   * *Action:* Implement onboarding incentives and price locks for transitioning month-to-month subscribers to 1-year commitments.
2. **Payment Method Friction:** Electronic check users exhibit the highest churn propensity (~45%).
   * *Action:* Offer a small monthly bill credit (e.g., \$2–\$5) for switching to automated ACH or credit card payments.
3. **Fiber Optic Support Deficit:** Churn is elevated among Fiber Optic users who lack `TechSupport` or `OnlineSecurity`.
   * *Action:* Bundle free 6-month trial access of security add-ons to high-speed broadband adopters.

---

