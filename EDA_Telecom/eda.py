import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "data" / "Telco-Customer-Churn.csv"
output_dir = base_dir / "output" / "eda"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 1. churn distribution
plt.figure(figsize=(6, 4))
df["Churn"].value_counts().plot(kind="bar", color=["#2ca02c", "#d62728"])
plt.title("target distribution: churn vs retained")
plt.xlabel("churn")
plt.ylabel("customer count")
plt.savefig(output_dir / "churn_distribution.png", bbox_inches="tight")
plt.close()

# 2. churn rate across key categorical drivers
cat_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for col, ax in zip(cat_cols, axes.flatten()):
    churn_rates = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean())
    churn_rates.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title(f"churn rate by {col}")
    ax.set_ylabel("churn proportion")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(output_dir / "categorical_churn_rates.png", bbox_inches="tight")
plt.close()

# 3. numeric drivers: monthly charges and tenure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(
    x="Churn",
    y="tenure",
    data=df,
    ax=ax1,
    hue="Churn",
    palette="Set2",
    legend=False,
)
ax1.set_title("tenure vs churn")

sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=df,
    ax=ax2,
    hue="Churn",
    palette="Set2",
    legend=False,
)
ax2.set_title("monthly charges vs churn")
plt.tight_layout()
plt.savefig(output_dir / "numeric_drivers.png", bbox_inches="tight")
plt.close()

print(f"eda plots saved successfully in {output_dir}")

