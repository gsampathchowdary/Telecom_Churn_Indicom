import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

base_dir = Path(__file__).resolve().parent.parent
output_dir = base_dir / "output" / "models"
os.makedirs(output_dir, exist_ok=True)


def train_and_evaluate(x_train, x_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }

    results = []
    roc_curves = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = (
            model.predict_proba(x_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc,
            }
        )

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "evaluation_metrics.csv", index=False)

    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc-auc comparison across models")
    plt.legend()
    plt.savefig(output_dir / "roc_auc_curve.png", bbox_inches="tight")
    plt.close()

    best_model_name = results_df.sort_values(
        by=["Recall", "ROC-AUC"], ascending=False
    ).iloc[0]["Model"]
    best_model = models[best_model_name]
    joblib.dump(best_model, output_dir / "best_model.joblib")

    y_pred_best = best_model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(f"confusion matrix: {best_model_name}")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Retained", "Churned"])
    plt.yticks(ticks, ["Retained", "Churned"])
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="red" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.ylabel("actual")
    plt.xlabel("predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "best_model_confusion_matrix.png")
    plt.close()

    return results_df, best_model_name