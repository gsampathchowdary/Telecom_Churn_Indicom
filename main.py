from pathlib import Path
from Models_Telecom.train_evaluate import train_and_evaluate
from Preprocessing_Telecom.preprocess import load_and_clean_data

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "data" / "Telco-Customer-Churn.csv"

    print("running preprocessing...")
    x_train, x_test, y_train, y_test = load_and_clean_data(data_file)

    print("running model training and evaluation...")
    metrics, best_model_name = train_and_evaluate(
        x_train, x_test, y_train, y_test
    )

    print("\n--- evaluation metrics summary ---")
    print(metrics.to_string(index=False))
    print(f"\nselected best model: {best_model_name}")
    print(f"all results and visual charts saved to: {base_dir / 'output'}")