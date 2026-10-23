from Preprocessing_Telecom.preprocess import preprocess_data
from Models_Telecom.train_evaluate import train_and_evaluate

def main():
    print("Step 1: Starting Data Preprocessing...")
    processed_file = preprocess_data()
    
    print("\nStep 2: Starting Model Training, Optuna Tuning, & Evaluation...")
    train_and_evaluate(processed_file)
    
    print("\nPipeline execution complete successfully!")

if __name__ == "__main__":
    main()