"""
Complete ML pipeline for plant disease detection.
Runs all steps from data preprocessing to model evaluation.
"""

from data_preprocessing import preprocess_data
from model_training import train_model
from evaluation import evaluate_saved_model
import os
import time


def main():
    """
    Execute the complete ML pipeline.
    """
    print("=" * 60)
    print("PLANT DISEASE DETECTION ML PIPELINE")
    print("=" * 60)

    start_time = time.time()

    try:
        # Step 1: Data Preprocessing
        print("\n🔄 STEP 1: DATA PREPROCESSING")
        print("-" * 40)
        X_train, X_test, y_train, y_test = preprocess_data()
        print("✅ Data preprocessing completed successfully!")

        # Step 2: Model Training
        print("\n🔄 STEP 2: MODEL TRAINING")
        print("-" * 40)
        model, model_name = train_model()
        print(f"✅ Model training completed! Best model: {model_name}")

        # Step 3: Model Evaluation
        print("\n🔄 STEP 3: MODEL EVALUATION")
        print("-" * 40)
        metrics = evaluate_saved_model()
        print("✅ Model evaluation completed successfully!")

        # Pipeline Summary
        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print("✅ All steps completed successfully!")
        print(f"⏱️  Total execution time: {duration:.2f} seconds")
        print(f"🎯 Final model accuracy: {metrics['accuracy']:.4f}")
        print("📊 Results saved to: reports/metrics.txt")
        print("🤖 Trained model saved to: models/plant_disease_model.pkl")

        print("\n📁 Generated Files:")
        print("  • data/processed/X_train.csv")
        print("  • data/processed/X_test.csv")
        print("  • data/processed/y_train.csv")
        print("  • data/processed/y_test.csv")
        print("  • data/processed/X_train_features.csv")
        print("  • data/processed/X_test_features.csv")
        print("  • models/plant_disease_model.pkl")
        print("  • reports/metrics.txt")

        print("\n🌱 Your plant disease detection model is ready for deployment!")

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        print("Please check the error message and try again.")
        raise


def verify_requirements():
    """
    Verify that all required files and directories exist.
    """
    print("🔍 Verifying requirements...")

    required_dirs = ["data", "data/raw", "src", "models", "reports"]
    required_files = ["data/raw/plant_disease_dataset.csv"]

    missing_items = []

    # Check directories
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_items.append(f"Directory: {directory}")

    # Check files
    for file in required_files:
        if not os.path.exists(file):
            missing_items.append(f"File: {file}")

    if missing_items:
        print("❌ Missing requirements:")
        for item in missing_items:
            print(f"  • {item}")
        return False
    else:
        print("✅ All requirements satisfied!")
        return True


if __name__ == "__main__":
    print("🌱 PLANT DISEASE DETECTION ML PROJECT")
    print("=" * 60)

    # Verify requirements before running
    if verify_requirements():
        print("\n🚀 Starting ML pipeline...")
        main()
    else:
        print("\n❌ Please fix missing requirements before running the pipeline.")
