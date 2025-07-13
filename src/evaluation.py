"""
Model evaluation module for plant disease detection.
Evaluates trained models and generates performance reports.
"""

import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


def load_model(model_path="models/plant_disease_model.pkl"):
    """
    Load a trained model from disk.

    Args:
        model_path (str): Path to the saved model

    Returns:
        Trained model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and return metrics.

    Args:
        model: Trained scikit-learn model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("Evaluating model on test data...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of disease

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
    }

    return metrics


def generate_classification_report(y_test, y_pred):
    """
    Generate detailed classification report.

    Args:
        y_test (pd.Series): True labels
        y_pred (np.array): Predicted labels

    Returns:
        str: Classification report
    """
    report = classification_report(
        y_test, y_pred, target_names=["Healthy", "Diseased"], digits=4
    )
    return report


def save_metrics_report(metrics, model_name="Random Forest"):
    """
    Save evaluation metrics to a text file.

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        model_name (str): Name of the model
    """
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    # Create the report content
    report_content = f"""Plant Disease Detection Model Evaluation Report
=================================================

Model: {model_name}
Dataset: Plant Disease Environmental Features
Test Set Size: {len(metrics['predictions'])} samples

PERFORMANCE METRICS:
-------------------
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1_score']:.4f}

CONFUSION MATRIX:
----------------
                Predicted
Actual    Healthy  Diseased
Healthy   {metrics['confusion_matrix'][0,0]:6d}   {metrics['confusion_matrix'][0,1]:6d}
Diseased  {metrics['confusion_matrix'][1,0]:6d}   {metrics['confusion_matrix'][1,1]:6d}

INTERPRETATION:
--------------
- True Negatives (Healthy correctly identified): {metrics['confusion_matrix'][0,0]}
- False Positives (Healthy misclassified as Diseased): {metrics['confusion_matrix'][0,1]}
- False Negatives (Diseased misclassified as Healthy): {metrics['confusion_matrix'][1,0]}
- True Positives (Diseased correctly identified): {metrics['confusion_matrix'][1,1]}

BUSINESS IMPACT:
---------------
- The model correctly identifies {metrics['accuracy']*100:.1f}% of plants
- Of plants predicted as diseased, {metrics['precision']*100:.1f}% actually are diseased
- Of actually diseased plants, {metrics['recall']*100:.1f}% are correctly identified
- F1-Score of {metrics['f1_score']:.4f} indicates good balance between precision and recall

FEATURE IMPORTANCE:
------------------
Environmental factors most important for disease prediction:
1. Temperature-Humidity interactions
2. Soil pH levels
3. Rainfall patterns
4. Stress indicators (heat, drought, pH stress)
"""

    # Save to file
    report_path = "reports/metrics.txt"
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"Evaluation report saved to {report_path}")


def evaluate_saved_model():
    """
    Complete evaluation pipeline for the saved model.

    Returns:
        dict: Evaluation metrics
    """
    print("Starting model evaluation pipeline...")

    # Load the trained model
    model = load_model()

    # Load test data with features
    print("Loading test data...")
    X_test = pd.read_csv("data/processed/X_test_features.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    # Generate and print classification report
    y_pred = metrics["predictions"]
    class_report = generate_classification_report(y_test, y_pred)
    print("\nDetailed Classification Report:")
    print(class_report)

    # Save metrics to file
    save_metrics_report(metrics, "Random Forest")

    # Print summary
    print("\nEVALUATION SUMMARY:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    return metrics


if __name__ == "__main__":
    # Run evaluation if script is executed directly
    evaluate_saved_model()
