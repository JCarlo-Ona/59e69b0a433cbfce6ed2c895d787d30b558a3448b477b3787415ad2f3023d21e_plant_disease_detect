"""
Model training module for plant disease detection.
Trains and saves machine learning models.
"""

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from feature_engineering import engineer_features


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels

    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model
    """
    print("Training Random Forest model...")

    # Initialize model with good parameters for our dataset
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # Train the model
    rf_model.fit(X_train, y_train)

    # Evaluate with cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="accuracy")
    print(
        f"Random Forest CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
    )

    return rf_model


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels

    Returns:
        sklearn.linear_model.LogisticRegression: Trained model
    """
    print("Training Logistic Regression model...")

    # Initialize model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)

    # Train the model
    lr_model.fit(X_train, y_train)

    # Evaluate with cross-validation
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="accuracy")
    print(
        f"Logistic Regression CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
    )

    return lr_model


def select_best_model(X_train, y_train):
    """
    Train multiple models and select the best one based on cross-validation.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels

    Returns:
        tuple: (best_model, model_name, best_score)
    """
    print("Training multiple models to select the best one...")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }

    best_score = 0
    best_model = None
    best_name = None

    for name, model in models.items():
        print(f"Training {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        print(f"{name} CV Accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} with CV accuracy: {best_score:.4f}")

    # Train the best model on full training set
    best_model.fit(X_train, y_train)

    return best_model, best_name, best_score


def save_model(model, model_name="plant_disease_model"):
    """
    Save the trained model to disk.

    Args:
        model: Trained scikit-learn model
        model_name (str): Name for the saved model file
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the model
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def train_model():
    """
    Complete model training pipeline.

    Returns:
        tuple: (trained_model, model_name)
    """
    print("Starting model training pipeline...")

    # Load processed data
    print("Loading processed data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    _y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Apply feature engineering
    print("Applying feature engineering...")
    X_train_features, X_test_features = engineer_features(X_train, X_test)

    # Train and select best model
    best_model, best_name, best_score = select_best_model(X_train_features, y_train)

    # Save the model
    save_model(best_model, "plant_disease_model")

    # Save feature-engineered data for evaluation
    os.makedirs("data/processed", exist_ok=True)
    X_train_features.to_csv("data/processed/X_train_features.csv", index=False)
    X_test_features.to_csv("data/processed/X_test_features.csv", index=False)

    print(f"Training completed! Best model: {best_name}")
    return best_model, best_name


if __name__ == "__main__":
    # Run model training if script is executed directly
    train_model()
