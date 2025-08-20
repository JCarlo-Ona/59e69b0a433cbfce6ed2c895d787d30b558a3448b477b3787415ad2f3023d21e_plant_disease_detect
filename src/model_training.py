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

import mlflow
import mlflow.sklearn


def train_random_forest(X_train, y_train):
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def train_logistic_regression(X_train, y_train):
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    return lr_model


def select_best_model(X_train, y_train):
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
        with mlflow.start_run(run_name=f"train_{name.replace(' ', '_')}", nested=True):
            mlflow.set_tag("model", name)
            mlflow.log_params(model.get_params())

            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="accuracy"
            )
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()

            mlflow.log_metric("cv_accuracy_mean", mean_score)
            mlflow.log_metric("cv_accuracy_std", std_score)

            print(f"{name} CV Accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name

            mlflow.end_run()

    print(f"\nBest model: {best_name} with CV accuracy: {best_score:.4f}")
    best_model.fit(X_train, y_train)

    return best_model, best_name, best_score


def save_model(model, model_name="plant_disease_model"):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path


def train_model():
    print("Starting model training pipeline...")

    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    _y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    print("Applying feature engineering...")
    X_train_features, X_test_features = engineer_features(X_train, X_test)

    # Start MLflow experiment
    mlflow.set_experiment("plant-disease-ml-training")
    with mlflow.start_run(run_name="best_model_training"):
        best_model, best_name, best_score = select_best_model(X_train_features, y_train)

        model_path = save_model(best_model, "plant_disease_model")
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        mlflow.log_artifact(model_path)

        mlflow.log_metric("best_cv_accuracy", best_score)
        mlflow.set_tag("best_model", best_name)

        # Save feature-engineered data
        os.makedirs("data/processed", exist_ok=True)
        X_train_features.to_csv("data/processed/X_train_features.csv", index=False)
        X_test_features.to_csv("data/processed/X_test_features.csv", index=False)

        mlflow.log_artifact("data/processed/X_train_features.csv")
        mlflow.log_artifact("data/processed/X_test_features.csv")

    print(f"Training completed! Best model: {best_name}")
    return best_model, best_name


if __name__ == "__main__":
    train_model()
