import pandas as pd
import os
import mlflow
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path="data/raw/plant_disease_dataset.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    print("Cleaning data...")
    if df.isnull().sum().any():
        df = df.fillna(df.median(numeric_only=True))
    df = df.drop_duplicates()
    return df


def split_data(df, target_column="disease_present", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    return X_train_df, X_test_df, scaler


def preprocess_data():
    with mlflow.start_run(run_name="data_preprocessing"):
        # Load, clean, split, and scale
        df = load_data()
        mlflow.log_param("raw_rows", df.shape[0])
        mlflow.log_param("raw_columns", df.shape[1])

        df_clean = clean_data(df)
        mlflow.log_param("clean_rows", df_clean.shape[0])

        X_train, X_test, y_train, y_test = split_data(df_clean)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # Save processed files
        os.makedirs("data/processed", exist_ok=True)
        X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
        X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)

        # Save scaler
        with open("data/processed/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # Log artifacts
        mlflow.log_artifact("data/processed/X_train.csv")
        mlflow.log_artifact("data/processed/X_test.csv")
        mlflow.log_artifact("data/processed/y_train.csv")
        mlflow.log_artifact("data/processed/y_test.csv")
        mlflow.log_artifact("data/processed/scaler.pkl")

        # Log basic metrics
        mlflow.log_metric("n_train", X_train.shape[0])
        mlflow.log_metric("n_test", X_test.shape[0])
        mlflow.log_metric("n_features", X_train.shape[1])
        mlflow.log_param("features", list(X_train.columns))

        # Optional: Drift detection hooks (for HW4)
        # Example: mlflow.log_metric("feature_mean_temperature", X_train["temperature"].mean())

        print("Preprocessing completed and logged to MLflow.")

        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    preprocess_data()
