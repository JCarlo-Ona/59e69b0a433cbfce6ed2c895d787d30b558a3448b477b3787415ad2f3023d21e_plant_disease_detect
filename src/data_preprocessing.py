"""
Data preprocessing module for plant disease detection.
Handles loading, cleaning, and splitting the dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_data(file_path="data/raw/plant_disease_dataset.csv"):
    """
    Load the plant disease dataset from CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers.

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("Cleaning data...")

    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print(f"Missing values found:\n{missing_values}")
        # Fill missing values with median for numerical columns
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("No missing values found")

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - len(df)
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate rows")

    print(f"Data cleaning completed. Final dataset: {df.shape[0]} rows")
    return df


def split_data(df, target_column="disease_present", test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): Cleaned dataset
        target_column (str): Name of the target column
        test_size (float): Proportion of dataset for testing
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("Splitting data into train/test sets...")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"Features: {list(X.columns)}")
    print(f"Target: {target_column}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features

    Returns:
        tuple: Scaled X_train, X_test, and fitted scaler
    """
    print("Scaling features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print("Feature scaling completed")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data():
    """
    Complete preprocessing pipeline.

    Returns:
        tuple: X_train, X_test, y_train, y_test (all preprocessed)
    """
    # Load and clean data
    df = load_data()
    df_clean = clean_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
    X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Preprocessing completed and data saved to data/processed/")
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    # Run preprocessing if script is executed directly
    preprocess_data()
