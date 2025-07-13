"""
Feature engineering module for plant disease detection.
Creates new features from environmental variables.
"""

import pandas as pd
import numpy as np


def create_interaction_features(df):
    """
    Create interaction features between environmental variables.

    Args:
        df (pd.DataFrame): Dataset with environmental features

    Returns:
        pd.DataFrame: Dataset with additional interaction features
    """
    print("Creating interaction features...")

    df_enhanced = df.copy()

    # Temperature-Humidity interaction (important for plant diseases)
    df_enhanced["temp_humidity_interaction"] = df["temperature"] * df["humidity"]

    # Rainfall-pH interaction (affects nutrient availability)
    df_enhanced["rainfall_ph_interaction"] = df["rainfall"] * df["soil_pH"]

    # Temperature-pH interaction (affects soil chemistry)
    df_enhanced["temp_ph_interaction"] = df["temperature"] * df["soil_pH"]

    print(f"Added {df_enhanced.shape[1] - df.shape[1]} interaction features")
    return df_enhanced


def create_derived_features(df):
    """
    Create derived features from environmental variables.

    Args:
        df (pd.DataFrame): Dataset with environmental features

    Returns:
        pd.DataFrame: Dataset with additional derived features
    """
    print("Creating derived features...")

    df_enhanced = df.copy()

    # Temperature categories (cool, moderate, warm, hot)
    df_enhanced["temp_category"] = pd.cut(
        df["temperature"], bins=4, labels=["cool", "moderate", "warm", "hot"]
    ).cat.codes

    # Humidity categories (low, medium, high)
    df_enhanced["humidity_category"] = pd.cut(
        df["humidity"], bins=3, labels=["low", "medium", "high"]
    ).cat.codes

    # pH categories (acidic, neutral, alkaline)
    df_enhanced["ph_category"] = pd.cut(
        df["soil_pH"], bins=[0, 6.5, 7.5, 14], labels=["acidic", "neutral", "alkaline"]
    ).cat.codes

    # Rainfall categories (low, medium, high)
    df_enhanced["rainfall_category"] = pd.cut(
        df["rainfall"], bins=3, labels=["low", "medium", "high"]
    ).cat.codes

    print(f"Added {df_enhanced.shape[1] - df.shape[1]} derived features")
    return df_enhanced


def create_stress_indicators(df):
    """
    Create plant stress indicator features.

    Args:
        df (pd.DataFrame): Dataset with environmental features

    Returns:
        pd.DataFrame: Dataset with stress indicator features
    """
    print("Creating stress indicator features...")

    df_enhanced = df.copy()

    # Heat stress indicator (high temperature + low humidity)
    df_enhanced["heat_stress"] = (
        (df["temperature"] > df["temperature"].quantile(0.75))
        & (df["humidity"] < df["humidity"].quantile(0.25))
    ).astype(int)

    # Drought stress indicator (low rainfall + high temperature)
    df_enhanced["drought_stress"] = (
        (df["rainfall"] < df["rainfall"].quantile(0.25))
        & (df["temperature"] > df["temperature"].quantile(0.75))
    ).astype(int)

    # pH stress indicator (extreme pH values)
    ph_mean = df["soil_pH"].mean()
    ph_std = df["soil_pH"].std()
    df_enhanced["ph_stress"] = (np.abs(df["soil_pH"] - ph_mean) > 2 * ph_std).astype(
        int
    )

    # Combined stress score
    df_enhanced["total_stress"] = (
        df_enhanced["heat_stress"]
        + df_enhanced["drought_stress"]
        + df_enhanced["ph_stress"]
    )

    print(f"Added {df_enhanced.shape[1] - df.shape[1]} stress indicator features")
    return df_enhanced


def engineer_features(X_train, X_test):
    """
    Apply feature engineering to training and testing sets.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features

    Returns:
        tuple: Enhanced X_train and X_test
    """
    print("Starting feature engineering...")

    # Apply feature engineering to training set
    X_train_enhanced = create_interaction_features(X_train)
    X_train_enhanced = create_derived_features(X_train_enhanced)
    X_train_enhanced = create_stress_indicators(X_train_enhanced)

    # Apply same transformations to test set
    X_test_enhanced = create_interaction_features(X_test)
    X_test_enhanced = create_derived_features(X_test_enhanced)
    X_test_enhanced = create_stress_indicators(X_test_enhanced)

    print("Feature engineering completed:")
    print(f"Training features: {X_train.shape[1]} → {X_train_enhanced.shape[1]}")
    print(f"Testing features: {X_test.shape[1]} → {X_test_enhanced.shape[1]}")

    return X_train_enhanced, X_test_enhanced


if __name__ == "__main__":
    # Test feature engineering with processed data
    try:
        print("Loading processed data for feature engineering test...")
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")

        X_train_features, X_test_features = engineer_features(X_train, X_test)

        print("\nFinal feature shapes:")
        print(f"X_train: {X_train_features.shape}")
        print(f"X_test: {X_test_features.shape}")
        print(f"\nNew features: {list(X_train_features.columns)}")

    except FileNotFoundError:
        print("Processed data not found. Run data_preprocessing.py first.")
