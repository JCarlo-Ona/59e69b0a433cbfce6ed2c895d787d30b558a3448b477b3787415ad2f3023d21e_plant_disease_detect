# ruff: noqa: E402
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from data_preprocessing import clean_data, load_data


def test_load_data():
    """Check that data loads with correct shape and columns."""
    if os.path.exists("data/raw/plant_disease_dataset.csv"):
        df = load_data()
        assert df.shape == (10000, 5)
        assert "disease_present" in df.columns
        assert df["temperature"].dtype in ["float64", "int64"]


def test_clean_data():
    """Test data cleaning removes duplicates and handles missing values."""
    test_df = pd.DataFrame(
        {
            "temperature": [25.0, np.nan, 25.0],
            "humidity": [60.0, 70.0, 60.0],
            "rainfall": [10.0, 15.0, 10.0],
            "soil_pH": [6.5, 7.0, 6.5],
            "disease_present": [0, 1, 0],
        }
    )

    cleaned = clean_data(test_df)

    assert cleaned.shape[0] == 2
    assert cleaned.isnull().sum().sum() == 0


def test_model_predictions():
    """Check model makes valid predictions."""
    if os.path.exists("models/plant_disease_model.pkl"):
        import joblib

        model = joblib.load("models/plant_disease_model.pkl")
        test_input = np.random.rand(1, 15)
        prediction = model.predict(test_input)

        assert prediction[0] in [0, 1]
