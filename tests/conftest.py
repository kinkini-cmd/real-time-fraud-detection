"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path("models/fraud_model.pkl")


@pytest.fixture(scope="session", autouse=True)
def ensure_model_available():
    """Train a small model if none exists so API tests are self-contained."""
    from src.api.main import predictor

    if predictor.is_loaded:
        return

    from src.features.feature_engineering import get_feature_columns
    from src.models.train import FraudDetectionModel

    rng = np.random.default_rng(42)
    columns = get_feature_columns()
    X = pd.DataFrame({column: rng.random(60) for column in columns})
    y = pd.Series([0] * 54 + [1] * 6)

    model = FraudDetectionModel()
    model.train(X, y, log_to_mlflow=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    predictor._load_model()
