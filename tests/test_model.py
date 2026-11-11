"""Tests for model training and prediction."""

import numpy as np
import pandas as pd
from src.models.train import FraudDetectionModel
from src.models.predict import ModelPredictor


def test_model_training():
    """Test model training."""
    X = pd.DataFrame({
        'amount': np.random.lognormal(7.5, 0.8, 100),
        'hour': np.random.randint(0, 24, 100),
        'is_new_device': np.random.randint(0, 2, 100),
        'is_foreign_location': np.random.randint(0, 2, 100),
        'transactions_last_hour': np.random.poisson(2, 100)
    })
    y = pd.Series([0] * 95 + [1] * 5).sample(frac=1, random_state=42).reset_index(drop=True)

    model = FraudDetectionModel(model_type="random_forest")
    metrics = model.train(X, y, log_to_mlflow=False)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_model_prediction():
    """Test model prediction."""
    model = FraudDetectionModel()
    model._create_model()
    
    X = pd.DataFrame({
        'amount': [100, 500000],
        'hour': [10, 3],
        'is_new_device': [0, 1],
        'is_foreign_location': [0, 1],
        'transactions_last_hour': [2, 8]
    })
    
    # Train with minimal data
    y = pd.Series([0, 1])
    model.model.fit(X, y)
    
    predictions = model.predict(X)
    assert len(predictions) == 2
    assert all(p in [0, 1] for p in predictions)