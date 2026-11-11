"""Model prediction utilities."""

import os
import joblib
import pandas as pd
import numpy as np

from src.features.feature_engineering import engineer_features


class ModelPredictor:
    """Wrapper for making predictions with a trained model."""

    def __init__(self, model_path: str = "models/fraud_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.model_type = None
        self.feature_columns = None
        self._load_model()

    def _load_model(self):
        """Load the model from disk if it exists."""
        if not os.path.exists(self.model_path):
            return

        data = joblib.load(self.model_path)
        self.model = data['model']
        self.model_type = data.get('model_type')
        self.feature_columns = data['feature_columns']

    @property
    def is_loaded(self) -> bool:
        """Return True if a model is available for predictions."""
        return self.model is not None

    def _prepare_features(self, features: dict) -> pd.DataFrame:
        """Build a model-ready feature frame from raw transaction fields."""
        df = pd.DataFrame([features])
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = engineer_features(df)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_columns].apply(pd.to_numeric, errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

    def predict_single(self, features: dict) -> dict:
        """Make a prediction for a single transaction."""
        if not self.is_loaded:
            raise ValueError(
                f"Model not found at '{self.model_path}'. "
                "Train it first with src.models.train."
            )

        df = self._prepare_features(features)

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]

        return {
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': float(probability),
            'is_fraud': int(prediction)
        }

    def predict_batch(self, features_list: list) -> list:
        """Make predictions for multiple transactions."""
        if not self.is_loaded:
            raise ValueError(
                f"Model not found at '{self.model_path}'. "
                "Train it first with src.models.train."
            )

        if not features_list:
            return []

        df = pd.DataFrame(features_list)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = engineer_features(df)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_columns].apply(pd.to_numeric, errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)[:, 1]

        return [
            {
                'prediction': 'FRAUD' if pred == 1 else 'LEGITIMATE',
                'fraud_probability': float(prob),
                'is_fraud': int(pred)
            }
            for pred, prob in zip(predictions, probabilities)
        ]


if __name__ == "__main__":
    predictor = ModelPredictor()
    sample = {
        'amount': 450000,
        'hour': 3,
        'is_new_device': 1,
        'is_foreign_location': 1,
        'transactions_last_hour': 8,
        'previous_transaction_amount': 2500
    }
    result = predictor.predict_single(sample)
    print(f"Prediction: {result}")
