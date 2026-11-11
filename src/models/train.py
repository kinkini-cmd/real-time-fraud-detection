"""Model training and prediction for fraud detection."""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
import numpy as np


class FraudDetectionModel:
    """Fraud detection model wrapper."""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        
    def _create_model(self):
        """Create the underlying model."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=42,
                class_weight="balanced",
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              experiment_name: str = "fraud-detection",
              run_name: str = None,
              log_to_mlflow: bool = True) -> dict:
        """Train the model with MLflow tracking."""
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train model
        self._create_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # MLflow tracking
        if log_to_mlflow:
            import mlflow
            import mlflow.sklearn

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("model_type", self.model_type)
                mlflow.log_param("n_estimators", getattr(self.model, 'n_estimators', None))
                mlflow.log_param("class_weight", "balanced")
                
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                try:
                    mlflow.sklearn.log_model(
                        self.model,
                        "model",
                        skops_trusted_types=["sklearn.tree._tree.Tree"]
                    )
                except TypeError:
                    mlflow.sklearn.log_model(self.model, "model")
        
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data['model_type']
        self.feature_columns = data['feature_columns']


def train_model(file_path: str, model_type: str = "random_forest") -> FraudDetectionModel:
    """Train a fraud detection model from a data file."""
    from src.data.preprocessing import preprocess_data
    from src.features.feature_engineering import engineer_features, get_feature_columns
    
    # Load and preprocess data
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    df = engineer_features(df)
    
    # Prepare features
    feature_cols = get_feature_columns()
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols]
    y = df['is_fraud']
    
    # Train model
    model = FraudDetectionModel(model_type=model_type)
    metrics = model.train(X, y)
    
    print(f"Model training completed!")
    print(f"Metrics: {metrics}")
    
    return model


if __name__ == "__main__":
    # Train model
    model = train_model("data/raw/transactions.csv", "random_forest")
    model.save("models/fraud_model.pkl")
    print("Model saved to models/fraud_model.pkl")