"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert 'model_loaded' in data


def test_predict_transaction():
    """Test the prediction endpoint."""
    transaction_data = {
        "transaction_id": "TX001",
        "amount": 450000,
        "hour": 3,
        "is_new_device": 1,
        "is_foreign_location": 1,
        "transactions_last_hour": 8,
        "previous_transaction_amount": 2500
    }
    
    response = client.post("/predict", json=transaction_data)
    assert response.status_code == 200
    data = response.json()
    assert data['transaction_id'] == 'TX001'
    assert data['prediction'] in ['FRAUD', 'LEGITIMATE']
    assert 0 <= data['fraud_probability'] <= 1


def test_batch_prediction():
    """Test batch prediction endpoint."""
    transactions = [
        {
            "transaction_id": "TX001",
            "amount": 450000,
            "hour": 3,
            "is_new_device": 1,
            "is_foreign_location": 1,
            "transactions_last_hour": 8,
            "previous_transaction_amount": 2500
        },
        {
            "transaction_id": "TX002",
            "amount": 2500,
            "hour": 14,
            "is_new_device": 0,
            "is_foreign_location": 0,
            "transactions_last_hour": 2,
            "previous_transaction_amount": 3000
        }
    ]
    
    response = client.post("/predict/batch", json=transactions)
    assert response.status_code == 200
    data = response.json()
    assert data['count'] == 2
    assert len(data['predictions']) == 2