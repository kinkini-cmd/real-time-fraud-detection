"""Tests for the streaming fraud detection pipeline."""

from src.kafka.pipeline import FraudStreamProcessor


class FakePredictor:
    """Predictor stub returning a fixed fraud result."""

    def predict_single(self, transaction):
        return {
            'prediction': 'FRAUD',
            'fraud_probability': 0.95,
            'is_fraud': 1
        }


class FakeDatabase:
    """Database stub recording inserted transactions."""

    def __init__(self):
        self.records = []

    def insert_transaction(self, record):
        self.records.append(record)
        return len(self.records)


def test_pipeline_scores_and_persists_transaction():
    """The processor scores, persists, and monitors a transaction."""
    database = FakeDatabase()
    processor = FraudStreamProcessor(
        predictor=FakePredictor(),
        database=database
    )

    result = processor.process_transaction({
        'transaction_id': 'TX999',
        'amount': 450000,
        'location': 'Foreign'
    })

    assert result['prediction'] == 'FRAUD'
    assert result['is_fraud'] == 1
    assert len(database.records) == 1
    assert database.records[0]['transaction_id'] == 'TX999'
    assert processor.monitor.get_ml_metrics()['total_predictions'] == 1
