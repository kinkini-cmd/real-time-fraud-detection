"""End-to-end streaming pipeline for real-time fraud detection."""

import time
from typing import Dict, Any, Optional

from src.models.predict import ModelPredictor
from src.monitoring.monitor import SystemMonitor


class FraudStreamProcessor:
    """Processes streaming transactions through the fraud detection model."""

    def __init__(self, predictor: Optional[ModelPredictor] = None,
                 database=None, monitor: Optional[SystemMonitor] = None):
        self.predictor = predictor or ModelPredictor()
        self.database = database
        self.monitor = monitor or SystemMonitor()

    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single transaction and persist/monitor the result."""
        start = time.perf_counter()
        result = self.predictor.predict_single(transaction)
        elapsed = time.perf_counter() - start

        record = {
            'transaction_id': transaction.get('transaction_id'),
            'amount': transaction.get('amount', 0),
            'prediction': result['prediction'],
            'fraud_probability': result['fraud_probability'],
            'is_fraud': result['is_fraud'],
            'location': transaction.get('location'),
            'device_id': transaction.get('device_id'),
        }

        if self.database is not None:
            try:
                record['id'] = self.database.insert_transaction(record)
            except Exception as exc:
                self.monitor.record_error(f"database insert failed: {exc}")

        self.monitor.record_prediction(record, elapsed)

        if result['is_fraud'] == 1:
            print(
                f"[ALERT] Fraud detected for {record['transaction_id']} "
                f"(probability={result['fraud_probability']:.3f})"
            )

        return record


def run_stream(bootstrap_servers: str = "localhost:9092",
               topic: str = "transactions") -> None:
    """Consume transactions from Kafka and score them in real time."""
    from src.kafka.consumer import TransactionConsumer
    from src.database.database import DatabaseManager

    database = DatabaseManager()
    try:
        database.connect()
    except Exception as exc:
        print(f"Database unavailable, continuing without persistence: {exc}")
        database = None

    processor = FraudStreamProcessor(database=database)
    consumer = TransactionConsumer(
        bootstrap_servers=bootstrap_servers,
        topic=topic
    )
    consumer.set_processing_callback(processor.process_transaction)

    try:
        consumer.start_consuming()
    except KeyboardInterrupt:
        consumer.stop()


if __name__ == "__main__":
    run_stream()
