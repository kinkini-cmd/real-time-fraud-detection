"""Kafka producer for streaming transactions."""

import json
import time
from typing import Dict, Any
from kafka import KafkaProducer
import numpy as np


class TransactionProducer:
    """Produces transaction events to Kafka."""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092", 
                 topic: str = "transactions"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
    
    def send_transaction(self, transaction: Dict[str, Any]):
        """Send a single transaction to Kafka."""
        self.producer.send(self.topic, transaction)
        self.producer.flush()
    
    def close(self):
        """Close the producer."""
        self.producer.close()


def generate_sample_transaction() -> Dict[str, Any]:
    """Generate a sample transaction for testing."""
    is_fraud = np.random.choice([0, 1], p=[0.95, 0.05])
    
    if is_fraud:
        return {
            'transaction_id': f'TX{np.random.randint(10000, 99999)}',
            'amount': float(np.random.lognormal(mean=12, sigma=0.7)),
            'hour': int(np.random.choice([0, 1, 2, 3, 4, 23])),
            'location': 'Foreign',
            'is_new_device': 1,
            'is_foreign_location': 1,
            'transactions_last_hour': int(np.random.poisson(lam=8)),
            'previous_transaction_amount': float(np.random.lognormal(mean=7.5, sigma=0.8))
        }
    else:
        return {
            'transaction_id': f'TX{np.random.randint(10000, 99999)}',
            'amount': float(np.random.lognormal(mean=7.5, sigma=0.8)),
            'hour': int(np.random.choice(range(8, 22))),
            'location': np.random.choice(['Columbo', 'Kandy', 'Galle']),
            'is_new_device': int(np.random.choice([0, 1], p=[0.9, 0.1])),
            'is_foreign_location': 0,
            'transactions_last_hour': int(np.random.poisson(lam=2)),
            'previous_transaction_amount': float(np.random.lognormal(mean=7.5, sigma=0.8))
        }


if __name__ == "__main__":
    producer = TransactionProducer()
    
    try:
        for i in range(10):
            transaction = generate_sample_transaction()
            producer.send_transaction(transaction)
            print(f"Sent transaction: {transaction['transaction_id']}")
            time.sleep(1)
    finally:
        producer.close()