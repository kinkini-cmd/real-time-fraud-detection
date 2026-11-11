"""Kafka consumer for processing transactions."""

import json
from typing import Dict, Any, Callable
from kafka import KafkaConsumer


class TransactionConsumer:
    """Consumes transaction events from Kafka and processes them."""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092",
                 topic: str = "transactions",
                 group_id: str = "fraud-detection-group"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        self.running = False
        self.processing_callback = None
    
    def set_processing_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback function to process transactions."""
        self.processing_callback = callback
    
    def start_consuming(self):
        """Start consuming messages from Kafka."""
        self.running = True
        print(f"Starting consumer for topic: {self.topic}")
        
        for message in self.consumer:
            if not self.running:
                break
            
            transaction = message.value
            print(f"Received transaction: {transaction.get('transaction_id')}")
            
            if self.processing_callback:
                self.processing_callback(transaction)
    
    def stop(self):
        """Stop consuming messages."""
        self.running = False
        self.consumer.close()
        print("Consumer stopped")


if __name__ == "__main__":
    def process_transaction(transaction):
        print(f"Processing: {transaction}")
    
    consumer = TransactionConsumer()
    consumer.set_processing_callback(process_transaction)
    
    try:
        consumer.start_consuming()
    except KeyboardInterrupt:
        consumer.stop()