"""Monitoring and metrics collection for fraud detection system."""

import time
import psutil
from typing import Dict, Any
from collections import deque
import numpy as np


class SystemMonitor:
    """Monitors system performance and ML metrics."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.predictions = deque(maxlen=max_samples)
        self.prediction_times = deque(maxlen=max_samples)
        self.errors = deque(maxlen=max_samples)
        self.start_time = time.time()
    
    def record_prediction(self, prediction_result: Dict[str, Any], processing_time: float):
        """Record a prediction result."""
        self.predictions.append(prediction_result)
        self.prediction_times.append(processing_time)
    
    def record_error(self, error: str):
        """Record an error."""
        self.errors.append({
            'timestamp': time.time(),
            'error': error
        })
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time
        }
    
    def get_ml_metrics(self) -> Dict[str, Any]:
        """Get ML-related metrics."""
        if not self.predictions:
            return {
                'total_predictions': 0,
                'fraud_rate': 0,
                'avg_prediction_time': 0
            }
        
        total = len(self.predictions)
        fraud_count = sum(1 for p in self.predictions if p.get('is_fraud') == 1)
        fraud_rate = (fraud_count / total) * 100
        
        avg_time = float(np.mean(self.prediction_times)) if self.prediction_times else 0.0
        
        return {
            'total_predictions': total,
            'fraud_count': fraud_count,
            'legitimate_count': total - fraud_count,
            'fraud_rate': fraud_rate,
            'avg_prediction_time_ms': avg_time * 1000
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            'system': self.get_system_metrics(),
            'ml': self.get_ml_metrics(),
            'errors': len(self.errors)
        }


class DataDriftDetector:
    """Detects data drift between training and production data."""
    
    def __init__(self, reference_data: np.ndarray, threshold: float = 0.1):
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)
    
    def detect_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect if there's significant data drift."""
        current_mean = np.mean(current_data, axis=0)
        current_std = np.std(current_data, axis=0)
        
        # Calculate normalized difference
        diff = np.abs(current_mean - self.reference_mean) / (self.reference_std + 1e-8)
        max_diff = np.max(diff)
        
        return {
            'drift_detected': bool(max_diff > self.threshold),
            'max_drift_score': float(max_diff),
            'threshold': self.threshold,
            'reference_mean': self.reference_mean.tolist(),
            'current_mean': current_mean.tolist()
        }


if __name__ == "__main__":
    monitor = SystemMonitor()
    
    # Simulate some predictions
    for i in range(10):
        monitor.record_prediction({
            'is_fraud': i % 2,
            'fraud_probability': 0.5 + i * 0.05
        }, processing_time=0.01 + i * 0.001)
    
    print("System Metrics:", monitor.get_all_metrics())