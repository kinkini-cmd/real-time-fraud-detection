"""Tests for monitoring and drift detection."""

import numpy as np

from src.monitoring.monitor import DataDriftDetector, SystemMonitor


def test_system_monitor_records_predictions():
    """Monitor aggregates prediction counts and fraud rate."""
    monitor = SystemMonitor()

    monitor.record_prediction({'is_fraud': 0}, processing_time=0.01)
    monitor.record_prediction({'is_fraud': 1}, processing_time=0.03)

    metrics = monitor.get_ml_metrics()

    assert metrics['total_predictions'] == 2
    assert metrics['fraud_count'] == 1
    assert metrics['legitimate_count'] == 1
    assert metrics['fraud_rate'] == 50.0
    assert metrics['avg_prediction_time_ms'] > 0


def test_system_monitor_empty():
    """Empty monitor returns zeroed metrics."""
    metrics = SystemMonitor().get_ml_metrics()

    assert metrics['total_predictions'] == 0
    assert metrics['fraud_rate'] == 0


def test_data_drift_detector_no_drift():
    """Identical distributions are not flagged as drift."""
    reference = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    detector = DataDriftDetector(reference, threshold=0.5)

    result = detector.detect_drift(reference)

    assert result['drift_detected'] is False
    assert result['max_drift_score'] < 0.5


def test_data_drift_detector_detects_shift():
    """A large mean shift is flagged as drift."""
    reference = np.array([[1.0], [2.0], [3.0]])
    detector = DataDriftDetector(reference, threshold=0.5)

    result = detector.detect_drift(reference + 10)

    assert result['drift_detected'] is True
