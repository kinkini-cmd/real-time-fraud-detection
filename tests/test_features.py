"""Tests for feature engineering."""

import pandas as pd

from src.features.feature_engineering import (
    create_amount_features,
    create_behavioral_features,
    engineer_features,
    extract_temporal_features,
    get_feature_columns,
)


def test_extract_temporal_features():
    """Temporal features are derived from the timestamp."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 10:00:00',  # Monday, daytime
            '2024-01-06 23:00:00',  # Saturday, night
        ])
    })

    result = extract_temporal_features(df)

    assert result['hour'].tolist() == [10, 23]
    assert result['day_of_week'].tolist() == [0, 5]
    assert result['is_weekend'].tolist() == [0, 1]
    assert result['is_night'].tolist() == [0, 1]


def test_create_amount_features():
    """Amount features compare the current and previous amounts."""
    df = pd.DataFrame({
        'amount': [200000, 1000],
        'previous_transaction_amount': [100000, 2000],
    })

    result = create_amount_features(df)

    assert result['amount_change'].tolist() == [100000, -1000]
    assert result['amount_ratio'].tolist() == [200000 / 100001, 1000 / 2001]
    assert result['is_high_amount'].tolist() == [1, 0]


def test_create_behavioral_features():
    """High-frequency transactions are flagged."""
    df = pd.DataFrame({'transactions_last_hour': [2, 10]})

    result = create_behavioral_features(df)

    assert result['is_high_frequency'].tolist() == [0, 1]


def test_engineer_features_pipeline():
    """The full pipeline adds all expected columns."""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-03-03 02:00:00']),
        'amount': [500000],
        'previous_transaction_amount': [1000],
        'transactions_last_hour': [9],
    })

    result = engineer_features(df)

    for column in ['hour', 'day_of_week', 'is_weekend', 'is_night',
                   'amount_change', 'amount_ratio', 'is_high_amount',
                   'is_high_frequency']:
        assert column in result.columns


def test_get_feature_columns():
    """Feature columns are returned as a non-empty list."""
    columns = get_feature_columns()
    assert isinstance(columns, list)
    assert 'amount' in columns
