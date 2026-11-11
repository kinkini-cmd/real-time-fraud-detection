"""Feature engineering for fraud detection."""

import pandas as pd
import numpy as np


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from timestamp."""
    df = df.copy()
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 20) | (df['hour'] <= 4)).astype(int)
    return df


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create amount-based features."""
    df = df.copy()
    if {'amount', 'previous_transaction_amount'}.issubset(df.columns):
        df['amount_change'] = df['amount'] - df['previous_transaction_amount']
        df['amount_ratio'] = df['amount'] / (df['previous_transaction_amount'] + 1)
        df['is_high_amount'] = (df['amount'] > 100000).astype(int)
    return df


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create behavioral features."""
    df = df.copy()
    if 'transactions_last_hour' in df.columns:
        df['is_high_frequency'] = (df['transactions_last_hour'] > 5).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = extract_temporal_features(df)
    df = create_amount_features(df)
    df = create_behavioral_features(df)
    return df


def get_feature_columns() -> list:
    """Return list of feature columns for model training."""
    return [
        'amount',
        'hour',
        'is_new_device',
        'is_foreign_location',
        'transactions_last_hour',
        'previous_transaction_amount',
        'amount_change',
        'amount_ratio',
        'is_high_amount',
        'is_high_frequency',
        'is_weekend',
        'is_night',
        'day_of_week'
    ]