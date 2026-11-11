"""Tests for data preprocessing."""

import pandas as pd
import numpy as np
from src.data.preprocessing import (
    remove_duplicates,
    handle_missing_values,
    preprocess_data
)


def test_remove_duplicates():
    """Test duplicate removal."""
    df = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': [4, 5, 5, 6]
    })
    result = remove_duplicates(df)
    assert len(result) == 3


def test_handle_missing_values():
    """Test missing value handling."""
    df = pd.DataFrame({
        'amount': [100, np.nan, 300],
        'location': ['A', None, 'C']
    })
    result = handle_missing_values(df)
    assert result['amount'].isnull().sum() == 0
    assert result['location'].isnull().sum() == 0
    assert result['location'][1] == 'Unknown'


def test_preprocess_data():
    """Test full preprocessing pipeline."""
    df = pd.DataFrame({
        'transaction_id': ['A', 'B', 'C'],
        'amount': [100, 200, np.nan],
        'location': ['A', None, 'C'],
        'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03']
    })
    result = preprocess_data(df)
    assert len(result) == 3
    assert result['amount'].isnull().sum() == 0