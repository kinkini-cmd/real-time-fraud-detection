"""Data preprocessing utilities for fraud detection."""

import pandas as pd
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """Load transaction data from CSV file."""
    return pd.read_csv(file_path)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    return df.drop_duplicates()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    df = df.copy()
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'is_fraud':
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    return df


def convert_timestamp(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """Convert timestamp column to datetime."""
    df = df.copy()
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps."""
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    if 'timestamp' in df.columns:
        df = convert_timestamp(df)
    return df