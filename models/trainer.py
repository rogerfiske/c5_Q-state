"""
Training Utilities

Implements:
- temporal_split(): 70/10/20 chronological split (no shuffling)
- train_with_early_stopping(): Training loop with validation-based early stopping

Author: Amelia (Senior Developer)
Date: 2026-04-03
"""

import pandas as pd
import numpy as np
from typing import Tuple


def temporal_split(df: pd.DataFrame,
                   train_pct: float = 0.7,
                   val_pct: float = 0.1,
                   test_pct: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (no shuffling) into train/val/test.

    Prevents temporal leakage by respecting time order.

    Args:
        df: DataFrame sorted by date
        train_pct: Percentage for training (default: 70%)
        val_pct: Percentage for validation (default: 10%)
        test_pct: Percentage for test (default: 20%)

    Returns:
        (train_df, val_df, test_df)

    Example:
        For 11,756 draws:
        - Train: 8,229 draws (1992-2015)
        - Val:   1,176 draws (2015-2018)
        - Test:  2,351 draws (2018-2026)
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, "Percentages must sum to 1.0"

    if 'date' in df.columns:
        assert df['date'].is_monotonic_increasing, "Data must be sorted by date"

    n = len(df)
    train_idx = int(n * train_pct)
    val_idx = train_idx + int(n * val_pct)

    train = df.iloc[:train_idx].copy()
    val = df.iloc[train_idx:val_idx].copy()
    test = df.iloc[val_idx:].copy()

    print(f"Temporal split:")
    print(f"  Train: {len(train):,} draws ({train_pct*100:.0f}%)")
    print(f"  Val:   {len(val):,} draws ({val_pct*100:.0f}%)")
    print(f"  Test:  {len(test):,} draws ({test_pct*100:.0f}%)")

    if 'date' in df.columns:
        print(f"  Train dates: {train['date'].min().date()} to {train['date'].max().date()}")
        print(f"  Val dates:   {val['date'].min().date()} to {val['date'].max().date()}")
        print(f"  Test dates:  {test['date'].min().date()} to {test['date'].max().date()}")

    return train, val, test


def train_with_early_stopping(model,
                               X_train,
                               y_train,
                               X_val,
                               y_val,
                               early_stopping_rounds: int = 10):
    """
    Train model with validation-based early stopping.

    Args:
        model: Model instance with fit() method
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        early_stopping_rounds: Patience before stopping

    Returns:
        Trained model
    """
    # Delegate to model's fit method with early stopping
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        early_stopping_rounds=early_stopping_rounds
    )

    return model
