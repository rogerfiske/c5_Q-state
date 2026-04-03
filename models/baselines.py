"""
Baseline Models for Lottery Prediction

Implements two baselines:
1. RandomBaseline: Randomly select top-20 numbers
2. FrequencyBaseline: Select top-20 most frequent numbers from training set

These establish the performance floor for TabM comparison.

Author: Amelia (Senior Developer)
Date: 2026-04-03
"""

import numpy as np
import pandas as pd
from typing import List
from collections import Counter


class RandomBaseline:
    """
    Random baseline: Randomly select 20 numbers from 1-39.

    Expected hit-rate: ~2.4% (C(20,5) / C(39,5))
    """

    def __init__(self, seed: int = 42):
        """
        Initialize random baseline.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def fit(self, X_train, y_train):
        """No training needed for random baseline."""
        return self

    def predict_top_k(self, X, k: int = 20) -> List[List[int]]:
        """
        Predict top-K numbers (randomly selected).

        Args:
            X: Feature matrix (not used, but kept for API consistency)
            k: Number of predictions per draw

        Returns:
            List of predictions (one list of K numbers per draw)
        """
        n_samples = len(X)
        predictions = []

        for _ in range(n_samples):
            # Randomly sample k numbers from 1-39
            pred = self.rng.choice(range(1, 40), size=k, replace=False).tolist()
            predictions.append(pred)

        return predictions


class FrequencyBaseline:
    """
    Frequency baseline: Select top-20 most frequent numbers from training set.

    Expected hit-rate: ~5-10% (exploits non-uniform distribution)
    """

    def __init__(self):
        """Initialize frequency baseline."""
        self.top_k_numbers = None

    def fit(self, X_train, y_train):
        """
        Fit by counting number frequencies in training set.

        Args:
            X_train: Feature matrix (not used)
            y_train: Target matrix (N × 39 binary matrix)

        Returns:
            self
        """
        # Count frequencies of each number (1-39) in training set
        # y_train[:, i] = 1 if number (i+1) appears
        frequencies = y_train.sum(axis=0)  # Sum over all draws

        # Get indices of top-20 most frequent (0-indexed)
        top_k_indices = np.argsort(frequencies)[-20:][::-1]

        # Convert to 1-indexed numbers
        self.top_k_numbers = [idx + 1 for idx in top_k_indices]

        return self

    def predict_top_k(self, X, k: int = 20) -> List[List[int]]:
        """
        Predict top-K numbers (same top-20 for all draws).

        Args:
            X: Feature matrix (not used, but kept for API consistency)
            k: Number of predictions per draw

        Returns:
            List of predictions (one list of K numbers per draw)
        """
        if self.top_k_numbers is None:
            raise ValueError("Model must be fit before predict")

        n_samples = len(X)

        # Return same top-k numbers for all samples
        predictions = [self.top_k_numbers[:k] for _ in range(n_samples)]

        return predictions
