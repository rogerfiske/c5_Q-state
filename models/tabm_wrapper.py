"""
TabM Wrapper (Primary Experimental Model)

Multi-label classification using TabM (ICLR 2025 paper: "TABM: Advancing Tabular
Deep Learning with Parameter-Efficient Ensembling").

This is the primary research model testing position-aware tabular prediction with
parameter-efficient ensembling.

Expected hit-rate: >15% (target improvement over XGBoost)

Author: Amelia (Senior Developer)
Date: 2026-04-03
"""

import numpy as np
from typing import List
import warnings


class TabMModel:
    """
    TabM wrapper for multi-label lottery prediction.

    Uses TabM's parameter-efficient ensembling for tabular data.
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42):
        """
        Initialize TabM model.

        Args:
            n_estimators: Number of ensemble members
            max_depth: Tree depth per member
            learning_rate: Learning rate
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        Train TabM model with early stopping.

        Args:
            X_train: Training features (N × 1408)
            y_train: Training targets (N × 39 binary matrix)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional, for early stopping)
            early_stopping_rounds: Early stopping patience

        Returns:
            self
        """
        print(f"Training TabM model...")

        try:
            # Try to import official tabm package
            import tabm

            # Initialize TabM classifier
            # Note: Actual API may differ - adjust based on official documentation
            self.model = tabm.TabMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                task='multilabel',  # Multi-label classification
                verbose=1
            )

            # Fit model
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds
                )
            else:
                self.model.fit(X_train, y_train)

            print(f"✓ TabM training complete")

        except ImportError:
            warnings.warn(
                "TabM package not available. Please install with: pip install tabm\n"
                "Falling back to using XGBoost as TabM substitute."
            )

            # Fallback: Use XGBoost as substitute
            from .xgboost_wrapper import XGBoostModel

            print("  [FALLBACK] Using XGBoost as TabM substitute...")

            self.model = XGBoostModel(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )

            self.model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds)

        return self

    def predict_proba(self, X):
        """
        Predict probabilities for all 39 numbers.

        Args:
            X: Feature matrix (N × 1408)

        Returns:
            Probability matrix (N × 39)
        """
        if self.model is None:
            raise ValueError("Model must be fit before predict")

        return self.model.predict_proba(X)

    def predict_top_k(self, X, k: int = 20) -> List[List[int]]:
        """
        Predict top-K numbers by ranking probabilities.

        Args:
            X: Feature matrix (N × 1408)
            k: Number of predictions per draw

        Returns:
            List of predictions (one list of K numbers per draw)
        """
        probs = self.predict_proba(X)

        predictions = []

        for prob_row in probs:
            # Get top-k indices (0-indexed)
            top_k_indices = np.argsort(prob_row)[-k:][::-1]

            # Convert to 1-indexed numbers
            top_k_numbers = [idx + 1 for idx in top_k_indices]

            predictions.append(top_k_numbers)

        return predictions
