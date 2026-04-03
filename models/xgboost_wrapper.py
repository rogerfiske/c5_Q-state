"""
XGBoost Wrapper (Strong Tabular Baseline)

Multi-label classification using XGBoost. Each of 39 numbers is a binary target.
Train single XGBoost model with 39 outputs. At inference, rank by probability and
select top-20.

This serves as the strong tabular baseline to compare against TabM.

Expected hit-rate: ~10-15%

Author: Amelia (Senior Developer)
Date: 2026-04-03
"""

import numpy as np
import xgboost as xgb
from typing import List


class XGBoostModel:
    """
    XGBoost wrapper for multi-label lottery prediction.

    Uses 39 binary XGBoost classifiers (one per number 1-39).
    """

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.models = []  # One model per number

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        Train 39 binary XGBoost classifiers.

        Args:
            X_train: Training features (N × 1408)
            y_train: Training targets (N × 39 binary matrix)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional, for early stopping)
            early_stopping_rounds: Early stopping patience

        Returns:
            self
        """
        print(f"Training XGBoost models (39 binary classifiers)...")

        self.models = []

        # Train one model per number (39 models total)
        for i in range(39):
            print(f"  Training model for number {i+1}/39...", end='\r')

            # Binary target for this number
            y_train_i = y_train[:, i]

            # XGBoost parameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'random_state': self.random_state,
                'tree_method': 'hist',  # Faster
                'verbosity': 0  # Suppress warnings
            }

            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train_i)

            # Setup evaluation sets
            evals = [(dtrain, 'train')]
            if X_val is not None and y_val is not None:
                y_val_i = y_val[:, i]
                dval = xgb.DMatrix(X_val, label=y_val_i)
                evals.append((dval, 'val'))

            # Train
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
                verbose_eval=False
            )

            self.models.append(model)

        print(f"\n✓ XGBoost training complete (39 models trained)")

        return self

    def predict_proba(self, X):
        """
        Predict probabilities for all 39 numbers.

        Args:
            X: Feature matrix (N × 1408)

        Returns:
            Probability matrix (N × 39)
        """
        if not self.models:
            raise ValueError("Model must be fit before predict")

        dtest = xgb.DMatrix(X)

        # Predict probabilities for each number
        probs = np.zeros((len(X), 39))

        for i, model in enumerate(self.models):
            probs[:, i] = model.predict(dtest)

        return probs

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
