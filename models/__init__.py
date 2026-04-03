"""
Models Package for c5_Q-state Lottery Prediction

Exports:
    - baselines: Random and Frequency baseline models
    - xgboost_wrapper: XGBoost wrapper (strong tabular baseline)
    - tabm_wrapper: TabM wrapper (primary experimental model)
    - trainer: Training utilities (temporal split, early stopping)
    - metrics: Evaluation metrics (Precision@20, Recall@20, Hit-rate)
"""

from .baselines import RandomBaseline, FrequencyBaseline
from .metrics import precision_at_k, recall_at_k, hit_rate
from .trainer import temporal_split, train_with_early_stopping

__all__ = [
    'RandomBaseline',
    'FrequencyBaseline',
    'precision_at_k',
    'recall_at_k',
    'hit_rate',
    'temporal_split',
    'train_with_early_stopping',
]
