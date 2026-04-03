"""
Evaluation Metrics for Lottery Prediction

Implements three success metrics:
- Precision@K: (# actual numbers in top-K) / K
- Recall@K: (# actual numbers in top-K) / (# actual numbers)
- Hit-rate: Binary (all actual numbers in top-K?)

Author: Amelia (Senior Developer)
Date: 2026-04-03
"""

from typing import List, Set
import numpy as np


def precision_at_k(predicted: List[int], actual: List[int], k: int = 20) -> float:
    """
    Compute Precision@K.

    Precision@K = (# actual numbers in top-K predictions) / K

    Args:
        predicted: List of predicted numbers (top-K)
        actual: List of actual numbers (5 numbers)
        k: Top-K cutoff (default: 20)

    Returns:
        Precision@K ∈ [0, 0.25] (max when all 5 actual numbers in top-20)
    """
    if len(predicted) == 0 or k == 0:
        return 0.0

    top_k = set(predicted[:k])
    actual_set = set(actual)

    overlap = len(top_k & actual_set)
    return overlap / k


def recall_at_k(predicted: List[int], actual: List[int], k: int = 20) -> float:
    """
    Compute Recall@K.

    Recall@K = (# actual numbers in top-K predictions) / (# actual numbers)

    Args:
        predicted: List of predicted numbers (top-K)
        actual: List of actual numbers (5 numbers)
        k: Top-K cutoff (default: 20)

    Returns:
        Recall@K ∈ [0, 1.0]
    """
    if len(actual) == 0:
        return 0.0

    top_k = set(predicted[:k])
    actual_set = set(actual)

    overlap = len(top_k & actual_set)
    return overlap / len(actual)


def hit_rate(predicted: List[int], actual: List[int], k: int = 20) -> int:
    """
    Compute Hit-rate (binary metric).

    Hit-rate = 1 if ALL actual numbers are in top-K predictions, else 0

    Args:
        predicted: List of predicted numbers (top-K)
        actual: List of actual numbers (5 numbers)
        k: Top-K cutoff (default: 20)

    Returns:
        1 if all actual numbers in top-K, else 0
    """
    top_k = set(predicted[:k])
    actual_set = set(actual)

    return 1 if actual_set.issubset(top_k) else 0


def compute_all_metrics(predicted: List[int], actual: List[int], k: int = 20) -> dict:
    """
    Compute all three metrics at once.

    Args:
        predicted: List of predicted numbers (top-K)
        actual: List of actual numbers (5 numbers)
        k: Top-K cutoff (default: 20)

    Returns:
        Dictionary with Precision@K, Recall@K, Hit-rate
    """
    return {
        'precision_at_k': precision_at_k(predicted, actual, k),
        'recall_at_k': recall_at_k(predicted, actual, k),
        'hit_rate': hit_rate(predicted, actual, k)
    }


def aggregate_metrics(all_predicted: List[List[int]],
                     all_actual: List[List[int]],
                     k: int = 20) -> dict:
    """
    Compute average metrics across multiple predictions.

    Args:
        all_predicted: List of predictions (one per draw)
        all_actual: List of actual draws (one per draw)
        k: Top-K cutoff (default: 20)

    Returns:
        Dictionary with averaged metrics
    """
    precisions = []
    recalls = []
    hits = []

    for pred, actual in zip(all_predicted, all_actual):
        precisions.append(precision_at_k(pred, actual, k))
        recalls.append(recall_at_k(pred, actual, k))
        hits.append(hit_rate(pred, actual, k))

    return {
        'precision_at_k': np.mean(precisions),
        'recall_at_k': np.mean(recalls),
        'hit_rate': np.mean(hits)  # Proportion of draws with all 5 numbers in top-K
    }
