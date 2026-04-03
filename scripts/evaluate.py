"""
Evaluation Script for Lottery Prediction Models

Trains and evaluates 4 models:
1. Random baseline
2. Frequency baseline
3. XGBoost (strong tabular baseline)
4. TabM (primary experimental model)

Computes 3 metrics: Precision@20, Recall@20, Hit-rate

Usage:
    python scripts/evaluate.py

Outputs:
    - Results table (console + CSV)
    - Comparison of all models on test set

Author: Amelia (Senior Developer)
Date: 2026-04-03
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from features import FeatureEngineer
from models.baselines import RandomBaseline, FrequencyBaseline
from models.xgboost_wrapper import XGBoostModel
from models.tabm_wrapper import TabMModel
from models.trainer import temporal_split
from models.metrics import aggregate_metrics


def load_and_prepare_data():
    """Load raw data and generate features."""
    print("=" * 80)
    print("LOTTERY PREDICTION: MODEL EVALUATION")
    print("=" * 80)

    # Load raw data
    print("\n1. Loading raw data...")
    df = pd.read_csv('data/raw/c5_Q-state.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"   Loaded {len(df):,} draws from {df['date'].min().date()} to {df['date'].max().date()}")

    # Temporal split (70/10/20)
    print("\n2. Temporal split (70/10/20)...")
    train_df, val_df, test_df = temporal_split(df, train_pct=0.7, val_pct=0.1, test_pct=0.2)

    # Generate features
    print("\n3. Generating features...")
    fe = FeatureEngineer(version='v1')

    print("   Train set...")
    X_train, y_train = fe.fit_transform(train_df)

    print("   Validation set...")
    X_val, y_val = fe.transform(val_df)

    print("   Test set...")
    X_test, y_test = fe.transform(test_df)

    print(f"\n   Feature matrix shape: {X_train.shape}")
    print(f"   Target matrix shape:  {y_train.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def extract_actual_numbers(y_matrix):
    """Extract actual 5 numbers from target matrix."""
    actuals = []
    # Convert to numpy array if DataFrame
    if isinstance(y_matrix, pd.DataFrame):
        y_array = y_matrix.values
    else:
        y_array = y_matrix

    for row in y_array:
        # Get indices where value is 1 (0-indexed)
        indices = np.where(row == 1)[0]
        # Convert to 1-indexed numbers
        numbers = [idx + 1 for idx in indices]
        actuals.append(numbers)
    return actuals


def evaluate_model(model_name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate a single model."""
    print(f"\n{'=' * 80}")
    print(f"{model_name}")
    print(f"{'=' * 80}")

    # Train
    print(f"\nTraining {model_name}...")
    if model_name in ["XGBoost", "TabM"]:
        model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=10)
    else:
        model.fit(X_train, y_train)

    # Predict on test set
    print(f"Evaluating on test set ({len(X_test):,} draws)...")
    predictions = model.predict_top_k(X_test, k=20)

    # Extract actual numbers
    actuals = extract_actual_numbers(y_test)

    # Compute metrics
    metrics = aggregate_metrics(predictions, actuals, k=20)

    print(f"\nResults:")
    print(f"  Precision@20: {metrics['precision_at_k']:.4f}")
    print(f"  Recall@20:    {metrics['recall_at_k']:.4f}")
    print(f"  Hit-rate:     {metrics['hit_rate']:.4f} ({metrics['hit_rate']*100:.2f}%)")

    return {
        'Model': model_name,
        'Precision@20': metrics['precision_at_k'],
        'Recall@20': metrics['recall_at_k'],
        'Hit-rate': metrics['hit_rate']
    }


def main():
    """Main evaluation pipeline."""
    # Load data and generate features
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()

    # Initialize models
    print("\n4. Initializing models...")
    models = {
        'Random': RandomBaseline(seed=42),
        'Frequency': FrequencyBaseline(),
        'XGBoost': XGBoostModel(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'TabM': TabMModel(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }

    # Evaluate each model
    results = []
    for model_name, model in models.items():
        result = evaluate_model(
            model_name, model,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test
        )
        results.append(result)

    # Create results table
    results_df = pd.DataFrame(results)

    # Display results
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Test Set)")
    print("=" * 80)
    print("\n" + results_df.to_string(index=False))
    print("\n")

    # Save results
    output_path = 'results_table.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    best_model = results_df.loc[results_df['Hit-rate'].idxmax()]

    print(f"\nBest model: {best_model['Model']}")
    print(f"  Hit-rate: {best_model['Hit-rate']:.4f} ({best_model['Hit-rate']*100:.2f}%)")

    # Check if TabM beat baselines
    tabm_hit = results_df[results_df['Model'] == 'TabM']['Hit-rate'].values[0]
    xgb_hit = results_df[results_df['Model'] == 'XGBoost']['Hit-rate'].values[0]
    freq_hit = results_df[results_df['Model'] == 'Frequency']['Hit-rate'].values[0]
    random_hit = results_df[results_df['Model'] == 'Random']['Hit-rate'].values[0]

    print(f"\nPerformance progression:")
    print(f"  Random:    {random_hit:.4f} ({random_hit*100:.2f}%)")
    print(f"  Frequency: {freq_hit:.4f} ({freq_hit*100:.2f}%) - {freq_hit/random_hit:.2f}x Random")
    print(f"  XGBoost:   {xgb_hit:.4f} ({xgb_hit*100:.2f}%) - {xgb_hit/random_hit:.2f}x Random")
    print(f"  TabM:      {tabm_hit:.4f} ({tabm_hit*100:.2f}%) - {tabm_hit/random_hit:.2f}x Random")

    if tabm_hit > xgb_hit:
        print(f"\n✓ TabM shows {(tabm_hit - xgb_hit)*100:.2f}% absolute improvement over XGBoost")
    elif tabm_hit >= xgb_hit * 0.95:
        print(f"\n~ TabM performs comparably to XGBoost (within 5%)")
    else:
        print(f"\n✗ TabM underperforms XGBoost by {(xgb_hit - tabm_hit)*100:.2f}%")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
