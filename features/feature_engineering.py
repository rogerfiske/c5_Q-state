"""
Feature Engineering Module for c5_Q-state Lottery Prediction

Implements position-aware tabular features based on Mary's strategic analysis:
- Priority 1: Position-specific recency features
- Priority 2: Relational gap features
- Priority 3: Boundary indicators
- Priority 4: Rolling frequency features
- Additional: Momentum indicators

Author: Amelia (Senior Developer)
Date: 2026-04-03
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering pipeline for lottery prediction.

    Transforms raw lottery draws into position-aware tabular features
    suitable for TabM ensembling.

    Key Principles:
    - NO LEAKAGE: Features for row i use only data from rows 0 to i
    - Target y[i] represents draw i+1
    - Position-specific features exploit predictability differences
    - Efficient pandas operations (avoid slow loops)

    Usage:
        fe = FeatureEngineer(version='v1')
        X_train, y_train = fe.fit_transform(train_df)
        X_val, y_val = fe.transform(val_df)
    """

    def __init__(self, version='v1'):
        """
        Initialize FeatureEngineer.

        Args:
            version: Feature version for reproducibility tracking
        """
        self.version = version
        self.position_cols = ['QS_1', 'QS_2', 'QS_3', 'QS_4', 'QS_5']
        self.numbers = list(range(1, 40))  # 1-39
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit on training data and transform.

        Args:
            df: Raw DataFrame with columns [date, QS_1, QS_2, QS_3, QS_4, QS_5]

        Returns:
            X: Feature DataFrame (N-1 rows, ~800+ columns)
            y: Target DataFrame (N-1 rows, 39 columns) - binary encoding of next draw
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        self.fitted = True

        # Generate features
        X = self._generate_features(df)

        # Generate targets (next draw)
        y = self._generate_targets(df)

        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform validation/test data (must call fit_transform first).

        Args:
            df: Raw DataFrame

        Returns:
            X, y: Feature and target DataFrames
        """
        if not self.fitted:
            raise ValueError("Must call fit_transform before transform")

        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        X = self._generate_features(df)
        y = self._generate_targets(df)

        return X, y

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from raw data.

        Args:
            df: Raw DataFrame

        Returns:
            X: Feature DataFrame
        """
        n = len(df)
        features_list = []

        print(f"Generating features for {n} draws...")

        # Initialize feature collectors
        all_features = []

        # PRIORITY 2: Relational Gap Features (current draw - fast to compute)
        print("  [Priority 2] Relational gap features...")
        relational = self._relational_features(df)
        all_features.append(relational)

        # PRIORITY 3: Boundary Indicators (current draw)
        print("  [Priority 3] Boundary indicators...")
        boundary = self._boundary_features(df)
        all_features.append(boundary)

        # PRIORITY 1: Position-Specific Recency Features (requires lookback)
        print("  [Priority 1] Position-specific recency features...")
        recency = self._recency_features(df)
        all_features.append(recency)

        # PRIORITY 4: Rolling Frequency Features
        print("  [Priority 4] Rolling frequency features...")
        rolling = self._rolling_frequency_features(df)
        all_features.append(rolling)

        # ADDITIONAL: Momentum Indicators
        print("  [Additional] Momentum indicators...")
        momentum = self._momentum_features(df)
        all_features.append(momentum)

        # Combine all features
        X = pd.concat(all_features, axis=1)

        # Drop last row (no next draw to predict)
        X = X.iloc[:-1].reset_index(drop=True)

        # Fill NaN values (first few rows may have incomplete rolling windows)
        X = X.fillna(0)

        print(f"  Generated {X.shape[1]} features for {X.shape[0]} draws")

        return X

    def _generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate target matrix: binary encoding of next draw.

        Args:
            df: Raw DataFrame

        Returns:
            y: (N-1, 39) binary matrix where y[i, num-1] = 1 if num appears in draw i+1
        """
        n = len(df)

        # Initialize target matrix
        y_data = np.zeros((n - 1, 39), dtype=np.int8)

        # For each draw i, encode draw i+1
        for i in range(n - 1):
            next_draw = df.iloc[i + 1][self.position_cols].values
            for num in next_draw:
                y_data[i, int(num) - 1] = 1  # num ranges 1-39, index 0-38

        # Convert to DataFrame
        y = pd.DataFrame(
            y_data,
            columns=[f'target_num_{num}' for num in self.numbers]
        )

        return y

    # =========================================================================
    # PRIORITY 1: Position-Specific Recency Features
    # =========================================================================

    def _recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate position-specific recency features.

        Features:
        - draws_since_last_appearance[num][pos]: Draws since num last appeared in pos
        - is_hot_10/50/100[num][pos]: Binary indicators (appeared in last N draws)

        Returns:
            DataFrame with recency features
        """
        n = len(df)
        features = {}

        # For each position
        for pos_idx, pos in enumerate(self.position_cols):
            # Track last seen index for each number in this position
            last_seen = {}

            for i in range(n):
                current_num = df.iloc[i][pos]

                # For each possible number, compute draws since last appearance
                for num in self.numbers:
                    if num in last_seen:
                        draws_since = i - last_seen[num]
                    else:
                        draws_since = i + 1  # Never seen before

                    # Store feature
                    feat_name = f'draws_since_{num}_in_{pos}'
                    if feat_name not in features:
                        features[feat_name] = []
                    features[feat_name].append(draws_since)

                    # is_hot indicators
                    for window in [10, 50, 100]:
                        is_hot = 1 if draws_since <= window else 0
                        hot_name = f'is_hot_{window}_{num}_in_{pos}'
                        if hot_name not in features:
                            features[hot_name] = []
                        features[hot_name].append(is_hot)

                # Update last seen
                last_seen[int(current_num)] = i

        return pd.DataFrame(features)

    # =========================================================================
    # PRIORITY 2: Relational Gap Features
    # =========================================================================

    def _relational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate relational gap features from current draw.

        Features:
        - gap_QS1_QS2, gap_QS2_QS3, gap_QS3_QS4, gap_QS4_QS5
        - total_spread (QS_5 - QS_1)
        - compression_ratio (total_spread / 38)

        Returns:
            DataFrame with relational features
        """
        features = {}

        # Gap between adjacent positions
        for i in range(len(self.position_cols) - 1):
            col1 = self.position_cols[i]
            col2 = self.position_cols[i + 1]
            gap_name = f'gap_{col1}_{col2}'
            features[gap_name] = (df[col2] - df[col1]).values

        # Total spread
        features['total_spread'] = (df['QS_5'] - df['QS_1']).values

        # Compression ratio (normalized 0-1)
        features['compression_ratio'] = features['total_spread'] / 38.0

        return pd.DataFrame(features)

    # =========================================================================
    # PRIORITY 3: Boundary Indicators
    # =========================================================================

    def _boundary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate boundary indicator features.

        Features:
        - is_1_in_QS1, is_39_in_QS5: Current draw boundary indicators
        - boundary_freq_hist[pos]: Historical frequency of boundary values per position

        Returns:
            DataFrame with boundary features
        """
        n = len(df)
        features = {}

        # Current draw boundary indicators
        features['is_1_in_QS1'] = (df['QS_1'] == 1).astype(int).values
        features['is_39_in_QS5'] = (df['QS_5'] == 39).astype(int).values

        # Historical boundary frequencies per position (rolling cumulative)
        for pos in self.position_cols:
            # Frequency of value 1 in this position (cumulative up to current row)
            freq_1 = []
            freq_39 = []

            count_1 = 0
            count_39 = 0

            for i in range(n):
                if i > 0:  # Avoid division by zero
                    freq_1.append(count_1 / i)
                    freq_39.append(count_39 / i)
                else:
                    freq_1.append(0)
                    freq_39.append(0)

                # Update counts
                if df.iloc[i][pos] == 1:
                    count_1 += 1
                if df.iloc[i][pos] == 39:
                    count_39 += 1

            features[f'boundary_freq_1_hist_{pos}'] = freq_1
            features[f'boundary_freq_39_hist_{pos}'] = freq_39

        return pd.DataFrame(features)

    # =========================================================================
    # PRIORITY 4: Rolling Frequency Features
    # =========================================================================

    def _rolling_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rolling frequency features.

        Features:
        - rolling_freq_50/500/1000[num][pos]: Frequency of num in pos over last N draws

        Returns:
            DataFrame with rolling frequency features
        """
        n = len(df)
        features = {}

        windows = [50, 500, 1000]

        # For each position and window size
        for pos in self.position_cols:
            for window in windows:
                # For each number, compute rolling frequency
                for num in self.numbers:
                    freq_list = []

                    for i in range(n):
                        # Look back 'window' draws
                        start_idx = max(0, i - window)
                        lookback = df.iloc[start_idx:i][pos]

                        if len(lookback) > 0:
                            freq = (lookback == num).sum() / len(lookback)
                        else:
                            freq = 0.0

                        freq_list.append(freq)

                    feat_name = f'rolling_freq_{window}_{num}_in_{pos}'
                    features[feat_name] = freq_list

        return pd.DataFrame(features)

    # =========================================================================
    # ADDITIONAL: Momentum Indicators
    # =========================================================================

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum indicator features.

        Features:
        - consecutive_repeats[pos]: Did same number repeat in this position?
        - position_mean/std[pos]: Mean/std of position over last N draws

        Returns:
            DataFrame with momentum features
        """
        n = len(df)
        features = {}

        # Consecutive repeats per position
        for pos in self.position_cols:
            repeats = [0]  # First draw has no previous

            for i in range(1, n):
                if df.iloc[i][pos] == df.iloc[i-1][pos]:
                    repeats.append(1)
                else:
                    repeats.append(0)

            features[f'consecutive_repeat_{pos}'] = repeats

        # Position-specific rolling statistics
        for pos in self.position_cols:
            for window in [50, 500]:
                mean_list = []
                std_list = []

                for i in range(n):
                    start_idx = max(0, i - window)
                    lookback = df.iloc[start_idx:i][pos]

                    if len(lookback) > 0:
                        mean_list.append(lookback.mean())
                        std_list.append(lookback.std() if len(lookback) > 1 else 0.0)
                    else:
                        mean_list.append(0.0)
                        std_list.append(0.0)

                features[f'rolling_mean_{window}_{pos}'] = mean_list
                features[f'rolling_std_{window}_{pos}'] = std_list

        return pd.DataFrame(features)

    def save(self, filepath: str):
        """Save FeatureEngineer state for reproducibility."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"FeatureEngineer saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Load FeatureEngineer from file."""
        import pickle
        with open(filepath, 'rb') as f:
            fe = pickle.load(f)
        print(f"FeatureEngineer loaded from {filepath}")
        return fe
