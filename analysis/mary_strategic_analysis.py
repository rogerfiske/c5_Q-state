"""
Strategic Business Analysis: c5_Q-state Lottery Dataset
Analyst: Mary (Senior Business Analyst)
Purpose: Identify per-column differences, cylindrical properties, and tabular learning opportunities

This analysis will reveal:
1. Per-column predictability differences (QS_1 vs QS_5 behavior)
2. Distribution overlaps between positions
3. Cylindrical adjacency patterns (39 ↔ 1 wrapping)
4. Strategic insights for TabM feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load dataset
df = pd.read_csv(r"c:\Users\Minis\CascadeProjects\c5_Q-state\data\raw\c5_Q-state.csv")
df['date'] = pd.to_datetime(df['date'])

print("=" * 80)
print("STRATEGIC ANALYSIS: c5_Q-state Lottery Dataset")
print("=" * 80)
print(f"\nDataset Shape: {df.shape}")
print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
print(f"Total Draws: {len(df):,}")

# ============================================================================
# PART 1: PER-COLUMN PREDICTABILITY DIFFERENCES
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: PER-COLUMN PREDICTABILITY DIFFERENCES")
print("=" * 80)

columns = ['QS_1', 'QS_2', 'QS_3', 'QS_4', 'QS_5']

# Basic statistics per column
print("\n--- Basic Statistics Per Column ---")
stats_summary = df[columns].describe()
print(stats_summary)

# Additional metrics
print("\n--- Advanced Metrics Per Column ---")
for col in columns:
    values = df[col].values
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()} / 39")
    print(f"  Skewness: {stats.skew(values):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(values):.4f}")
    print(f"  Coefficient of Variation: {(values.std() / values.mean()):.4f}")

    # Autocorrelation at lag 1
    autocorr_1 = pd.Series(values).autocorr(lag=1)
    print(f"  Autocorrelation (lag=1): {autocorr_1:.4f}")

    # Entropy (as predictability proxy)
    value_counts = pd.Series(values).value_counts()
    probs = value_counts / len(values)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(39)  # Maximum possible entropy for 39 values
    normalized_entropy = entropy / max_entropy
    print(f"  Shannon Entropy: {entropy:.4f} (normalized: {normalized_entropy:.4f})")
    print(f"  Predictability Proxy (1 - norm_entropy): {(1 - normalized_entropy):.4f}")

# Frequency distribution per column
print("\n--- Top 10 Most Frequent Values Per Column ---")
for col in columns:
    top_10 = df[col].value_counts().head(10)
    print(f"\n{col}:")
    for val, count in top_10.items():
        freq = count / len(df) * 100
        print(f"  {val:2d}: {count:4d} ({freq:5.2f}%)")

# ============================================================================
# PART 2: OVERLAPPING DISTRIBUTIONS
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: OVERLAPPING DISTRIBUTIONS")
print("=" * 80)

# Quantile ranges per column
print("\n--- Quantile Ranges Per Column ---")
quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
for col in columns:
    q_values = df[col].quantile(quantiles)
    print(f"\n{col}:")
    for q, val in zip(quantiles, q_values):
        print(f"  {int(q*100):2d}th percentile: {val:5.1f}")
    print(f"  IQR: {q_values[0.75] - q_values[0.25]:.1f}")

# Distribution overlap analysis
print("\n--- Distribution Overlap Between Adjacent Positions ---")
for i in range(len(columns) - 1):
    col1, col2 = columns[i], columns[i + 1]

    # Calculate overlap percentage
    min1, max1 = df[col1].min(), df[col1].max()
    min2, max2 = df[col2].min(), df[col2].max()

    overlap_min = max(min1, min2)
    overlap_max = min(max1, max2)
    overlap_range = max(0, overlap_max - overlap_min + 1)

    range1 = max1 - min1 + 1
    range2 = max2 - min2 + 1

    overlap_pct = (overlap_range / min(range1, range2)) * 100

    print(f"\n{col1} vs {col2}:")
    print(f"  {col1} range: [{min1}, {max1}] (span: {range1})")
    print(f"  {col2} range: [{min2}, {max2}] (span: {range2})")
    print(f"  Overlap: [{overlap_min}, {overlap_max}] (span: {overlap_range}, {overlap_pct:.1f}%)")

    # Kolmogorov-Smirnov test for distribution similarity
    ks_stat, ks_pval = stats.ks_2samp(df[col1], df[col2])
    print(f"  KS Test: statistic={ks_stat:.4f}, p-value={ks_pval:.4e}")
    if ks_pval < 0.001:
        print(f"  → Distributions are SIGNIFICANTLY DIFFERENT (p < 0.001)")
    else:
        print(f"  → Distributions are similar")

# ============================================================================
# PART 3: CYLINDRICAL ADJACENCY PROPERTIES (39 ↔ 1 wrap-around)
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: CYLINDRICAL ADJACENCY PROPERTIES (39 ↔ 1 WRAP-AROUND)")
print("=" * 80)

# Boundary value frequencies
print("\n--- Boundary Value Frequencies (1 and 39) ---")
for col in columns:
    count_1 = (df[col] == 1).sum()
    count_39 = (df[col] == 39).sum()
    freq_1 = count_1 / len(df) * 100
    freq_39 = count_39 / len(df) * 100
    print(f"{col}: 1 appears {count_1:4d} times ({freq_1:5.2f}%), 39 appears {count_39:4d} times ({freq_39:5.2f}%)")

# Calculate gaps treating numbers as circular (1-39 wrapping)
def circular_gap(a, b, modulo=39):
    """Calculate circular distance between two numbers"""
    forward = (b - a) % modulo
    backward = (a - b) % modulo
    return min(forward, backward)

print("\n--- Circular Gap Statistics (treating 1-39 as circular) ---")
for i in range(len(columns) - 1):
    col1, col2 = columns[i], columns[i + 1]
    gaps = [circular_gap(row[col1], row[col2], 39) for _, row in df.iterrows()]

    print(f"\n{col1} → {col2} (circular gaps):")
    print(f"  Mean gap: {np.mean(gaps):.2f}")
    print(f"  Median gap: {np.median(gaps):.1f}")
    print(f"  Std gap: {np.std(gaps):.2f}")
    print(f"  Min gap: {np.min(gaps)}")
    print(f"  Max gap: {np.max(gaps)}")

# Boundary transition patterns (38→39, 39→1, 1→2)
print("\n--- Boundary Transition Patterns ---")
# Check for consecutive boundary crossings across all positions
boundary_crossings = []
for idx, row in df.iterrows():
    numbers = row[columns].tolist()
    for i in range(len(numbers) - 1):
        if (numbers[i] >= 38 and numbers[i+1] <= 2):  # Potential wrap-around
            boundary_crossings.append({
                'draw': idx,
                'from': numbers[i],
                'to': numbers[i+1],
                'position': f"{columns[i]}→{columns[i+1]}"
            })

print(f"\nTotal potential boundary wraps: {len(boundary_crossings)}")
if len(boundary_crossings) > 0:
    crossing_df = pd.DataFrame(boundary_crossings)
    print("\nMost common boundary transitions:")
    print(crossing_df.groupby(['from', 'to']).size().sort_values(ascending=False).head(10))

# ============================================================================
# PART 4: REPEAT PATTERNS & TEMPORAL BEHAVIOR
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: REPEAT PATTERNS & TEMPORAL BEHAVIOR")
print("=" * 80)

print("\n--- Consecutive Repeat Analysis (same number appears in consecutive draws) ---")
for col in columns:
    repeats = 0
    for i in range(1, len(df)):
        if df[col].iloc[i] == df[col].iloc[i-1]:
            repeats += 1
    repeat_rate = repeats / (len(df) - 1) * 100
    print(f"{col}: {repeats} consecutive repeats ({repeat_rate:.2f}%)")

    # Expected rate under random assumption (1/39)
    expected_rate = (1/39) * 100
    print(f"  Expected (random): {expected_rate:.2f}%")
    print(f"  Ratio: {repeat_rate / expected_rate:.2f}x")

# ============================================================================
# PART 5: STRATEGIC INSIGHTS FOR TABM
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: STRATEGIC INSIGHTS FOR TABM FEATURE ENGINEERING")
print("=" * 80)

print("\n--- Why Tabular Ensembling vs Time-Series? ---")
print("""
✓ POSITION-SPECIFIC BEHAVIOR: Each QS column (QS_1 through QS_5) exhibits
  distinct statistical properties (different means, ranges, skewness).

✓ STRUCTURED CONSTRAINTS: Values are strictly ascending within each row,
  creating relational dependencies perfect for tabular learning.

✓ CYLINDRICAL SPACE: Numbers 1-39 form a circular space (39↔1 adjacency),
  which tabular models can learn through modular features.

✓ LOW TEMPORAL AUTOCORRELATION: Weak autocorr suggests limited time-series
  predictive power, but strong cross-column dependencies.
""")

print("\n--- Recommended Feature Engineering Strategies ---")
print("""
1. POSITION-AWARE FEATURES:
   - Separate embeddings/models for each QS position
   - Position-specific frequency encodings
   - Relative position within observed range

2. CYLINDRICAL FEATURES:
   - Modular arithmetic (value % 39)
   - Sine/cosine encoding: sin(2π*value/39), cos(2π*value/39)
   - Circular gap features between positions

3. RELATIONAL FEATURES:
   - Gap sizes between consecutive positions (QS_2 - QS_1, etc.)
   - Overlap indicators (is QS_2 within typical QS_1 range?)
   - Total spread (QS_5 - QS_1)

4. HISTORICAL FEATURES:
   - Rolling frequency (last N draws)
   - Time since last appearance per number
   - Recency-weighted occurrence rates

5. ENSEMBLE TARGETS:
   - TabM can learn position-specific sub-models
   - Per-position classification (39-class problem × 5)
   - OR top-k prediction across all 39 numbers
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
