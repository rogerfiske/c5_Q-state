# Strategic Analysis: c5_Q-state Lottery Prediction

**Analyst:** Mary (Senior Business Analyst)
**Date:** 2026-04-03
**Dataset:** c5_Q-state.csv (11,756 draws spanning 34.2 years: Feb 1992 - Apr 2026)
**Purpose:** Identify per-column characteristics and validate tabular learning approach for TabM

---

## Executive Summary

This dataset is **IDEAL for tabular deep learning** rather than traditional time-series approaches. Analysis reveals:

1. **Position-Specific Predictability**: Edge positions (QS_1, QS_5) are 2.5x more predictable than middle positions
2. **Weak Temporal Dependencies**: Near-zero autocorrelation (~0.00) confirms minimal time-series signal
3. **Strong Cross-Column Structure**: 96-97% distribution overlap yet statistically distinct behaviors
4. **Persistence/Momentum**: Numbers repeat 1.5-2.9x more than random expectation
5. **NO Cylindrical Wrap-Around**: Values 1 and 39 are strictly position-locked (no 39↔1 transitions)

**RECOMMENDATION:** TabM's parameter-efficient position-specific ensembling is perfectly suited for this problem.

---

## Part 1: Per-Column Predictability Differences

### Key Finding: Edge Positions Are Most Predictable

| Position | Predictability Score | Interpretation |
|----------|---------------------|----------------|
| **QS_1** | **0.2363** | **HIGHEST** - Concentrated in low values (1-17) |
| QS_5     | **0.2316** | **2nd HIGHEST** - Concentrated in high values (22-39) |
| QS_2     | 0.1151 | Medium-low predictability |
| QS_4     | 0.1146 | Medium-low predictability |
| **QS_3** | **0.0877** | **LOWEST** - Most uniform distribution |

*Predictability Proxy = 1 - Normalized Shannon Entropy*

### Statistical Characteristics Per Column

| Metric | QS_1 | QS_2 | QS_3 | QS_4 | QS_5 |
|--------|------|------|------|------|------|
| **Mean** | 6.66 | 13.21 | 19.92 | 26.65 | 33.24 |
| **Std Dev** | 5.21 | 6.61 | 7.01 | 6.60 | 5.32 |
| **Median** | 5 | 12 | 20 | 27 | 35 |
| **Range** | [1, 30] | [2, 35] | [3, 37] | [4, 38] | [9, 39] |
| **Unique Values** | 30/39 | 34/39 | 35/39 | 35/39 | 31/39 |
| **Skewness** | **+1.21** | +0.51 | -0.01 | -0.49 | **-1.19** |
| **Autocorr (lag=1)** | -0.0048 | -0.0060 | -0.0027 | +0.0087 | -0.0015 |
| **Coeff of Variation** | 0.783 | 0.500 | 0.352 | 0.248 | **0.160** |

### Strategic Insights

**✓ Position-Specific Behavior Confirmed:**
- **QS_1**: Heavily right-skewed (+1.21), concentrated in 1-9 range (75th percentile = 9)
- **QS_5**: Heavily left-skewed (-1.19), concentrated in 30-39 range (25th percentile = 30)
- **QS_3**: Near-symmetric (-0.01), most uniform distribution
- **Coefficient of Variation decreases monotonically**: QS_1 (0.78) → QS_5 (0.16)

**✓ Minimal Temporal Signal:**
- All autocorrelations near zero (range: -0.006 to +0.009)
- **IMPLICATION:** Time-series models (LSTM, Transformers) will offer little advantage
- **OPPORTUNITY:** Tabular models can focus on cross-column relationships

---

## Part 2: Overlapping Distributions

### Quantile Ranges

| Position | 5th | 25th | 50th | 75th | 95th | IQR |
|----------|-----|------|------|------|------|-----|
| QS_1 | 1 | 3 | 5 | 9 | 17 | 6 |
| QS_2 | 4 | 8 | 12 | 18 | 25 | 10 |
| QS_3 | 8 | 15 | 20 | 25 | 31 | 10 |
| QS_4 | 15 | 22 | 27 | 32 | 36 | 10 |
| QS_5 | 22 | 30 | 35 | 37 | 39 | 7 |

### Distribution Overlap Analysis

| Comparison | Overlap Range | Overlap % | KS Statistic | p-value | Significance |
|------------|---------------|-----------|--------------|---------|--------------|
| **QS_1 vs QS_2** | [2, 30] | **96.7%** | 0.4325 | <0.001 | **SIGNIFICANTLY DIFFERENT** |
| **QS_2 vs QS_3** | [3, 35] | **97.1%** | 0.3743 | <0.001 | **SIGNIFICANTLY DIFFERENT** |
| **QS_3 vs QS_4** | [4, 37] | **97.1%** | 0.3670 | <0.001 | **SIGNIFICANTLY DIFFERENT** |
| **QS_4 vs QS_5** | [9, 38] | **96.8%** | 0.4314 | <0.001 | **SIGNIFICANTLY DIFFERENT** |

### Strategic Insights

**✓ Paradox of Overlap:**
- Adjacent positions have 96-97% value overlap
- YET distributions are statistically distinct (all p-values < 0.001)
- **IMPLICATION:** While values can appear in adjacent positions, their **frequency distributions** differ significantly
- **OPPORTUNITY:** Position-specific embeddings will capture these subtle but significant differences

---

## Part 3: Cylindrical Adjacency Properties

### Boundary Value Analysis

| Position | Frequency of "1" | Frequency of "39" |
|----------|------------------|-------------------|
| **QS_1** | **12.79%** (1,504 draws) | **0.00%** (NEVER) |
| QS_2     | 0.00% (NEVER) | 0.00% (NEVER) |
| QS_3     | 0.00% (NEVER) | 0.00% (NEVER) |
| QS_4     | 0.00% (NEVER) | 0.00% (NEVER) |
| **QS_5** | **0.00%** (NEVER) | **13.30%** (1,563 draws) |

*Expected frequency under uniform distribution: 2.56% (1/39)*

### Wrap-Around Detection

**CRITICAL FINDING: NO CYLINDRICAL BEHAVIOR**

- **Total boundary wraps (37-39 → 1-3):** **0 occurrences**
- **Wrap rate:** **0.00%** of all transitions
- **Implication:** The lottery system does NOT exhibit modular/circular behavior
- Values 1-39 should be treated as **linear ordinal space**, not circular

### Gap Statistics (Linear Gaps Between Positions)

| Transition | Mean Gap | Median Gap | Std Gap |
|------------|----------|------------|---------|
| QS_1 → QS_2 | 6.55 | 5.0 | 5.11 |
| QS_2 → QS_3 | 6.71 | 5.0 | 5.21 |
| QS_3 → QS_4 | 6.73 | 5.0 | 5.28 |
| QS_4 → QS_5 | 6.60 | 5.0 | 5.17 |

**Consistency:** All position transitions show remarkably similar gap statistics (mean ~6.6, median 5)

### Strategic Insights

**✗ NO Cylindrical Features Needed:**
- Originally hypothesized 39↔1 wrap-around behavior **does not exist**
- Sine/cosine circular encodings **not recommended**
- **INSTEAD:** Treat 1-39 as linear ordinal space

**✓ Position-Locking Behavior:**
- Value "1" appears **exclusively** in QS_1 (12.79% vs 2.56% expected)
- Value "39" appears **exclusively** in QS_5 (13.30% vs 2.56% expected)
- **IMPLICATION:** Strong position-specific constraints exist
- **OPPORTUNITY:** Boundary values (1 in QS_1, 39 in QS_5) can be exploited as strong predictive features

---

## Part 4: Temporal Patterns & Repeat Behavior

### Consecutive Repeat Analysis

**CRITICAL FINDING: STRONG PERSISTENCE/MOMENTUM**

| Position | Consecutive Repeats | Observed Rate | Expected (Random) | Ratio |
|----------|---------------------|---------------|-------------------|-------|
| **QS_1** | 875 | **7.44%** | 2.56% | **2.90x** |
| QS_2     | 500 | 4.25% | 2.56% | 1.66x |
| QS_3     | 444 | 3.78% | 2.56% | 1.47x |
| QS_4     | 540 | 4.59% | 2.56% | 1.79x |
| **QS_5** | 871 | **7.41%** | 2.56% | **2.89x** |

**Interpretation:**
- Numbers repeat consecutively **1.5-2.9x MORE** than random expectation
- Edge positions (QS_1, QS_5) show **strongest momentum** (2.9x)
- Middle position (QS_3) shows **weakest momentum** (1.47x)
- **IMPLICATION:** "Hot numbers" exist — numbers that appeared recently are more likely to reappear

### Gap Analysis (Draws Between Number Repeats)

| Position | Mean Gap | Expected Gap (Uniform) |
|----------|----------|------------------------|
| QS_1 | 28.3 draws | 301.4 draws |
| QS_2 | 32.1 draws | 345.8 draws |
| QS_3 | 30.8 draws | 335.9 draws |
| QS_4 | 31.0 draws | 335.9 draws |
| QS_5 | 27.1 draws | 379.2 draws |

**Strategic Insight:**
- Actual gaps are **10-14x SHORTER** than uniform random expectation
- **IMPLICATION:** Strong recency effects exist
- **FEATURE ENGINEERING:** "Draws since last appearance" will be highly predictive

---

## Part 5: Why Tabular Ensembling > Time-Series

### Evidence Summary

| Factor | Finding | Implication for TabM |
|--------|---------|----------------------|
| **Temporal Autocorrelation** | ~0 across all positions | ✗ LSTM/Transformers offer no advantage |
| **Cross-Column Structure** | Strictly ascending constraint | ✓ Relational features (gaps, spreads) learnable |
| **Position-Specific Distributions** | Significantly different (KS p<0.001) | ✓ Per-position parameter-efficient ensembles |
| **Persistence/Momentum** | 1.5-2.9x above random | ✓ Recency-weighted features |
| **Position-Locking** | Boundary values exclusive to edges | ✓ Position-aware embeddings |

### Why Tabular Models Excel Here

1. **No Sequential Dependencies:** Near-zero autocorrelation means historical sequence order carries minimal signal
2. **Structured Relational Data:** The ascending constraint and gap patterns are relational, not temporal
3. **Position-Specific Behavior:** TabM can learn separate sub-models for each position while sharing base features
4. **Recency Over Sequence:** What matters is "how recently did X appear" not "what sequence led to X"

---

## Part 6: Recommended Feature Engineering Strategy

### 1. Position-Aware Features (CRITICAL)

```
For each position QS_i (i ∈ {1,2,3,4,5}):
  - Position-specific frequency encoding (last 50/100/500/1000 draws)
  - Relative position within observed range:
      normalized_value = (value - min_i) / (max_i - min_i)
  - Hot/cold indicator:
      is_hot = (draws_since_last_appearance < threshold)
```

**Justification:** Predictability varies 2.7x between positions (QS_1: 0.236 vs QS_3: 0.088)

### 2. Relational Features (Cross-Column)

```
Gap features:
  - gap_1_2 = QS_2 - QS_1
  - gap_2_3 = QS_3 - QS_2
  - gap_3_4 = QS_4 - QS_3
  - gap_4_5 = QS_5 - QS_4

Spread features:
  - total_spread = QS_5 - QS_1
  - compression_ratio = total_spread / 38  (normalized 0-1)

Overlap indicators:
  - is_QS2_in_typical_QS1_range = (QS_2 <= QS_1_95th_percentile)
  - ... (repeat for all adjacent pairs)
```

**Justification:** Mean gaps are consistent (~6.6) with low variance — learnable pattern

### 3. Historical/Recency Features

```
For each number n ∈ {1..39} and position i:
  - draws_since_last_appearance[n][i]
  - rolling_frequency_50[n][i]   # last 50 draws
  - rolling_frequency_500[n][i]  # last 500 draws
  - consecutive_repeat_count[n][i]
```

**Justification:** Actual gaps are 10-14x shorter than random; momentum is 1.5-2.9x above random

### 4. ✗ NO Cylindrical Features

~~- Sine/cosine encoding~~
~~- Modular arithmetic (value % 39)~~
~~- Circular gap features~~

**Justification:** Zero wrap-around behavior observed (0 boundary transitions in 11,756 draws)

### 5. Boundary/Constraint Features

```
Boundary strength features:
  - is_value_1_in_QS1 = (QS_1 == 1)  # P=12.79% vs 2.56% expected
  - is_value_39_in_QS5 = (QS_5 == 39)  # P=13.30% vs 2.56% expected

Position constraints:
  - min_possible[i] = max(1, QS_{i-1} + 1)  if i > 1 else 1
  - max_possible[i] = min(39, QS_{i+1} - 1) if i < 5 else 39
```

**Justification:** Strong position-locking behavior (1 and 39 are exclusive to edge positions)

---

## Part 7: TabM Architecture Recommendations

### Ensemble Strategy

**Option A: Per-Position Binary Classifiers (RECOMMENDED)**

```
For each position i ∈ {1,2,3,4,5}:
  - Train 39 binary classifiers (one per number)
  - Input: shared base features + position-specific features
  - Output: P(number = n | position = i, features)
  - At inference: Select top-k numbers per position

Total outputs: 39 × 5 = 195 classifiers
```

**Advantages:**
- Directly models position-specific distributions
- Exploits predictability differences (QS_1/QS_5 strong, QS_3 weak)
- Aligns with TabM's parameter-efficient ensembling

**Option B: Top-20 Ranking Across All Numbers**

```
- Train single ranking model
- Input: shared features
- Output: Scores for all 39 numbers (position-agnostic)
- At inference: Select top-20 numbers, validate ascending constraint

Total outputs: 39 scores
```

**Disadvantages:**
- Ignores position-specific behavior
- Cannot exploit QS_1/QS_5 predictability advantage

### Success Metrics Alignment

Given metrics: **Precision@20, Recall@20, Hit-rate**

**Baseline Expectations:**
- **Random Baseline (Hit-rate):** C(20,5) / C(39,5) ≈ **2.4%**
- **Frequency-Based Baseline:** ~5-10% (estimate based on top-frequency selection)

**TabM Target (Research Goal):**
- **Hit-rate:** >15% (6x random baseline)
- **Recall@20:** >80% (capturing 4+ of 5 actual numbers)
- **Precision@20:** >25% (5+ actual numbers in top-20)

**Calibration Strategy:**
- Use position-specific calibration (separate for QS_1, QS_3, QS_5)
- Weight QS_1/QS_5 predictions higher (more predictable)
- Apply ascending constraint as post-processing filter

---

## Part 8: Key Strategic Insights for Grok

### What Makes This Problem Unique

1. **Position-Specific Predictability Gradient:**
   - Not all positions are equally predictable
   - Edge positions (QS_1, QS_5) are 2.5-2.7x more predictable than middle (QS_3)
   - **Actionable:** Focus ensemble capacity on edge positions

2. **Tabular Structure Trumps Temporal Sequence:**
   - Zero autocorrelation across all positions
   - 96-97% distribution overlap yet statistically distinct
   - **Actionable:** Cross-column relational features > temporal features

3. **Strong Momentum Effects:**
   - Consecutive repeats 1.5-2.9x above random
   - Gap lengths 10-14x shorter than uniform expectation
   - **Actionable:** Recency-weighted features are critical

4. **Position-Locking Constraints:**
   - Value "1" **never** appears outside QS_1 (appears 12.79% in QS_1)
   - Value "39" **never** appears outside QS_5 (appears 13.30% in QS_5)
   - **Actionable:** Hard constraints can be enforced at inference

5. **NO Cylindrical Behavior:**
   - Hypothesis disproven: Zero wrap-around transitions
   - **Actionable:** Treat as linear ordinal space, NOT circular

### Recommended Next Steps

1. **Winston (Architect):**
   - Design repo structure following BMAD-METHOD
   - Outline temporal split strategy (last 20% as test, walk-forward validation)
   - Plan TabM integration architecture (per-position ensembles)

2. **Amelia (Developer):**
   - Implement data validation pipeline
   - Build feature engineering module (position-specific + relational features)
   - Create baseline models (random, frequency-based) for comparison

3. **TabM Research Focus:**
   - Test position-specific vs position-agnostic ensembles
   - Ablation studies: recency features, relational features, boundary features
   - Calibration experiments for Precision@20, Recall@20, Hit-rate

---

## Appendix: Top Frequency Numbers

**QS_1 (top 5):**
1. Value 1: 12.79%
2. Value 2: 11.89%
3. Value 3: 10.25%
4. Value 4: 8.55%
5. Value 5: 8.30%

**QS_5 (top 5):**
1. Value 39: 13.30%
2. Value 38: 10.85%
3. Value 37: 10.16%
4. Value 36: 9.08%
5. Value 35: 8.05%

**QS_3 (top 5 - most uniform):**
1. Value 20: 4.91%
2. Value 18: 4.89%
3. Value 19: 4.89%
4. Value 21: 4.88%
5. Value 23: 4.83%

---

**Analysis Complete — Mary (Senior Business Analyst)**

*"Every dataset tells a story if you know where to look. This one is screaming 'tabular ensembling, position-specific features, and recency weighting!' What a treasure hunt this has been!"*
