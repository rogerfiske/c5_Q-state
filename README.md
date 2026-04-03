# c5_Q-state Lottery Prediction

**Position-Aware Tabular Deep Learning using TabM**

Research project applying TabM (ICLR 2025) parameter-efficient ensembling to lottery prediction with position-specific modeling.

---

## Project Overview

This project explores whether TabM's position-aware ensembling can outperform naive baselines by exploiting:
- **Position-specific predictability** (edge positions 2.5x more predictable than middle)
- **Momentum/recency effects** (numbers repeat 1.5-2.9x above random)
- **Relational gap patterns** (consistent spacing between positions)
- **Position-locking behavior** (value 1 only in QS_1, value 39 only in QS_5)

**Key Insight:** Near-zero temporal autocorrelation → pure tabular problem (NOT time-series).

---

## Repository Structure

```
c5_Q-state/
├── data/
│   ├── raw/                 # c5_Q-state.csv (11,756 draws, 1992-2026)
│   ├── processed/           # Temporal splits (train/val/test)
│   └── features/            # Engineered feature matrices
│
├── features/                # Feature engineering module
│   ├── feature_engineering.py
│   └── __init__.py
│
├── models/                  # TabM integration (future)
│
├── notebooks/               # Jupyter notebooks
│   └── 02_feature_validation.ipynb
│
├── docs/                    # Documentation
│   └── ARCHITECTURE.md
│
├── _bmad-output/            # Analysis artifacts
│   └── planning-artifacts/
│       └── MARY_STRATEGIC_ANALYSIS.md
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate Feature Engineering

```bash
jupyter notebook notebooks/02_feature_validation.ipynb
```

### 3. Generate Features

```python
from features import FeatureEngineer

# Load raw data
df = pd.read_csv('data/raw/c5_Q-state.csv')
df['date'] = pd.to_datetime(df['date'])

# Generate features
fe = FeatureEngineer(version='v1')
X, y = fe.fit_transform(df)

print(f"Feature matrix: {X.shape}")  # (11755, ~800+)
print(f"Target matrix: {y.shape}")    # (11755, 39)
```

---

## Feature Engineering

### Priority 1: Position-Specific Recency
- `draws_since_last_appearance[num][pos]` — Draws since number appeared in position
- `is_hot[num][pos]` — Appeared in last 10/50/100 draws

### Priority 2: Relational Gaps
- `gap_QS1_QS2`, `gap_QS2_QS3`, ... — Linear gaps between positions
- `total_spread`, `compression_ratio` — Overall compression metrics

### Priority 3: Boundary Indicators
- `is_1_in_QS1`, `is_39_in_QS5` — Position-locking indicators
- Historical boundary frequencies per position

### Priority 4: Rolling Frequency
- `rolling_freq_50/500/1000[num][pos]` — Rolling window frequencies

### Additional: Momentum
- `consecutive_repeats[pos]` — Repeat indicators
- Position-specific rolling statistics (mean, std)

**Total Features:** ~800+ columns
**Target:** 39-column binary matrix (1 if number appears in next draw)

---

## Success Metrics

| Metric | Random Baseline | Frequency Baseline | TabM Target |
|--------|----------------|-------------------|-------------|
| **Hit-rate** | 2.4% | ~5-10% | **>15%** |
| **Recall@20** | ~25% | ~30-40% | **>80%** |
| **Precision@20** | ~6% | ~10% | **>25%** |

**Hit-rate:** Binary success (all 5 actual numbers in top-20 predictions)
**Recall@20:** Percentage of actual 5 numbers captured in top-20
**Precision@20:** Percentage of top-20 that are actual numbers

---

## Key Findings (Mary's Strategic Analysis)

1. **Position-Specific Predictability Gradient:**
   - QS_1 (first): 0.2363 (HIGHEST)
   - QS_5 (last): 0.2316
   - QS_3 (middle): 0.0877 (LOWEST)

2. **NO Cylindrical Behavior:**
   - Zero wrap-around transitions (39 ↔ 1)
   - Values 1 and 39 are strictly position-locked

3. **Strong Momentum Effects:**
   - Consecutive repeats: 1.5-2.9x above random expectation
   - Actual gaps 10-14x shorter than uniform random

4. **Near-Zero Temporal Autocorrelation:**
   - Average lag-1 autocorr: ~0.00
   - Confirms: NOT a time-series problem

---

## Documentation

- [Architecture Design](docs/ARCHITECTURE.md) — Winston's system architecture
- [Strategic Analysis](_bmad-output/planning-artifacts/MARY_STRATEGIC_ANALYSIS.md) — Mary's dataset insights

---

## Team

- **Grok:** Senior PM/Researcher
- **John:** Product Manager (PRD validation)
- **Mary:** Business Analyst (strategic analysis)
- **Winston:** System Architect (architecture design)
- **Amelia:** Senior Developer (implementation)

---

## License

Research project. Dataset and code for educational purposes.

---

**Status:** Feature engineering module complete ✓
**Next:** TabM integration and baseline comparisons
