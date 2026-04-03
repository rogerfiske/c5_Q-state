# c5_Q-state Lottery Prediction: System Architecture

**Architect:** Winston (Senior System Architect)
**Date:** 2026-04-03
**Project:** TabM-based lottery prediction using position-aware tabular ensembling
**Status:** Initial Design v1.0

---

## Executive Summary

This architecture balances research flexibility with production-ready patterns. Key decisions:

1. **Modular feature engineering pipeline** — Pluggable, versioned, reproducible
2. **Position-aware modeling** — Exploit 2.5x predictability difference between edge/middle positions
3. **Temporal validation** — Prevent leakage, validate generalization
4. **Boring technology** — pandas, scikit-learn, PyTorch (TabM) — proven, stable, debuggable
5. **Research-first, production-ready** — Easy experimentation, clear path to deployment

**Core Insight from Mary's Analysis:**
Near-zero temporal autocorrelation + strong position-specific behavior = **pure tabular problem**. TabM's parameter-efficient ensembling is perfectly suited.

---

## 1. Repository Structure

```
c5_Q-state/
├── data/
│   ├── raw/                      # Immutable source data
│   │   └── c5_Q-state.csv
│   ├── processed/                # Cleaned, validated data
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── features/                 # Engineered feature matrices
│       ├── train_features_v1.parquet
│       ├── val_features_v1.parquet
│       └── test_features_v1.parquet
│
├── features/                     # Feature engineering module
│   ├── __init__.py
│   ├── feature_engineering.py   # Main FeatureEngineer class
│   ├── position_features.py     # Position-specific recency features
│   ├── relational_features.py   # Gap, spread, overlap features
│   ├── boundary_features.py     # Position-locking indicators
│   └── rolling_features.py      # Historical frequency features
│
├── models/                       # Model training and inference
│   ├── __init__.py
│   ├── baselines.py             # Random, frequency-based baselines
│   ├── tabm_wrapper.py          # TabM integration wrapper
│   ├── position_aware.py        # Position-specific ensemble architecture
│   ├── training.py              # Training loop, checkpointing
│   ├── inference.py             # Prediction with constraint enforcement
│   └── evaluation.py            # Metrics: Precision@20, Recall@20, Hit-rate
│
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_validation.ipynb  # Validate feature engineering
│   ├── 03_baseline_experiments.ipynb  # Random, frequency baselines
│   ├── 04_tabm_position_agnostic.ipynb  # TabM without position-awareness
│   ├── 05_tabm_position_aware.ipynb     # TabM with position-specific heads
│   └── 06_ablation_studies.ipynb        # Feature importance ablations
│
├── tests/                        # Unit and integration tests
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   └── test_inference.py
│
├── scripts/                      # CLI tools and automation
│   ├── prepare_data.py          # Data splitting (temporal)
│   ├── train.py                 # Model training CLI
│   ├── predict.py               # Batch prediction CLI
│   └── evaluate.py              # Metric calculation CLI
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md          # This file
│   ├── API.md                   # Feature engineering API spec
│   ├── EXPERIMENTS.md           # Experiment tracking log
│   └── DEPLOYMENT.md            # Deployment guide (future)
│
├── _bmad-output/                 # BMAD-METHOD artifacts
│   ├── planning-artifacts/
│   │   └── MARY_STRATEGIC_ANALYSIS.md
│   └── implementation-artifacts/
│
├── analysis/                     # One-off analysis scripts
│   └── mary_strategic_analysis_text.py
│
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment (optional)
├── README.md                     # Project overview
└── .gitignore                    # Git exclusions
```

### Design Principles

1. **Separation of Concerns:**
   - `features/` = feature engineering logic (pure functions, no model knowledge)
   - `models/` = model architecture, training, inference
   - `notebooks/` = exploration, validation, experimentation
   - `scripts/` = automation, reproducibility

2. **Data Versioning:**
   - Raw data is immutable
   - Processed data is versioned (`train_v1.parquet`, `train_v2.parquet`)
   - Features are versioned (`train_features_v1.parquet`)
   - Models reference feature version explicitly

3. **Reproducibility:**
   - All randomness is seeded
   - Feature engineering is deterministic
   - Experiments are logged (experiment ID, hyperparams, metrics)

---

## 2. Data Pipeline Architecture

### 2.1 Temporal Data Splitting

**Critical Constraint:** Prevent temporal leakage (no future information in training).

```
Dataset: 11,756 draws (Feb 1992 - Apr 2026)

Split Strategy (Temporal):
├── Train:  70% = 8,229 draws (Feb 1992 - Sep 2015)
├── Val:    10% = 1,176 draws (Sep 2015 - Jul 2018)
└── Test:   20% = 2,351 draws (Jul 2018 - Apr 2026)

Rationale:
- Train on historical data (23+ years)
- Validate on intermediate period (3 years)
- Test on recent period (8 years) to measure generalization
```

**Implementation:** `scripts/prepare_data.py`

```python
def temporal_split(df, train_pct=0.7, val_pct=0.1, test_pct=0.2):
    """
    Split data temporally (no shuffling).

    Args:
        df: DataFrame sorted by date
        train_pct, val_pct, test_pct: Split percentages

    Returns:
        train_df, val_df, test_df
    """
    assert df['date'].is_monotonic_increasing, "Data must be sorted by date"

    n = len(df)
    train_idx = int(n * train_pct)
    val_idx = train_idx + int(n * val_pct)

    train = df.iloc[:train_idx]
    val = df.iloc[train_idx:val_idx]
    test = df.iloc[val_idx:]

    return train, val, test
```

### 2.2 Walk-Forward Validation (Future Enhancement)

**Purpose:** Simulate realistic deployment (predicting next draw based on historical data).

```
Walk-Forward Strategy:
For each test point t:
    1. Train on all data before t
    2. Predict draw at t
    3. Evaluate against actual draw at t
    4. Slide window forward

Advantages:
- Most realistic evaluation
- Detects model degradation over time
- Validates feature stability

Trade-offs:
- Computationally expensive (requires N model trainings)
- Implement AFTER validating base TabM approach
```

**Status:** Phase 2 (post-baseline validation)

### 2.3 Feature Engineering Pipeline

**Design Goal:** Modular, composable, version-controlled feature generation.

```python
# High-level API
from features import FeatureEngineer

fe = FeatureEngineer(version='v1')
X_train, y_train = fe.fit_transform(train_df)
X_val, y_val = fe.transform(val_df)
X_test, y_test = fe.transform(test_df)

# Save for reproducibility
fe.save('data/features/feature_engineer_v1.pkl')
X_train.to_parquet('data/features/train_features_v1.parquet')
```

**Feature Priority Tiers (from Mary's Analysis):**

1. **Priority 1: Position-Specific Recency Features**
   - `draws_since_last_appearance[num][pos]` — How many draws since number `num` appeared in position `pos`
   - `is_hot[num][pos]` — Binary: appeared in last N draws (N=10, 50, 100)
   - Rationale: Momentum effects are 1.5-2.9x above random

2. **Priority 2: Relational Gap Features**
   - `gap_QS1_QS2`, `gap_QS2_QS3`, ... — Linear gaps between adjacent positions
   - `total_spread = QS_5 - QS_1` — Overall compression
   - `compression_ratio = total_spread / 38` — Normalized compression
   - Rationale: Consistent gap patterns (mean ~6.6, median 5.0)

3. **Priority 3: Boundary Indicators**
   - `is_1_in_QS1`, `is_39_in_QS5` — Position-locking indicators
   - `boundary_strength[pos]` — Frequency of boundary values (1, 39) per position
   - Rationale: Value 1 appears 12.79% in QS_1 (vs 2.56% expected), never elsewhere

4. **Priority 4: Rolling Frequency Features**
   - `rolling_freq_50[num][pos]` — Frequency of `num` in position `pos` over last 50 draws
   - `rolling_freq_500[num][pos]` — Last 500 draws
   - `rolling_freq_1000[num][pos]` — Last 1000 draws
   - Rationale: Capture medium/long-term trends

**Feature Output Shape:**
- Input: Raw DataFrame (11,756 × 6 columns: date, QS_1, ..., QS_5)
- Output X: Feature matrix (11,755 × ~500 columns) — one row per draw, predicting NEXT draw
- Output y: Target matrix (11,755 × 39 columns) — binary presence of each number 1-39 in next draw

**Note:** We lose 1 row (last draw has no "next" to predict).

---

## 3. TabM Integration Architecture

### 3.1 Model Design Options

**Option A: Position-Agnostic (Baseline)**

```
Input: Shared feature vector X (all engineered features)
Architecture:
  - TabM base model
  - Single output head: 39-dimensional softmax (or sigmoid for multi-label)
  - Predicts: P(number n appears in next draw) for n ∈ {1..39}

Output: 39 probabilities
Inference: Select top-20 numbers, enforce ascending constraint

Advantages:
- Simple baseline
- Fast to train
- Easy to debug

Disadvantages:
- Ignores position-specific predictability (QS_1: 0.236 vs QS_3: 0.088)
- Treats all positions equally (suboptimal)
```

**Option B: Position-Aware (Recommended)**

```
Input: Shared feature vector X
Architecture:
  - TabM base model (shared trunk)
  - 5 position-specific output heads: one per QS position
  - Each head: 39-dimensional output (P(number n appears in position i))

Output: 5 × 39 = 195 probabilities
Inference:
  1. For each position, rank numbers by probability
  2. Greedily select top numbers respecting ascending constraint
  3. Backtrack if needed to enforce constraint

Advantages:
- Exploits position-specific predictability
- Aligns with Mary's findings (edge positions more predictable)
- Can weight QS_1/QS_5 predictions higher

Disadvantages:
- More complex inference (constraint enforcement)
- Slightly slower training (5 heads vs 1)

Trade-off: Complexity is worth it given 2.5x predictability difference.
```

**Decision:** Implement both. Start with Option A (baseline), then Option B (research contribution).

### 3.2 TabM Configuration

```python
# Pseudo-code (actual TabM API may differ)
from tabm import TabMClassifier

# Position-agnostic baseline
model_agnostic = TabMClassifier(
    n_estimators=100,          # Number of ensemble members
    max_depth=6,               # Tree depth per member
    learning_rate=0.01,
    objective='binary',        # Multi-label classification
    n_jobs=-1
)

# Position-aware (5 separate models, one per position)
models_position_aware = {
    f'QS_{i}': TabMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.01,
        objective='binary',
        n_jobs=-1
    )
    for i in range(1, 6)
}
```

### 3.3 Loss Function and Calibration

**Primary Metric:** Hit-rate (binary: all 5 actual numbers in top-20 predictions)

**Secondary Metrics:**
- Precision@20 = (# actual numbers in top-20) / 20
- Recall@20 = (# actual numbers in top-20) / 5

**Loss Function:**
- Binary cross-entropy (per number, per position)
- Optional: Weighted BCE to emphasize edge positions (QS_1, QS_5)

**Calibration Strategy:**
- Post-training: Apply Platt scaling or isotonic regression
- Goal: Ensure predicted probabilities → actual frequencies
- Metric: Calibration error (ECE - Expected Calibration Error)

---

## 4. Training and Inference Pipeline

### 4.1 Training Loop

```python
# scripts/train.py (pseudo-code)

def train_tabm_position_aware(X_train, y_train, X_val, y_val):
    """
    Train position-aware TabM ensemble.

    Args:
        X_train: (N, F) feature matrix
        y_train: (N, 39) binary target matrix (next draw)
        X_val, y_val: Validation set

    Returns:
        models: Dict of 5 TabM models (one per position)
        metrics: Validation metrics
    """
    models = {}

    # Train separate model for each position
    for pos in range(1, 6):
        print(f"Training QS_{pos}...")

        # Extract position-specific targets
        # (Which numbers appear in QS_pos in next draw?)
        y_train_pos = extract_position_targets(y_train, pos)
        y_val_pos = extract_position_targets(y_val, pos)

        # Train TabM
        model = TabMClassifier(n_estimators=100, max_depth=6)
        model.fit(
            X_train, y_train_pos,
            eval_set=[(X_val, y_val_pos)],
            early_stopping_rounds=10,
            verbose=True
        )

        models[f'QS_{pos}'] = model

    # Evaluate on validation set
    metrics = evaluate_position_aware(models, X_val, y_val)

    return models, metrics
```

### 4.2 Inference with Constraint Enforcement

**Challenge:** Ensure predicted 5 numbers are strictly ascending.

**Greedy Algorithm:**

```python
def predict_with_constraint(models, X):
    """
    Predict next draw respecting ascending constraint.

    Args:
        models: Dict of position-specific TabM models
        X: (1, F) feature vector for next prediction

    Returns:
        predicted_numbers: List of 5 numbers (strictly ascending)
        probabilities: Associated probabilities
    """
    # Get probabilities for each position
    probs = {}
    for pos in range(1, 6):
        probs[f'QS_{pos}'] = models[f'QS_{pos}'].predict_proba(X)[0]  # (39,)

    # Greedy selection with backtracking
    selected = []
    for pos in range(1, 6):
        # Filter numbers: must be > last selected
        min_value = selected[-1] + 1 if selected else 1
        max_value = 39

        # Rank eligible numbers by probability
        eligible = [(num, probs[f'QS_{pos}'][num-1])
                    for num in range(min_value, max_value + 1)]
        eligible.sort(key=lambda x: x[1], reverse=True)

        # Select top number
        if eligible:
            selected.append(eligible[0][0])
        else:
            # Fallback: backtrack or sample uniformly
            selected.append(min_value)

    return selected
```

**Trade-off:** Greedy algorithm is fast but suboptimal. Future: beam search or dynamic programming.

### 4.3 Top-20 Prediction (for Precision@20, Recall@20, Hit-rate)

**Strategy:** Aggregate position-specific probabilities into top-20 ranking.

```python
def predict_top20(models, X):
    """
    Predict top-20 most likely numbers (position-agnostic).

    Args:
        models: Dict of position-specific TabM models
        X: (1, F) feature vector

    Returns:
        top_20: List of 20 numbers (sorted by aggregated probability)
    """
    # Aggregate probabilities across positions
    aggregated = np.zeros(39)
    for pos in range(1, 6):
        probs = models[f'QS_{pos}'].predict_proba(X)[0]  # (39,)
        aggregated += probs  # Simple sum (or weighted by position predictability)

    # Normalize
    aggregated /= 5.0

    # Select top-20
    top_20_indices = np.argsort(aggregated)[-20:][::-1]
    top_20_numbers = [idx + 1 for idx in top_20_indices]  # Convert 0-indexed to 1-indexed

    return top_20_numbers
```

**Enhancement:** Weight by position predictability:

```python
# Weight QS_1 and QS_5 higher (more predictable)
weights = [0.2363, 0.1151, 0.0877, 0.1146, 0.2316]  # From Mary's analysis
weights = np.array(weights) / np.sum(weights)  # Normalize to sum=1

for pos, weight in enumerate(weights, start=1):
    probs = models[f'QS_{pos}'].predict_proba(X)[0]
    aggregated += weight * probs
```

---

## 5. Evaluation Strategy

### 5.1 Baseline Comparisons

**Random Baseline:**
```python
def random_baseline():
    """Randomly select 20 numbers from 1-39."""
    return random.sample(range(1, 40), 20)

# Expected Hit-rate: C(20,5) / C(39,5) ≈ 2.4%
```

**Frequency-Based Baseline:**
```python
def frequency_baseline(train_df):
    """Select top-20 most frequent numbers from training set."""
    # Flatten all QS columns
    all_numbers = train_df[['QS_1', 'QS_2', 'QS_3', 'QS_4', 'QS_5']].values.flatten()

    # Count frequencies
    freq = Counter(all_numbers)

    # Select top-20
    top_20 = [num for num, _ in freq.most_common(20)]

    return top_20

# Expected Hit-rate: ~5-10% (Mary's estimate)
```

**Position-Specific Frequency Baseline:**
```python
def position_frequency_baseline(train_df):
    """Select most frequent number per position."""
    predicted = []
    for pos in ['QS_1', 'QS_2', 'QS_3', 'QS_4', 'QS_5']:
        most_frequent = train_df[pos].value_counts().index[0]
        predicted.append(most_frequent)

    return predicted

# Sanity check: May violate ascending constraint → need post-processing
```

### 5.2 Success Criteria (from John's PRD)

| Metric | Random Baseline | Frequency Baseline | TabM Target |
|--------|----------------|--------------------| ------------|
| **Hit-rate** | 2.4% | ~5-10% | **>15%** (6x random) |
| **Recall@20** | ~25% | ~30-40% | **>80%** (4+ of 5 captured) |
| **Precision@20** | ~6% | ~10% | **>25%** (5+ in top-20) |

**Research Goal:** Demonstrate TabM's position-aware ensembling beats naive baselines by exploiting:
1. Position-specific predictability
2. Momentum/recency effects
3. Relational gap patterns

---

## 6. Deployment Considerations

### 6.1 Batch Prediction Workflow

**Use Case:** Predict next lottery draw (weekly).

```bash
# Weekly prediction pipeline
$ python scripts/predict.py \
    --model models/tabm_position_aware_v1.pkl \
    --features data/features/feature_engineer_v1.pkl \
    --input data/raw/c5_Q-state.csv \
    --output predictions/2026-04-10.json

# Output format:
{
  "date": "2026-04-10",
  "predicted_5_numbers": [3, 12, 19, 28, 36],
  "top_20_numbers": [1, 2, 3, 5, 7, ..., 39],
  "model_version": "tabm_position_aware_v1",
  "feature_version": "v1",
  "confidence_scores": {
    "QS_1": 0.85,
    "QS_2": 0.72,
    ...
  }
}
```

### 6.2 Feature Versioning

**Problem:** Features evolve over time. How to ensure reproducibility?

**Solution:**
1. **Pickle FeatureEngineer:** Save entire feature engineering state
2. **Version features:** `train_features_v1.parquet`, `train_features_v2.parquet`
3. **Model-feature coupling:** Model metadata stores feature version

```python
# Model metadata
metadata = {
    'model_type': 'tabm_position_aware',
    'feature_version': 'v1',
    'trained_on': '2026-04-03',
    'train_size': 8229,
    'val_metrics': {
        'hit_rate': 0.18,
        'recall_20': 0.85,
        'precision_20': 0.28
    }
}
```

### 6.3 Model Registry (Future)

**When to implement:** After validating TabM approach.

**Tools:** MLflow, Weights & Biases, or simple JSON registry.

```json
// models/registry.json
{
  "models": [
    {
      "id": "tabm_position_aware_v1",
      "path": "models/tabm_position_aware_v1.pkl",
      "feature_version": "v1",
      "metrics": {
        "hit_rate": 0.18,
        "recall_20": 0.85
      },
      "status": "production"
    }
  ]
}
```

---

## 7. Technology Stack

### 7.1 Core Dependencies

```
# requirements.txt
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
torch>=2.0.0              # For TabM (if PyTorch-based)
tabm>=0.1.0               # TabM library (adjust based on actual package)
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
pytest>=7.4.0
```

**Why these choices?**
- **pandas:** Ubiquitous, well-understood, great for tabular data
- **scikit-learn:** Standard preprocessing, metrics, baselines
- **PyTorch:** TabM likely built on PyTorch (verify from ICLR 2025 paper)
- **pytest:** Industry-standard testing

**Boring Technology = Good:** Avoid bleeding-edge packages. Prioritize stability, debuggability, community support.

### 7.2 Development Environment

**Recommended:** Conda or venv for reproducibility.

```bash
# Create environment
$ conda create -n c5_qstate python=3.10
$ conda activate c5_qstate
$ pip install -r requirements.txt

# Or with venv
$ python -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate
$ pip install -r requirements.txt
```

---

## 8. Risks and Mitigations

### 8.1 Overfitting Risk

**Risk:** Model memorizes training data, fails to generalize.

**Mitigations:**
1. **Temporal validation:** No shuffling, realistic test set (last 20%)
2. **Early stopping:** Monitor validation loss
3. **Feature ablation:** Remove features that overfit
4. **Regularization:** TabM's ensemble averaging provides implicit regularization

**Red Flag:** Training hit-rate >> Test hit-rate (e.g., 50% vs 10%)

### 8.2 Feature Leakage Risk

**Risk:** Future information leaks into training features.

**Mitigations:**
1. **Careful feature engineering:** Only use data *before* prediction point
2. **Unit tests:** Verify no look-ahead bias
3. **Manual review:** Check all rolling/lagged features

**Example Leakage:**
```python
# BAD: Uses future information
df['next_draw_QS_1'] = df['QS_1'].shift(-1)  # Leaks next draw!

# GOOD: Uses only past information
df['prev_draw_QS_1'] = df['QS_1'].shift(1)  # Previous draw
```

### 8.3 TabM Integration Risk

**Risk:** TabM library is new (ICLR 2025), may have bugs or API changes.

**Mitigations:**
1. **Version pinning:** Lock TabM version in `requirements.txt`
2. **Wrapper abstraction:** `models/tabm_wrapper.py` isolates TabM API
3. **Fallback:** If TabM fails, use XGBoost/LightGBM as substitute

**Contingency Plan:** XGBoost with same architecture (position-aware heads).

---

## 9. Success Metrics (Research Goals)

### 9.1 Primary Research Questions

1. **Does position-aware modeling beat position-agnostic?**
   - Measure: Hit-rate difference between Option A vs Option B
   - Hypothesis: Position-aware should win (exploits 2.5x predictability gap)

2. **Do recency features dominate?**
   - Ablation: Train with/without recency features
   - Measure: Feature importance, performance delta

3. **Does TabM beat traditional ensembles (XGBoost, LightGBM)?**
   - Comparison: TabM vs XGBoost (same features, same architecture)
   - Measure: Hit-rate, training time, model size

### 9.2 Experiment Tracking

**Log every experiment:**
```python
# experiments.json
{
  "experiment_id": "exp_001",
  "timestamp": "2026-04-03T12:00:00Z",
  "model_type": "tabm_position_aware",
  "feature_version": "v1",
  "hyperparams": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.01
  },
  "metrics": {
    "train_hit_rate": 0.22,
    "val_hit_rate": 0.18,
    "test_hit_rate": 0.16
  },
  "notes": "Baseline position-aware model"
}
```

**Maintain:** `docs/EXPERIMENTS.md` with all experiment logs.

---

## 10. Next Steps (Implementation Roadmap)

### Phase 1: Foundation (Week 1)
- [x] Mary's strategic analysis ✅
- [x] Winston's architecture design ✅
- [ ] Amelia: Data validation pipeline
- [ ] Amelia: FeatureEngineer class (Priority 1-4 features)
- [ ] Amelia: Unit tests for features

### Phase 2: Baselines (Week 2)
- [ ] Random baseline implementation
- [ ] Frequency-based baseline implementation
- [ ] Evaluation metrics (Precision@20, Recall@20, Hit-rate)
- [ ] Baseline performance documentation

### Phase 3: TabM Integration (Week 3-4)
- [ ] TabM wrapper (`models/tabm_wrapper.py`)
- [ ] Position-agnostic model (Option A)
- [ ] Position-aware model (Option B)
- [ ] Training pipeline (`scripts/train.py`)
- [ ] Inference pipeline (`scripts/predict.py`)

### Phase 4: Experimentation (Week 5-6)
- [ ] Ablation studies (feature importance)
- [ ] Hyperparameter tuning
- [ ] Position-aware vs position-agnostic comparison
- [ ] TabM vs XGBoost comparison

### Phase 5: Documentation & Reporting (Week 7)
- [ ] Experiment log (`docs/EXPERIMENTS.md`)
- [ ] Final report for Grok
- [ ] GitHub repo polish (README, examples)
- [ ] Research paper draft (if publication target)

---

## 11. Conclusion

This architecture balances **research flexibility** (easy experimentation, clear ablations) with **production-ready patterns** (versioned features, reproducible pipelines, modular code).

**Key Design Decisions:**
1. **Position-aware modeling** — Justified by 2.5x predictability difference
2. **Temporal validation** — Prevents leakage, measures realistic performance
3. **Modular features** — Easy to add/remove, version-controlled
4. **Boring technology** — pandas, scikit-learn, PyTorch — proven and stable

**Expected Outcomes:**
- Hit-rate: 15-20% (6-8x random baseline)
- Recall@20: 80-90% (4-5 of 5 actual numbers captured)
- Research contribution: Validate TabM for position-aware tabular prediction

**Next:** Amelia implements `features/feature_engineering.py` based on Mary's priority order.

---

**Winston (System Architect)**
*"Simple solutions that scale when needed. This architecture ships."*
