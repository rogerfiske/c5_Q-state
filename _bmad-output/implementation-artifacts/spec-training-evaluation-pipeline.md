---
title: 'Training and Evaluation Pipeline for TabM Lottery Prediction'
type: 'feature'
created: '2026-04-03'
status: 'in-progress'
baseline_commit: 'f6b24d8e9c952d10c2887dc3737d8d6604bb79b8'
context: ['docs/ARCHITECTURE.md', '_bmad-output/planning-artifacts/MARY_STRATEGIC_ANALYSIS.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** FeatureEngineer module is complete (1,408 features validated), but we have no baseline comparisons or TabM training pipeline to validate whether position-aware tabular ensembling beats naive approaches. Cannot measure success metrics (Precision@20, Recall@20, Hit-rate) or demonstrate TabM's advantage.

**Approach:** Implement full training and evaluation pipeline with four models: Random baseline, Frequency baseline, XGBoost wrapper (strong tabular baseline), and TabM wrapper (primary experimental model using official `tabm` package). Use Winston's temporal split (70/10/20) with early stopping on validation set. Produce comparison table showing all four approaches against success metrics on test set.

## Boundaries & Constraints

**Always:**
- Use existing FeatureEngineer without modification (1,408 features, validated)
- Temporal split: Train 70% (1992-2015), Val 10% (2015-2018), Test 20% (2018-2026)
- No data shuffling (respects temporal order, prevents leakage)
- Three success metrics: Precision@20, Recall@20, Hit-rate (binary: all 5 in top-20)
- Reproducible results (set random seeds)
- Clear docstrings and type hints

**Ask First:**
- If tabm package unavailable or causes errors, confirm how to proceed
- If training takes >10 minutes, confirm proceeding or reduce dataset size
- If test Hit-rate <5% for all models, confirm data/feature issues

**Never:**
- Modify FeatureEngineer class or features module
- Use future information in training (strict temporal ordering)
- Skip any of the four models (Random, Frequency, XGBoost, TabM all required)
- Implement position-aware TabM in first iteration (start position-agnostic, defer to future)
- Use XGBoost as TabM substitute (both must be implemented separately)

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Random baseline | Test set (2,351 draws) | Top-20 random numbers per draw, metrics computed | N/A (deterministic with seed) |
| Frequency baseline | Train set frequencies | Top-20 most frequent numbers, metrics on test set | N/A (deterministic) |
| XGBoost training | Train features (8,229 × 1408), Val features (1,176 × 1408) | Trained XGBoost model, early-stopped on validation loss | Halt if OOM or divergence |
| XGBoost inference | Test features (2,351 × 1408) | Top-20 numbers per draw, metrics computed | Halt if NaN predictions |
| TabM training | Train features (8,229 × 1408), Val features (1,176 × 1408) | Trained TabM model, early-stopped on validation loss | Halt if OOM or divergence |
| TabM inference | Test features (2,351 × 1408) | Top-20 numbers per draw, metrics computed | Halt if NaN predictions |
| Metrics computation | Predictions (top-20 numbers), Actuals (5 numbers) | Precision@20, Recall@20, Hit-rate | Error if list lengths mismatch |
| No overlap case | top-20 = [1-20], actuals = [21-25] | Precision=0, Recall=0, Hit-rate=0 | No error, valid edge case |
| Perfect case | top-20 contains all 5 actuals | Precision=5/20=0.25, Recall=5/5=1.0, Hit-rate=1 | No error, valid edge case |

</frozen-after-approval>

## Code Map

- `features/feature_engineering.py` -- Existing FeatureEngineer class (no modifications)
- `data/raw/c5_Q-state.csv` -- Raw lottery data (11,756 draws)
- `models/` -- New directory for model implementations
- `models/__init__.py` -- Package initialization
- `models/baselines.py` -- Random and Frequency-based baseline models
- `models/xgboost_wrapper.py` -- XGBoost wrapper (strong tabular baseline)
- `models/tabm_wrapper.py` -- TabM wrapper (primary experimental model)
- `models/trainer.py` -- Training loop with temporal split and early stopping
- `models/metrics.py` -- Precision@20, Recall@20, Hit-rate computation
- `scripts/` -- New directory for CLI tools
- `scripts/evaluate.py` -- Main evaluation script (trains all models, produces results table)
- `notebooks/03_model_evaluation.ipynb` -- Results visualization and comparison
- `requirements.txt` -- Add xgboost and tabm dependencies

## Tasks & Acceptance

**Execution:**
- [x] `models/__init__.py` -- Create package initialization -- Enables `from models import` syntax
- [x] `models/metrics.py` -- Implement Precision@20, Recall@20, Hit-rate functions -- Core evaluation metrics needed by all models
- [x] `models/baselines.py` -- Implement RandomBaseline and FrequencyBaseline classes -- Establishes performance floor
- [x] `models/xgboost_wrapper.py` -- Implement XGBoostModel class (strong tabular baseline) -- Position-agnostic multi-label classifier using XGBoost
- [x] `models/tabm_wrapper.py` -- Implement TabMModel class (primary experimental model) -- Position-agnostic multi-label classifier using official tabm package
- [x] `models/trainer.py` -- Implement temporal_split() and train_with_early_stopping() -- Handles 70/10/20 split and validation-based early stopping
- [x] `scripts/evaluate.py` -- Implement CLI script that trains all 4 models and outputs results table -- End-to-end evaluation pipeline
- [x] `notebooks/03_model_evaluation.ipynb` -- Create notebook that runs evaluate.py and visualizes 4-model comparison -- Human-readable results presentation
- [x] `requirements.txt` -- Add xgboost>=2.0.0 and tabm dependencies -- Required for both strong baseline and primary model

**Acceptance Criteria:**
- Given raw data at `data/raw/c5_Q-state.csv`, when running `python scripts/evaluate.py`, then it produces a results table with Random/Frequency/XGBoost/TabM rows and Precision@20/Recall@20/Hit-rate columns
- Given test set predictions, when computing metrics, then Precision@20 ∈ [0, 0.25], Recall@20 ∈ [0, 1.0], Hit-rate ∈ {0, 1}
- Given Random baseline, when evaluated on test set, then Hit-rate ≈ 2.4% (theoretical expectation)
- Given Frequency baseline, when evaluated on test set, then Hit-rate > Random (demonstrates non-trivial pattern)
- Given XGBoost model, when evaluated on test set, then Hit-rate > Frequency (demonstrates strong tabular baseline)
- Given TabM model, when evaluated on test set, then Hit-rate ≥ XGBoost (demonstrates TabM's parameter-efficient advantage)
- Given training on 8,229 draws (70%), when validation on 1,176 draws (10%), then models early-stop before overfitting
- Given notebook execution, when all cells run, then 4-model comparison table is displayed and no errors occur

## Design Notes

**Temporal Split Implementation:**
```python
def temporal_split(df, train_pct=0.7, val_pct=0.1, test_pct=0.2):
    """Split chronologically (no shuffling)."""
    n = len(df)
    train_idx = int(n * train_pct)
    val_idx = train_idx + int(n * val_pct)

    train = df.iloc[:train_idx]
    val = df.iloc[train_idx:val_idx]
    test = df.iloc[val_idx:]
    return train, val, test
```

**XGBoost Wrapper (Strong Tabular Baseline):**
Multi-label classification using XGBoost. Each of 39 numbers is a binary target. Train single XGBoost model with 39 outputs (logistic regression per number). At inference, rank numbers by probability and select top-20. This serves as the strong tabular baseline to compare against TabM.

**TabM Wrapper (Primary Experimental Model):**
Use official `tabm` package (ICLR 2025) for parameter-efficient tabular ensembling. Same multi-label setup as XGBoost but leveraging TabM's ensemble architecture. This is the primary research model testing position-aware prediction.

**Metrics Definitions:**
- **Precision@20** = (# actual numbers in top-20) / 20
- **Recall@20** = (# actual numbers in top-20) / 5
- **Hit-rate** = 1 if all 5 actual numbers are in top-20, else 0

**Expected Performance Progression (from Mary's analysis):**
- Random: ~2.4% hit-rate (C(20,5) / C(39,5))
- Frequency: ~5-10% hit-rate (exploits non-uniform distribution)
- XGBoost: ~10-15% hit-rate (strong tabular baseline)
- TabM: Target >15% hit-rate (exploits position-specific predictability + momentum via parameter-efficient ensembling)

## Verification

**Commands:**
- `python scripts/evaluate.py` -- expected: Completes without errors, prints 4-model results table
- `pytest tests/test_metrics.py` -- expected: All metrics tests pass (if time permits to add tests)
- `python -c "from models import baselines, xgboost_wrapper, tabm_wrapper, trainer, metrics"` -- expected: All imports successful
- `jupyter nbconvert --to notebook --execute notebooks/03_model_evaluation.ipynb` -- expected: Notebook executes without errors

**Manual checks (if no CLI):**
- Results table shows 4 rows (Random, Frequency, XGBoost, TabM) × 3 metric columns
- Performance progression: TabM ≥ XGBoost > Frequency > Random (demonstrates improvement)
- No NaN values in results table
- Notebook displays 4-model comparison visualizations clearly
