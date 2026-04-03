# Executive Summary: c5_Q-state TabM Lottery Prediction

**Project:** c5_Q-state Lottery Prediction using TabM
**Date:** 2026-04-03
**Status:** ⚠️ EVALUATION COMPLETE - CRITICAL FINDINGS
**PM:** John (via Grok)
**Author:** Amelia (Senior Developer)

---

## TL;DR - Critical Findings

**ALL MODELS PERFORM AT OR BELOW RANDOM BASELINE**

| Model | Hit-rate | vs Random | Expected | Status |
|-------|----------|-----------|----------|--------|
| **Random** | 2.47% | 1.00x | ~2.4% | ✓ Baseline |
| **Frequency** | 2.55% | 1.03x | 5-10% | ❌ 50% below expectation |
| **XGBoost** | 2.30% | 0.93x | 10-15% | ❌ 80% below expectation, WORSE than random |
| **TabM** | 2.30% | 0.93x | >15% | ⚠️ XGBoost fallback (PyTorch impl required) |

**Conclusion:** Lottery draws exhibit **extremely weak predictive signals**. Position-specific patterns identified by Mary do not translate to actionable prediction advantage.

---

## Detailed Results

### Full Metrics Table

```
    Model  Precision@20  Recall@20  Hit-rate
   Random      0.1266      0.5065    0.0247 (2.47%)
Frequency      0.1281      0.5124    0.0255 (2.55%)
  XGBoost      0.1271      0.5082    0.0230 (2.30%)
     TabM      0.1271      0.5082    0.0230 (2.30%)
```

### Metric Definitions

- **Precision@20** = (# actual numbers in top-20) / 20
- **Recall@20** = (# actual numbers in top-20) / 5
- **Hit-rate** = 1 if all 5 actual numbers in top-20, else 0

### Temporal Split

- **Train:** 8,229 draws (70%) - 1992-02-04 to 2016-08-05
- **Val:** 1,175 draws (10%) - 2016-08-06 to 2019-10-24
- **Test:** 2,352 draws (20%) - 2019-10-25 to 2026-04-02
- **Features:** 1,408 engineered features per draw

---

## Analysis

### Why Did This Happen?

**1. Frequency Baseline Underperformance (2.55% vs expected 5-10%)**

The Frequency baseline barely outperforms Random (1.03x lift). This indicates:
- Lottery number distribution is **nearly uniform** across draws
- Historical frequency provides **minimal predictive signal**
- Test set (2019-2026) distribution differs from train set (1992-2016)

**2. XGBoost Performing BELOW Random (2.30% < 2.47%)**

This is a **red flag** indicating:
- **Overfitting:** XGBoost learned spurious patterns in training data that don't generalize
- **Weak signals:** 1,408 engineered features capture noise, not signal
- **Early stopping ineffective:** Validation set doesn't prevent overfitting

**3. TabM Implementation Gap**

- `tabm` package (v0.0.3) installed successfully
- **BUT:** TabM is a PyTorch `nn.Module`, not sklearn-style classifier
- Requires custom PyTorch training loop with ensemble batching
- Current wrapper fell back to XGBoost
- Full PyTorch implementation deferred (see Recommendations)

### Reconciling with Mary's Strategic Analysis

Mary identified:
- Position-specific predictability gradient (QS_1: 0.236, QS_5: 0.232 >> QS_3: 0.088)
- Momentum/recency effects (repeats 1.5-2.9x above random)
- NO cylindrical behavior

**Key Insight:** These are **relative** patterns, not absolute predictability.
- Normalized entropy measures show *differential* predictability across positions
- Absolute predictability remains **extremely low** for all positions
- Lottery is designed to be unpredictable - working as intended

---

## What Worked

✓ **Feature Engineering:** 1,408 features generated without leakage, validated
✓ **Temporal Split:** 70/10/20 split respects chronological order
✓ **Evaluation Pipeline:** All 4 models trained and evaluated successfully
✓ **Random Baseline:** 2.47% matches theoretical expectation (2.4%)
✓ **Reproducibility:** All results saved to `results_table.csv`
✓ **Git Workflow:** BMAD-METHOD compliant commits, pushed to GitHub

---

## What Didn't Work

❌ **Frequency Baseline:** Expected 5-10%, got 2.55% (minimal lift over random)
❌ **XGBoost:** Expected 10-15%, got 2.30% (WORSE than random, overfitting)
❌ **TabM:** Requires PyTorch custom training loop (not implemented)
❌ **Predictive Signals:** 1,408 features fail to capture actionable patterns

---

## Recommendations

### Immediate Next Steps

**1. Investigate XGBoost Overfitting**

- **Action:** Reduce model complexity (lower `max_depth`, fewer `n_estimators`)
- **Action:** Add stronger regularization (`min_child_weight`, `gamma`)
- **Action:** Cross-validate on validation set to ensure early stopping works
- **Hypothesis:** Current hyperparameters too complex for weak signal

**2. Simplify Feature Set**

- **Action:** Feature selection (drop low-importance features)
- **Action:** Try **only** Mary's Priority 1 features (position-specific recency)
- **Hypothesis:** 1,408 features introduce noise, simpler may generalize better

**3. Validate Data Quality**

- **Action:** Check train/val/test distribution shifts
- **Action:** Verify target encoding (y matrix correctness)
- **Action:** Sample predictions manually to ensure no implementation bugs

### TabM Implementation Options

**Option A: Full PyTorch Training Loop (HIGH EFFORT)**

- Implement custom TabM training with ensemble batching
- Estimated effort: 2-3 days
- **ONLY** pursue if XGBoost shows promise after fixes

**Option B: Defer TabM (RECOMMENDED)**

- XGBoost already underperforming - TabM unlikely to help
- Lottery prediction may be fundamentally unpredictable
- Focus on understanding **why** models fail before adding complexity

### Strategic Decision Point

**Question for John/Grok:**

> Given that ALL models (including strong XGBoost baseline) perform at/below random chance, should we:
>
> **[A]** Continue debugging and refining models (pursue Recommendations 1-3)
> **[B]** Pivot to smaller problem (predict single position, not full draw)
> **[C]** Accept finding: Lottery is unpredictable as designed, document results as negative result

---

## Deliverables (Completed)

- ✅ `models/baselines.py` - Random & Frequency baselines
- ✅ `models/xgboost_wrapper.py` - XGBoost strong tabular baseline
- ✅ `models/tabm_wrapper.py` - TabM wrapper (XGBoost fallback)
- ✅ `models/trainer.py` - Temporal split & early stopping
- ✅ `models/metrics.py` - Precision@20, Recall@20, Hit-rate
- ✅ `scripts/evaluate.py` - Full evaluation pipeline (CLI)
- ✅ `results_table.csv` - 4-model comparison results
- ✅ `notebooks/03_model_evaluation.ipynb` - Visualization (not executed yet)
- ✅ Git commits with BMAD-METHOD messages
- ✅ Pushed to https://github.com/rogerfiske/c5_Q-state

---

## Files Changed

**Commit 1:** `fb3f610` - Full pipeline implementation (9 files)
**Commit 2:** `776278d` - Unicode encoding fix (4 files)

**Current status:** All code functional, results table generated, but models severely underperforming expectations.

---

## Technical Notes

### TabM Package Details

- **Version:** 0.0.3
- **Type:** PyTorch `nn.Module` (ensemble MLP backbone)
- **API:** Not sklearn-compatible, requires manual training loop
- **Key Classes:** `TabM`, `make_tabm_backbone`, `MLPBackboneEnsemble`
- **Documentation:** Requires ensemble batching strategy (see official notebook)

### XGBoost Training Details

- **Architecture:** 39 separate binary classifiers (one per number 1-39)
- **Parameters:** `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`
- **Early Stopping:** Enabled on validation set (`early_stopping_rounds=10`)
- **Result:** All 39 models trained successfully, predictions generated

---

## Conclusion

The evaluation pipeline is **fully functional** and produces **reproducible results**. However, the core finding is sobering:

> **Lottery prediction with ML appears to be extremely difficult, with all models performing at or below random chance.**

This is consistent with lottery design principles (randomness by construction) but contradicts initial expectations based on Mary's position-specific pattern analysis.

**Next step:** Awaiting strategic direction from John/Grok on whether to debug further, pivot approach, or document as negative result.

---

**Author:** Amelia (Senior Developer)
**Reviewers:** Awaiting feedback from John (PM) and Grok (Senior PM/Researcher)
**Status:** Ready for strategic review
