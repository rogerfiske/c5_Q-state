# Phase 3: Expanded Model Zoo & Rolling Backtest Report

**Generated**: 2026-04-10 11:39:57
**Backtest**: 365 days | **Models**: 10 | **April 9 Actuals**: 1, 5, 17, 23, 32

## 1. Exact Analytical Baseline

```
P(X_(k) = v) = C(v-1, k-1) * C(39-v, 5-k) / C(39, 5)
```

| Position | Support | Peak | Peak P | Entropy (bits) |
|----------|---------|------|--------|----------------|
| QS_1 | [1..35] | 1 | 0.1282 | 4.04 |
| QS_2 | [2..36] | 10 | 0.0571 | 4.68 |
| QS_3 | [3..37] | 20 | 0.0508 | 4.82 |
| QS_4 | [4..38] | 30 | 0.0571 | 4.68 |
| QS_5 | [5..39] | 39 | 0.1282 | 4.04 |

## 2. Backtest: Top-5 Hit Rate

| Model | QS_1 | QS_2 | QS_3 | QS_4 | QS_5 |
|-------|------|------|------|------|------|
| Analytical | 47.95% | 26.58% | 26.03% | 28.49% | 50.41% |
| CatBoost | 45.75% | 25.48% | 23.01% | 26.58% | 47.67% |
| Ensemble_XLC | 43.29% | 24.11% | 22.74% | 27.12% | 48.22% |
| HistFreq | 47.95% | 27.67% | 27.40% | 28.49% | 50.41% |
| LightGBM | 41.64% | 24.38% | 23.29% | 17.26% | 43.56% |
| MLP | 44.93% | 25.75% | 21.10% | 29.32% | 50.41% |
| MonteCarlo | 47.95% | 26.58% | 26.03% | 28.77% | 50.41% |
| RandomForest | 48.22% | 29.04% | 28.49% | 26.58% | 49.86% |
| TabICLv2 | 47.95% | 27.40% | 25.48% | 28.49% | 50.41% |
| XGBoost | 44.66% | 24.38% | 23.56% | 28.49% | 47.95% |

## 3. Backtest: Mean Neg-Log-Probability

| Model | QS_1 | QS_2 | QS_3 | QS_4 | QS_5 |
|-------|------|------|------|------|------|
| Analytical | 2.849 | 3.235 | 3.329 | 3.221 | 2.798 |
| CatBoost | 2.875 | 3.271 | 3.328 | 3.228 | 2.806 |
| Ensemble_XLC | 2.917 | 3.321 | 3.367 | 3.424 | 2.853 |
| HistFreq | 2.850 | 3.240 | 3.330 | 3.224 | 2.802 |
| LightGBM | 3.022 | 3.459 | 3.484 | 12.021 | 3.246 |
| MLP | 2.873 | 3.282 | 3.336 | 3.241 | 2.820 |
| MonteCarlo | 2.849 | 3.236 | 3.329 | 3.220 | 2.798 |
| RandomForest | 2.863 | 3.249 | 3.329 | 3.229 | 2.803 |
| TabICLv2 | 2.859 | 3.243 | 3.332 | 3.224 | 2.800 |
| XGBoost | 2.931 | 3.339 | 3.383 | 3.264 | 2.839 |

## 4. Best Model per Variable

| Variable | Best Model | NegLogP | Top-5 Hit |
|----------|-----------|---------|-----------|
| QS_1 | **Analytical** | 2.8493 | 47.95% |
| QS_2 | **Analytical** | 3.2353 | 26.58% |
| QS_3 | **CatBoost** | 3.3281 | 23.01% |
| QS_4 | **MonteCarlo** | 3.2201 | 28.77% |
| QS_5 | **Analytical** | 2.7979 | 50.41% |

## 5. April 9, 2026 Evaluation

Actuals: **QS_1=1, QS_2=5, QS_3=17, QS_4=23, QS_5=32**

### QS_1 (actual = 1)

| Model | Top-5 | Hit? | P(actual) |
|-------|-------|------|-----------|
| LightGBM | [1, 2, 6, 5, 3] | YES | 0.1525 |
| XGBoost | [1, 2, 6, 5, 3] | YES | 0.1469 |
| Ensemble_XLC | [1, 2, 6, 3, 5] | YES | 0.1459 |
| CatBoost | [1, 2, 3, 5, 6] | YES | 0.1384 |
| TabICLv2 | [1, 2, 6, 3, 5] | YES | 0.1304 |
| RandomForest | [1, 2, 3, 5, 6] | YES | 0.1295 |
| HistFreq | [1, 2, 3, 4, 5] | YES | 0.1284 |
| Analytical | [1, 2, 3, 4, 5] | YES | 0.1282 |
| MonteCarlo | [1, 2, 3, 4, 5] | YES | 0.1278 |
| MLP | [2, 1, 3, 5, 4] | YES | 0.1193 |

### QS_2 (actual = 5)

| Model | Top-5 | Hit? | P(actual) |
|-------|-------|------|-----------|
| LightGBM | [5, 7, 13, 14, 9] | YES | 0.2558 |
| Ensemble_XLC | [5, 7, 14, 9, 13] | YES | 0.1173 |
| XGBoost | [12, 9, 14, 10, 7] | no | 0.0561 |
| MonteCarlo | [10, 11, 8, 9, 12] | no | 0.0420 |
| Analytical | [10, 11, 9, 12, 8] | no | 0.0416 |
| RandomForest | [10, 12, 11, 9, 7] | no | 0.0412 |
| TabICLv2 | [10, 11, 7, 12, 13] | no | 0.0409 |
| HistFreq | [10, 11, 8, 9, 13] | no | 0.0408 |
| CatBoost | [15, 7, 14, 11, 10] | no | 0.0402 |
| MLP | [10, 12, 8, 7, 9] | no | 0.0380 |

### QS_3 (actual = 17)

| Model | Top-5 | Hit? | P(actual) |
|-------|-------|------|-----------|
| CatBoost | [18, 19, 17, 23, 14] | YES | 0.0538 |
| TabICLv2 | [18, 20, 17, 19, 24] | YES | 0.0504 |
| Analytical | [20, 19, 21, 18, 22] | no | 0.0481 |
| HistFreq | [19, 21, 20, 23, 18] | no | 0.0474 |
| MonteCarlo | [19, 20, 21, 22, 18] | no | 0.0474 |
| RandomForest | [20, 19, 18, 16, 23] | no | 0.0470 |
| MLP | [19, 18, 20, 21, 17] | YES | 0.0453 |
| Ensemble_XLC | [18, 19, 13, 14, 24] | no | 0.0429 |
| XGBoost | [14, 18, 11, 19, 24] | no | 0.0374 |
| LightGBM | [13, 33, 19, 18, 14] | no | 0.0374 |

### QS_4 (actual = 23)

| Model | Top-5 | Hit? | P(actual) |
|-------|-------|------|-----------|
| TabICLv2 | [30, 29, 27, 28, 32] | no | 0.0446 |
| HistFreq | [30, 29, 28, 32, 31] | no | 0.0434 |
| Analytical | [30, 29, 31, 28, 32] | no | 0.0428 |
| MonteCarlo | [30, 31, 29, 28, 27] | no | 0.0420 |
| MLP | [30, 32, 28, 29, 27] | no | 0.0407 |
| RandomForest | [27, 31, 29, 32, 28] | no | 0.0383 |
| CatBoost | [31, 29, 32, 27, 24] | no | 0.0350 |
| Ensemble_XLC | [29, 20, 24, 31, 27] | no | 0.0273 |
| XGBoost | [29, 24, 31, 20, 28] | no | 0.0258 |
| LightGBM | [20, 29, 24, 27, 31] | no | 0.0209 |

### QS_5 (actual = 32)

| Model | Top-5 | Hit? | P(actual) |
|-------|-------|------|-----------|
| TabICLv2 | [39, 38, 37, 36, 35] | no | 0.0757 |
| LightGBM | [39, 37, 35, 38, 33] | no | 0.0628 |
| MonteCarlo | [39, 38, 37, 36, 35] | no | 0.0558 |
| Analytical | [39, 38, 37, 36, 35] | no | 0.0546 |
| HistFreq | [39, 38, 37, 36, 35] | no | 0.0541 |
| Ensemble_XLC | [39, 37, 35, 36, 38] | no | 0.0538 |
| RandomForest | [39, 37, 38, 36, 35] | no | 0.0519 |
| XGBoost | [39, 37, 36, 35, 38] | no | 0.0502 |
| CatBoost | [39, 37, 38, 36, 35] | no | 0.0484 |
| MLP | [38, 36, 37, 39, 35] | no | 0.0478 |

## 6. Value Difficulty Analysis

### Hardest 15 Values

| Value | Appearances | Top-5 Hit Rate | Max Analytical P |
|-------|------------|---------------|------------------|
| 25 | 470 | 4.68% | 0.0492 |
| 26 | 490 | 7.35% | 0.0519 |
| 14 | 560 | 7.86% | 0.0519 |
| 33 | 590 | 8.31% | 0.0625 |
| 15 | 380 | 9.47% | 0.0492 |
| 7 | 420 | 10.48% | 0.0625 |
| 16 | 410 | 10.73% | 0.0461 |
| 17 | 560 | 11.43% | 0.0481 |
| 24 | 470 | 11.70% | 0.0461 |
| 6 | 430 | 14.42% | 0.0711 |
| 22 | 350 | 16.57% | 0.0496 |
| 34 | 450 | 17.33% | 0.0711 |
| 27 | 520 | 17.69% | 0.0542 |
| 23 | 470 | 20.00% | 0.0481 |
| 13 | 520 | 21.15% | 0.0542 |

### Easiest 10 Values

| Value | Appearances | Top-5 Hit Rate | Max Analytical P |
|-------|------------|---------------|------------------|
| 31 | 390 | 48.46% | 0.0564 |
| 35 | 500 | 52.80% | 0.0805 |
| 36 | 380 | 55.00% | 0.0909 |
| 4 | 420 | 66.67% | 0.0909 |
| 3 | 470 | 70.43% | 0.1023 |
| 37 | 490 | 74.49% | 0.1023 |
| 2 | 520 | 84.81% | 0.1147 |
| 38 | 440 | 88.18% | 0.1147 |
| 39 | 480 | 97.92% | 0.1282 |
| 1 | 410 | 99.76% | 0.1282 |

### Value 23 Deep Dive

- Appearances: 470
- Overall hit rate: 20.00%
- Per-model:

| Model | Hit Rate |
|-------|----------|
| HistFreq | 42.55% |
| RandomForest | 34.04% |
| XGBoost | 29.79% |
| Ensemble_XLC | 27.66% |
| LightGBM | 27.66% |
| CatBoost | 21.28% |
| MLP | 8.51% |
| TabICLv2 | 8.51% |
| Analytical | 0.00% |
| MonteCarlo | 0.00% |

Analytical probability per position:

- QS_1: **0.0032**
- QS_2: **0.0214**
- QS_3: **0.0481**
- QS_4: **0.0428**
- QS_5: **0.0127**

## 7. Conclusions

1. **The analytical baseline is the information-theoretic optimum.** All well-calibrated models converge to the same distribution.

2. **No ML model systematically beats the exact PMF** across the backtest. Minor variations are statistical noise.

3. **Value difficulty is position-determined**: extreme values (1-5, 35-39) are easy; middle values (15-25) are hard. Value 23 falls in the hardest zone.

4. **Recommendation**: Use the **Analytical PMF** as the production model for all variables. It requires zero training, is provably optimal, and incurs no computation cost.

## 8. Artifacts

| File | Description |
|------|-------------|
| `backtest_results.csv` | Full backtest log |
| `model_summary.csv` | Aggregated metrics per model/variable |
| `april9_evaluation.csv` | April 9 2026 evaluation |
| `analytical_pmfs.json` | Exact combinatorial PMFs |
| `value_difficulty.json` | Per-value difficulty data |
| `model_comparison_heatmap.png` | Hit rate + NegLogP heatmaps |
| `value_difficulty_chart.png` | Per-value hit rate bar chart |
| `analytical_pmf_overlay.png` | Exact PMF curves |
| `april9_results_chart.png` | April 9 per-model performance |
