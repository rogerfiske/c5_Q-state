# Phase 1: Exploratory Data Analysis Report

**Generated**: 2026-04-09 12:11:50
**Project**: c5_Q-state
**Data Location**: `data/raw/`

## 1. Dataset Overview

| Dataset | Rows | Cols | Date Range | Unique Dates | Duplicates | Nulls |
|---------|------|------|------------|-------------|------------|-------|
| QS_1 | 11,762 | 2 | 1992-02-04 → 2026-04-08 | 11,762 | 0 | 0 |
| QS_2 | 11,762 | 2 | 1992-02-04 → 2026-04-08 | 11,762 | 0 | 0 |
| QS_3 | 11,762 | 2 | 1992-02-04 → 2026-04-08 | 11,762 | 0 | 0 |
| QS_4 | 11,762 | 2 | 1992-02-04 → 2026-04-08 | 11,762 | 0 | 0 |
| QS_5 | 11,762 | 2 | 1992-02-04 → 2026-04-08 | 11,762 | 0 | 0 |
| Combined | 11,762 | 6 | 1992-02-04 → 2026-04-08 | 11,762 | 0 | 0 |

## 2. Descriptive Statistics

| Variable | Count | Mean | Std | Min | 25% | 50% | 75% | Max | Skew | Kurtosis | Mode | IQR |
|----------|-------|------|-----|-----|-----|-----|-----|-----|------|----------|------|-----|
| QS_1 | 11,762 | 6.66 | 5.21 | 1 | 3.0 | 5.0 | 9.0 | 30 | 1.205 | 1.248 | 1 | 6.0 |
| QS_2 | 11,762 | 13.21 | 6.61 | 2 | 8.0 | 12.0 | 18.0 | 35 | 0.509 | -0.328 | 10 | 10.0 |
| QS_3 | 11,762 | 19.91 | 7.01 | 3 | 15.0 | 20.0 | 25.0 | 37 | -0.009 | -0.699 | 20 | 10.0 |
| QS_4 | 11,762 | 26.65 | 6.60 | 4 | 22.0 | 27.0 | 32.0 | 38 | -0.488 | -0.358 | 30 | 10.0 |
| QS_5 | 11,762 | 33.24 | 5.32 | 9 | 30.0 | 35.0 | 37.0 | 39 | -1.192 | 1.172 | 39 | 7.0 |

## 3. Complete Value Frequency Table

| Value | QS_1 | QS_2 | QS_3 | QS_4 | QS_5 |
|-------|------|------|------|------|------|
|  1 |  1505 |     0 |     0 |     0 |     0 |
|  2 |  1398 |   181 |     0 |     0 |     0 |
|  3 |  1205 |   310 |    19 |     0 |     0 |
|  4 |  1008 |   412 |    44 |     1 |     0 |
|  5 |   976 |   476 |    66 |     4 |     0 |
|  6 |   837 |   573 |   122 |     4 |     0 |
|  7 |   728 |   618 |   159 |    22 |     0 |
|  8 |   626 |   665 |   186 |    25 |     0 |
|  9 |   583 |   653 |   251 |    30 |     2 |
| 10 |   493 |   715 |   314 |    54 |     1 |
| 11 |   422 |   695 |   373 |    75 |     4 |
| 12 |   339 |   646 |   398 |    94 |    12 |
| 13 |   340 |   652 |   461 |   110 |    15 |
| 14 |   245 |   587 |   499 |   150 |    12 |
| 15 |   212 |   600 |   487 |   189 |    26 |
| 16 |   146 |   517 |   544 |   210 |    36 |
| 17 |   128 |   485 |   562 |   230 |    53 |
| 18 |   132 |   419 |   575 |   300 |    58 |
| 19 |   105 |   390 |   575 |   350 |    60 |
| 20 |    81 |   375 |   578 |   346 |    80 |
| 21 |    63 |   290 |   574 |   434 |   105 |
| 22 |    47 |   313 |   540 |   475 |   127 |
| 23 |    44 |   230 |   570 |   510 |   155 |
| 24 |    35 |   211 |   562 |   528 |   178 |
| 25 |    20 |   194 |   502 |   564 |   238 |
| 26 |    19 |   145 |   499 |   591 |   242 |
| 27 |    11 |   122 |   479 |   624 |   308 |
| 28 |     7 |    78 |   388 |   666 |   362 |
| 29 |     3 |    59 |   334 |   674 |   412 |
| 30 |     4 |    60 |   298 |   681 |   499 |
| 31 |     0 |    36 |   247 |   659 |   558 |
| 32 |     0 |    27 |   182 |   661 |   643 |
| 33 |     0 |    20 |   150 |   599 |   710 |
| 34 |     0 |     7 |   124 |   583 |   818 |
| 35 |     0 |     1 |    59 |   492 |   946 |
| 36 |     0 |     0 |    26 |   397 |  1067 |
| 37 |     0 |     0 |    15 |   278 |  1196 |
| 38 |     0 |     0 |     0 |   152 |  1275 |
| 39 |     0 |     0 |     0 |     0 |  1564 |

## 4. Correlation Analysis

### Pearson Correlation

|      |   QS_1 |   QS_2 |   QS_3 |   QS_4 |   QS_5 |
|:-----|-------:|-------:|-------:|-------:|-------:|
| QS_1 | 1      | 0.6496 | 0.4573 | 0.3199 | 0.2081 |
| QS_2 | 0.6496 | 1      | 0.7091 | 0.4926 | 0.318  |
| QS_3 | 0.4573 | 0.7091 | 1      | 0.7001 | 0.449  |
| QS_4 | 0.3199 | 0.4926 | 0.7001 | 1      | 0.6417 |
| QS_5 | 0.2081 | 0.318  | 0.449  | 0.6417 | 1      |

### Spearman Rank Correlation

|      |   QS_1 |   QS_2 |   QS_3 |   QS_4 |   QS_5 |
|:-----|-------:|-------:|-------:|-------:|-------:|
| QS_1 | 1      | 0.6209 | 0.4224 | 0.2853 | 0.1707 |
| QS_2 | 0.6209 | 1      | 0.6933 | 0.4649 | 0.2781 |
| QS_3 | 0.4224 | 0.6933 | 1      | 0.6864 | 0.4143 |
| QS_4 | 0.2853 | 0.4649 | 0.6864 | 1      | 0.6132 |
| QS_5 | 0.1707 | 0.2781 | 0.4143 | 0.6132 | 1      |

## 5. Time-Series Gap Analysis

| Variable | Mean Gap (days) | Std Gap | Gaps > 7d | Gaps > 30d |
|----------|----------------|---------|-----------|------------|
| QS_1 | 1.06 | 0.35 | 0 | 0 |
| QS_2 | 1.06 | 0.35 | 0 | 0 |
| QS_3 | 1.06 | 0.35 | 0 | 0 |
| QS_4 | 1.06 | 0.35 | 0 | 0 |
| QS_5 | 1.06 | 0.35 | 0 | 0 |

## 6. Day-of-Week Distribution

| Day | Count | % |
|-----|-------|---|
| Monday | 1,658 | 14.1% |
| Tuesday | 1,783 | 15.2% |
| Wednesday | 1,549 | 13.2% |
| Thursday | 1,782 | 15.2% |
| Friday | 1,781 | 15.1% |
| Saturday | 1,555 | 13.2% |
| Sunday | 1,654 | 14.1% |

## 7. Visualizations

All plots saved to `reports/phase1_eda/`:

- `timeseries_all_variables.png`
- `histograms_value_distributions.png`
- `correlation_heatmaps.png`
- `rolling_mean_overlay.png`
- `rolling_std_overlay.png`
- `boxplots_comparison.png`
- `pairplot_combined.png`
- `yearly_heatmap.png`

## 8. Key Insights & Observations

1. **Strict ordering by variable index**: QS_1(mean=6.7) < QS_2(13.2) < QS_3(19.9) < QS_4(26.6) < QS_5(33.2). The 5 variables are **order statistics** from sorted draws on {1..39} -- each QS_x occupies a distinct band of the value space.
2. **Strong adjacent-pair correlation (tridiagonal pattern)**: QS_2/QS_3 Pearson=0.709, QS_3/QS_4=0.700, QS_1/QS_2=0.650, QS_4/QS_5=0.642. Distant pairs are weaker: QS_1/QS_5=0.208. Adjacent variables share significant mutual information.
3. **Mirror-image skewness**: QS_1 is strongly right-skewed (+1.21, mode=1, pile-up at floor), while QS_5 is strongly left-skewed (-1.19, mode=39, pile-up at ceiling). QS_3 sits at the symmetric center (skew=-0.01). This confirms order-statistic structure.
4. **Discrete integer values on {1..39}**: Not continuous data. QS_1 uses 30 of 39 possible values, QS_5 uses 31. Each variable has a constrained effective range.
5. **No temporal trend or seasonality**: Rolling means are stationary across 34 years. The process appears to be memoryless / i.i.d. across time.
6. **Near-daily spacing with systematic gaps**: Mean gap=1.06 days, max=4 days. All 7 weekdays are represented (~13-15% each). 340 occurrences of 2-day gaps and 125 occurrences of 4-day gaps suggest occasional skipped days.

## 9. Data Readiness Assessment

| Check | Status |
|-------|--------|
| All 6 files present and loadable | PASS |
| Dates parseable and consistent | PASS |
| No null values | PASS |
| No duplicate dates | PASS |
| Values within expected range (1-39) | PASS |
| Combined dataset aligns with individuals | PASS |
| Sufficient history for modeling (34+ years) | PASS |

**Verdict: Data is CLEAN and READY for modeling.**
