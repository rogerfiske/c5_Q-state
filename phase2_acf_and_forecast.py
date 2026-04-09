"""
c5_Q-state Phase 2: Autocorrelation Gate + TabICLv2 Forecasting
================================================================
1. ACF/PACF analysis to confirm i.i.d. hypothesis
2. TabICLv2 next-day top-5 predictions for April 9, 2026

Output: reports/phase2_acf_pacf.png, reports/phase2_forecast/
"""

import sys
import io
import warnings
import json
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ── Configuration ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
REPORT_DIR   = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_DIR = REPORT_DIR / "phase2_forecast"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  Phase 2: Autocorrelation Gate + TabICLv2 Forecasting")
print("=" * 70)
print(f"  Timestamp: {datetime.now().isoformat()}")
print()

# ── Load Data ────────────────────────────────────────────────────────────
datasets = {}
for i in range(1, 6):
    name = f"QS_{i}"
    df = pd.read_csv(DATA_DIR / f"c5_Q-state-{i}.csv")
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
    df = df.sort_values('date').reset_index(drop=True)
    datasets[name] = df
    print(f"  Loaded {name}: {len(df)} rows, last date = {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

df_combined = pd.read_csv(DATA_DIR / "c5_Q-state.csv")
df_combined['date'] = pd.to_datetime(df_combined['date'], format='mixed', dayfirst=False)
df_combined = df_combined.sort_values('date').reset_index(drop=True)
print(f"  Loaded combined: {len(df_combined)} rows")
print()


# ════════════════════════════════════════════════════════════════════════
# SECTION 1: AUTOCORRELATION GATE
# ════════════════════════════════════════════════════════════════════════

print("-" * 70)
print("SECTION 1: Autocorrelation Analysis (ACF/PACF, lag 1-30)")
print("-" * 70)

n_lags = 30
acf_results = {}
pacf_results = {}

# Compute ACF/PACF
for name, df in datasets.items():
    col = name
    series = df[col].values
    acf_vals = acf(series, nlags=n_lags, fft=True)
    pacf_vals = pacf(series, nlags=n_lags, method='ywm')
    acf_results[name] = acf_vals
    pacf_results[name] = pacf_vals

# Confidence band (approximate 95% CI)
n_obs = len(datasets['QS_1'])
ci_bound = 1.96 / np.sqrt(n_obs)
print(f"  95% CI bound: +/- {ci_bound:.4f} (n={n_obs})")
print()

# Report significant lags
print("  Significant ACF lags (|acf| > CI bound, excluding lag 0):")
any_significant = False
for name in datasets:
    acf_vals = acf_results[name]
    sig_lags = []
    for lag in range(1, n_lags + 1):
        if abs(acf_vals[lag]) > ci_bound:
            sig_lags.append((lag, acf_vals[lag]))
    if sig_lags:
        any_significant = True
        print(f"    {name}: ", end="")
        for lag, val in sig_lags[:10]:
            print(f"lag{lag}={val:.4f}", end="  ")
        print(f"  ({len(sig_lags)} significant out of {n_lags})")
    else:
        print(f"    {name}: NONE")

print()
print("  Significant PACF lags:")
for name in datasets:
    pacf_vals = pacf_results[name]
    sig_lags = []
    for lag in range(1, n_lags + 1):
        if abs(pacf_vals[lag]) > ci_bound:
            sig_lags.append((lag, pacf_vals[lag]))
    if sig_lags:
        print(f"    {name}: ", end="")
        for lag, val in sig_lags[:10]:
            print(f"lag{lag}={val:.4f}", end="  ")
        print(f"  ({len(sig_lags)} significant)")
    else:
        print(f"    {name}: NONE")

print()
if not any_significant:
    print("  >>> GATE RESULT: NO significant autocorrelation. i.i.d. CONFIRMED.")
else:
    # Check if the magnitudes are practically meaningful
    max_acf = max(abs(acf_results[name][lag])
                  for name in datasets
                  for lag in range(1, n_lags + 1))
    print(f"  >>> Max |ACF| across all variables/lags: {max_acf:.4f}")
    if max_acf < 0.05:
        print("  >>> GATE RESULT: Technically significant but practically negligible.")
        print("  >>> Autocorrelations are < 0.05 -- i.i.d. CONFIRMED for practical purposes.")
    else:
        print(f"  >>> GATE RESULT: Some autocorrelation detected (max={max_acf:.4f}).")
        print("  >>> Lag features may provide marginal improvement.")
print()

# ── Plot ACF/PACF ────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('ACF and PACF Analysis (QS_1 through QS_5, lags 1-30)', fontsize=16, y=0.98)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, name in enumerate(datasets):
    # ACF plot
    ax_acf = axes[i][0]
    plot_acf(datasets[name][name].values, lags=n_lags, ax=ax_acf,
             title=f'{name} ACF', color=colors[i], alpha=0.7)
    ax_acf.set_ylabel(name)

    # PACF plot
    ax_pacf = axes[i][1]
    plot_pacf(datasets[name][name].values, lags=n_lags, ax=ax_pacf,
              title=f'{name} PACF', color=colors[i], method='ywm', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.97])
acf_path = REPORT_DIR / "phase2_acf_pacf.png"
plt.savefig(acf_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {acf_path.name}")
print()


# ════════════════════════════════════════════════════════════════════════
# SECTION 2: TabICLv2 FORECASTING
# ════════════════════════════════════════════════════════════════════════

print("-" * 70)
print("SECTION 2: TabICLv2 Next-Day Forecasting (April 9, 2026)")
print("-" * 70)
print()

from tabicl import TabICLClassifier

# Strategy: Each QS_x is treated as an independent classification problem.
# Features = previous N lagged values. Target = next value (class 1-39).
# Even though i.i.d. is confirmed, we include lags as a courtesy check --
# TabICL will learn the marginal distribution regardless.

N_LAGS = 10  # Number of lag features

def build_lag_dataset(series_values, n_lags):
    """Create a supervised dataset from a time series using lag features."""
    X_rows = []
    y_rows = []
    for t in range(n_lags, len(series_values)):
        # Features: [val(t-1), val(t-2), ..., val(t-n_lags)]
        features = series_values[t - n_lags:t][::-1].tolist()
        target = series_values[t]
        X_rows.append(features)
        y_rows.append(target)
    X = np.array(X_rows)
    y = np.array(y_rows)
    return X, y

def build_prediction_input(series_values, n_lags):
    """Build the feature vector for the next unseen step."""
    last_n = series_values[-n_lags:][::-1].tolist()
    return np.array([last_n])

# Process each QS variable
all_predictions = {}

for name, df in datasets.items():
    col = name
    values = df[col].values

    print(f"  [{name}] Building lag-{N_LAGS} dataset...")
    X_train, y_train = build_lag_dataset(values, N_LAGS)
    X_pred = build_prediction_input(values, N_LAGS)

    print(f"    Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"    Prediction input (last {N_LAGS} values): {X_pred[0].tolist()}")

    # Get unique classes (the possible values this QS can take)
    unique_classes = sorted(np.unique(y_train))
    print(f"    Unique classes: {len(unique_classes)} ({min(unique_classes)}-{max(unique_classes)})")

    # Train TabICLClassifier
    print(f"    Training TabICLClassifier...")
    clf = TabICLClassifier(
        n_estimators=8,
        random_state=42,
        verbose=False,
    )
    clf.fit(X_train, y_train)
    print(f"    Model fitted.")

    # Predict probabilities
    proba = clf.predict_proba(X_pred)[0]  # shape: (n_classes,)
    classes = clf.classes_

    # Build probability map: value -> probability
    prob_map = {int(cls): float(prob) for cls, prob in zip(classes, proba)}

    # Sort by probability descending
    sorted_preds = sorted(prob_map.items(), key=lambda x: -x[1])
    top5 = sorted_preds[:5]

    all_predictions[name] = {
        'top5': top5,
        'full_distribution': prob_map,
        'last_values': X_pred[0].tolist(),
    }

    print(f"    Top 5 predictions for {name} on Apr 9, 2026:")
    for rank, (val, prob) in enumerate(top5, 1):
        print(f"      #{rank}: value={val:2d}  prob={prob:.4f} ({prob*100:.2f}%)")
    print()

# Also compute marginal (historical frequency) for comparison
print("-" * 70)
print("  BASELINE: Historical Frequency Distribution (for comparison)")
print("-" * 70)
baseline_top5 = {}
for name, df in datasets.items():
    col = name
    vc = df[col].value_counts(normalize=True).sort_values(ascending=False)
    top5 = [(int(val), float(prob)) for val, prob in vc.head(5).items()]
    baseline_top5[name] = top5
    print(f"  {name} historical top 5: ", end="")
    for val, prob in top5:
        print(f"{val}({prob:.3f})", end="  ")
    print()

print()

# ════════════════════════════════════════════════════════════════════════
# SECTION 3: SAVE RESULTS
# ════════════════════════════════════════════════════════════════════════

print("-" * 70)
print("SECTION 3: Saving Forecast Results")
print("-" * 70)

# ── Markdown report ──────────────────────────────────────────────────────
lines = []
lines.append("# Phase 2: TabICLv2 Next-Day Forecast (April 9, 2026)")
lines.append("")
lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"**Model**: TabICLClassifier v2.0.3 (CPU)")
lines.append(f"**Features**: {N_LAGS} lagged values")
lines.append(f"**Training**: Full history (11,762 observations)")
lines.append("")

lines.append("## Autocorrelation Gate")
lines.append("")
lines.append(f"95% confidence interval: +/- {ci_bound:.4f}")
lines.append("")
lines.append("| Variable | Max |ACF| (lags 1-30) | Significant Lags | Verdict |")
lines.append("|----------|---------------------|------------------|---------|")
for name in datasets:
    acf_vals = acf_results[name]
    max_abs = max(abs(acf_vals[lag]) for lag in range(1, n_lags + 1))
    sig_count = sum(1 for lag in range(1, n_lags + 1) if abs(acf_vals[lag]) > ci_bound)
    verdict = "i.i.d." if max_abs < 0.05 else "Weak autocorr"
    lines.append(f"| {name} | {max_abs:.4f} | {sig_count}/{n_lags} | {verdict} |")
lines.append("")

lines.append("## Top-5 Predicted Values per Variable")
lines.append("")
lines.append("### TabICLv2 Predictions")
lines.append("")

for name in datasets:
    pred = all_predictions[name]
    lines.append(f"#### {name}")
    lines.append("")
    lines.append(f"Last {N_LAGS} values (features): `{pred['last_values']}`")
    lines.append("")
    lines.append("| Rank | Value | Probability |")
    lines.append("|------|-------|-------------|")
    for rank, (val, prob) in enumerate(pred['top5'], 1):
        lines.append(f"| {rank} | **{val}** | {prob:.4f} ({prob*100:.2f}%) |")
    lines.append("")

lines.append("### Historical Frequency Baseline (for comparison)")
lines.append("")
lines.append("| Variable | #1 | #2 | #3 | #4 | #5 |")
lines.append("|----------|----|----|----|----|-----|")
for name in datasets:
    top5 = baseline_top5[name]
    cells = [f"{val} ({prob:.3f})" for val, prob in top5]
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
lines.append("")

lines.append("## Summary Comparison Table")
lines.append("")
lines.append("| Variable | TabICLv2 Top-5 | Historical Top-5 |")
lines.append("|----------|---------------|-----------------|")
for name in datasets:
    ticl_vals = [str(v) for v, _ in all_predictions[name]['top5']]
    hist_vals = [str(v) for v, _ in baseline_top5[name]]
    lines.append(f"| {name} | {', '.join(ticl_vals)} | {', '.join(hist_vals)} |")
lines.append("")

report_path = FORECAST_DIR / "next_day_predictions.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"  Saved: {report_path}")

# ── JSON export ──────────────────────────────────────────────────────────
export = {
    'prediction_date': '2026-04-09',
    'model': 'TabICLClassifier v2.0.3',
    'n_lags': N_LAGS,
    'generated': datetime.now().isoformat(),
    'predictions': {},
    'baseline_historical': {},
    'acf_gate': {},
}
for name in datasets:
    export['predictions'][name] = {
        'top5': [{'value': int(v), 'probability': round(float(p), 6)} for v, p in all_predictions[name]['top5']],
        'last_input_values': all_predictions[name]['last_values'],
    }
    export['baseline_historical'][name] = {
        'top5': [{'value': int(v), 'probability': round(float(p), 6)} for v, p in baseline_top5[name]]
    }
    acf_vals = acf_results[name]
    max_abs = max(abs(acf_vals[lag]) for lag in range(1, n_lags + 1))
    export['acf_gate'][name] = {
        'max_abs_acf': round(float(max_abs), 6),
        'ci_bound': round(float(ci_bound), 6),
        'verdict': 'iid' if max_abs < 0.05 else 'weak_autocorrelation',
    }

json_path = FORECAST_DIR / "predictions.json"
with open(json_path, 'w') as f:
    json.dump(export, f, indent=2)
print(f"  Saved: {json_path}")

# ── Prediction distribution plots ────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('TabICLv2 Predicted Probability Distribution for April 9, 2026', fontsize=14, y=0.98)

for i, name in enumerate(datasets):
    ax = axes[i // 2][i % 2]
    dist = all_predictions[name]['full_distribution']
    vals = sorted(dist.keys())
    probs = [dist[v] for v in vals]

    # Highlight top 5
    top5_vals = {v for v, _ in all_predictions[name]['top5']}
    bar_colors = [colors[i] if v in top5_vals else '#cccccc' for v in vals]

    ax.bar(vals, probs, color=bar_colors, edgecolor='white', linewidth=0.3)
    ax.set_title(f'{name} - Top 5 highlighted', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Predicted Probability')
    ax.set_xlim(0, 40)
    ax.grid(axis='y', alpha=0.3)

    # Annotate top 5
    for val, prob in all_predictions[name]['top5'][:3]:
        ax.annotate(f'{val}', (val, prob), textcoords="offset points",
                   xytext=(0, 5), ha='center', fontsize=8, fontweight='bold')

axes[2][1].axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.97])
pred_plot_path = FORECAST_DIR / "prediction_distributions.png"
plt.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {pred_plot_path.name}")

print()
print("=" * 70)
print("  Phase 2 COMPLETE")
print("=" * 70)
print(f"  Artifacts in: {FORECAST_DIR}")
for f_path in sorted(FORECAST_DIR.glob('*')):
    print(f"    {f_path.name:45s}  {f_path.stat().st_size:>8,} bytes")
print(f"  ACF plot: {acf_path}")
print()
print("  Ready for final commit and push.")
print("=" * 70)
