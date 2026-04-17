"""
Analyze error bounds produced by error_bound.py.
  - Box plot of inferred parameters across trials
  - Actual vs predicted plot with error bars on test predictions

Usage:
    python analyze_error_bounds.py
"""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from nqs import compute_nqs_regularized

INPUT_DIR = "error_bound"
PARAM_FILE = f"{INPUT_DIR}/error_bounds.json"
PRED_FILE = f"{INPUT_DIR}/error_bounds_preds.csv"

# Full-dataset fitted parameters (from demo_fit_and_predict.py)
PRE_FITTED_NQS = {
    'p': 1.1169923126408567,
    'q': 0.5880068752176018,
    'P': 3.581383803474162,
    'Q': 0.9355347298617339,
    'e_irr': 0.44947765223894076,
    'R': 4.282677723527906,
    'r': 1.4850559677444017,
}
init_weight_norm_squared_fn = lambda N: N * 0.02**2

# ------------------------------------------------------------------ #
#  1. Load data                                                       #
# ------------------------------------------------------------------ #

with open(PARAM_FILE, 'r') as f:
    param_data = json.load(f)

trials = param_data['trials']
param_names = list(param_data['param_bounds'].keys())
n_trials = len(trials)

pred_rows = []
with open(PRED_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pred_rows.append(row)

train_rows = [r for r in pred_rows if r['split'] == 'train']
test_rows = [r for r in pred_rows if r['split'] == 'test']

# ------------------------------------------------------------------ #
#  2. Box plot of inferred parameters                                 #
# ------------------------------------------------------------------ #

fig, axes = plt.subplots(1, len(param_names), figsize=(2.0 * len(param_names), 3.0))

LOG_SCALE_PARAMS = {'P', 'Q', 'e_irr', 'R'}

for ax, k in zip(axes, param_names):
    values = [t[k] for t in trials]
    ax.boxplot(values, widths=0.5)
    ax.scatter(np.ones(n_trials), values, color='blue', alpha=0.7, s=30, zorder=3)
    ax.scatter(1, PRE_FITTED_NQS[k], color='red', s=50, zorder=4, marker='D')
    if k in LOG_SCALE_PARAMS:
        ax.set_yscale('log')
    ax.set_title(k, fontweight='bold')
    ax.set_xticks([])

fig.suptitle('Parameter Estimates Across Trials', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{INPUT_DIR}/param_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {INPUT_DIR}/param_boxplot.png")

# ------------------------------------------------------------------ #
#  3. Actual vs predicted with error bars                             #
# ------------------------------------------------------------------ #

# Compute predictions using full-dataset fit
train_nn = [float(r['nn_loss']) for r in train_rows]
train_nqs = []
for r in train_rows:
    h = {'N': int(r['N']), 'K': int(r['K']), 'B': int(r['B'])}
    train_nqs.append(float(compute_nqs_regularized(PRE_FITTED_NQS, h, init_weight_norm_squared_fn)))

test_nn = [float(r['nn_loss']) for r in test_rows]
test_nqs = []
for r in test_rows:
    h = {'N': int(r['N']), 'K': int(r['K']), 'B': int(r['B'])}
    test_nqs.append(float(compute_nqs_regularized(PRE_FITTED_NQS, h, init_weight_norm_squared_fn)))

test_pred_mean = [float(r['pred_mean']) for r in test_rows]
test_pred_qlo = [float(r['pred_qlo']) for r in test_rows]
test_pred_qhi = [float(r['pred_qhi']) for r in test_rows]

# Error bars: asymmetric around subsampled mean (5th-95th percentile)
test_err_lo = [m - lo for m, lo in zip(test_pred_mean, test_pred_qlo)]
test_err_hi = [hi - m for m, hi in zip(test_pred_mean, test_pred_qhi)]

fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(train_nn, train_nqs, c='blue', label='train (full fit)', alpha=0.3, s=10)
ax.scatter(test_nn, test_nqs, c='red', label='test (full fit)', alpha=0.3, s=10)
ax.errorbar(test_nn, test_pred_mean, yerr=[test_err_lo, test_err_hi],
            fmt='none', ecolor='red', alpha=0.5,
            capsize=2, capthick=1, elinewidth=1, label='test (90% CI)')

# Diagonal reference line
all_vals = train_nn + test_nn + train_nqs + test_nqs
lo, hi = min(all_vals), max(all_vals)
ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('NN loss (ground truth)')
ax.set_ylabel('NQS risk (predicted)')
ax.set_title('Actual vs Predicted', fontweight='bold')
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{INPUT_DIR}/actual_vs_predicted_with_bounds.png', dpi=150)
plt.close()
print(f"Saved {INPUT_DIR}/actual_vs_predicted_with_bounds.png")

# ------------------------------------------------------------------ #
#  4. IsoFLOP / IsoToken curves with error bound shading              #
# ------------------------------------------------------------------ #

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Load full dataset for type/C info, and build a lookup from (split,N,B,K) to CSV row
DATASET_CSV = "dataset.csv"
all_rows = []
with open(DATASET_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['split'] == 'train' and row.get('filtered', '') == 'True':
            continue
        all_rows.append(row)

# Build lookup: (split, N, B, K) -> pred CSV row
pred_lookup = {}
for r in pred_rows:
    key = (r['split'], int(r['N']), int(r['B']), int(r['K']))
    pred_lookup[key] = r

# Build results with full-fit predictions + per-trial predictions
results = []
for row in all_rows:
    split = row['split']
    h = {'N': int(row['N']), 'K': int(row['K']), 'B': int(row['B'])}
    nqs_risk = float(compute_nqs_regularized(PRE_FITTED_NQS, h, init_weight_norm_squared_fn))

    key = (split, h['N'], h['B'], h['K'])
    pr = pred_lookup.get(key)

    results.append({
        'nn_loss': float(row['loss']),
        'nqs_risk': nqs_risk,
        'pred_qlo': float(pr['pred_qlo']) if pr else nqs_risk,
        'pred_qhi': float(pr['pred_qhi']) if pr else nqs_risk,
        'split': split,
        'type': row['type'],
        'N': h['N'],
        'B': h['B'],
        'K': h['K'],
        'C': int(row['C']),
    })

TARGET_PF = [0.92, 4, 15, 59, 236, 944, 3780, 15100, 60500]
TARGET_C = [pf * 1e6 for pf in TARGET_PF]

def is_target_C(c_val):
    return any(abs(c_val - t) / t < 0.10 for t in TARGET_C)

def format_C(c):
    c_pf = c / 1e6
    if c_pf >= 1:
        return f"  {int(round(c_pf))} PF"
    return f"  {c_pf:.2f} PF"

color_map = {'train': 'blue', 'test': 'red'}

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.5, 3.0))

# --- IsoFLOP: loss vs N for each C ---
isoflop_res = [r for r in results if r['type'] == 'isoflop']
isoflop_Cs = sorted(c for c in set(r['C'] for r in isoflop_res) if is_target_C(c))
isoflop_label_Cs = set()
for split in ['train', 'test']:
    split_Cs = [c for c in isoflop_Cs if any(r['C'] == c and r['split'] == split for r in isoflop_res)]
    if split_Cs:
        isoflop_label_Cs.add(max(split_Cs))

for c_val in isoflop_Cs:
    group = sorted([r for r in isoflop_res if r['C'] == c_val], key=lambda r: r['N'])
    Ns = [r['N'] for r in group]
    nn = [r['nn_loss'] for r in group]
    nqs = [r['nqs_risk'] for r in group]
    lo = [r['pred_qlo'] for r in group]
    hi = [r['pred_qhi'] for r in group]
    color = color_map[group[0]['split']]

    ax_left.scatter(Ns, nn, color=color, marker='o', s=10, alpha=0.4, ec='none')
    ax_left.plot(Ns, nqs, color=color, linestyle='-', linewidth=0.8, alpha=0.5)
    ax_left.fill_between(Ns, lo, hi, color='pink', alpha=0.4)

    if c_val in isoflop_label_Cs:
        label = format_C(c_val)
        ax_left.text(Ns[-1] * 1.05, nqs[-1], label, color=color, fontsize=8, fontweight='bold', va='center')

ax_left.set_xscale("log")
ax_left.set_yscale("log")
ax_left.set_xlabel("Parameters Count")
ax_left.set_ylabel("Loss")
ax_left.set_title("IsoFLOP Curves", fontweight='bold')
x_min, x_max = ax_left.get_xlim()
ax_left.set_xlim(x_min * 0.5, x_max * 2.5)

# --- IsoToken: loss vs B for each C ---
isotoken_res = [r for r in results if r['type'] == 'isotoken']
isotoken_Cs = sorted(c for c in set(r['C'] for r in isotoken_res) if is_target_C(c))
isotoken_label_Cs = set()
for split in ['train', 'test']:
    split_Cs = [c for c in isotoken_Cs if any(r['C'] == c and r['split'] == split for r in isotoken_res)]
    if split_Cs:
        isotoken_label_Cs.add(max(split_Cs))

for c_val in isotoken_Cs:
    group = sorted([r for r in isotoken_res if r['C'] == c_val], key=lambda r: r['B'])
    Bs = [r['B'] for r in group]
    nn = [r['nn_loss'] for r in group]
    nqs = [r['nqs_risk'] for r in group]
    lo = [r['pred_qlo'] for r in group]
    hi = [r['pred_qhi'] for r in group]
    color = color_map[group[0]['split']]

    ax_right.scatter(Bs, nn, color=color, marker='o', s=10, alpha=0.4, ec='none')
    ax_right.plot(Bs, nqs, color=color, linestyle='-', linewidth=0.8, alpha=0.5)
    ax_right.fill_between(Bs, lo, hi, color='pink', alpha=0.4)

    if c_val in isotoken_label_Cs:
        label = format_C(c_val)
        ax_right.text(Bs[-1] * 1.05, nqs[-1], label, color=color, fontsize=8, fontweight='bold', va='center')

ax_right.set_xscale("log")
ax_right.set_yscale("log")
ax_right.set_xlabel("Batch Size")
ax_right.set_ylabel("Loss")
ax_right.set_title("IsoToken Curves", fontweight='bold')
x_min, x_max = ax_right.get_xlim()
ax_right.set_xlim(x_min * 0.5, x_max * 2.5)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label='NN (ground truth)'),
    Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='NQS (full fit)'),
    Patch(facecolor='pink', alpha=0.4, label='90% CI'),
    Line2D([0], [0], color='blue', linestyle='-', linewidth=4, label='Train'),
    Line2D([0], [0], color='red', linestyle='-', linewidth=4, label='Test'),
]
fig.legend(handles=legend_elements, fontsize=9, loc='lower center',
           bbox_to_anchor=(0.5, -0.08), ncol=5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{INPUT_DIR}/isoflop_isotoken_with_bounds.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {INPUT_DIR}/isoflop_isotoken_with_bounds.png")
