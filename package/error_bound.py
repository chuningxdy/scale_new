"""
Error bounds via subsampling: fit NQS on random 50% subsets of train data
and report variability in (1) inferred parameters and (2) test predictions.

Usage:
    python error_bound.py
"""

import csv
import json
import numpy as np
from nqs import fit_nqs, compute_nqs_regularized

DATASET_CSV = "dataset.csv"
N_SAMPLES = 100
SUBSAMPLE_FRAC = 0.5
SEED = 42
QUANTILE_LO = 5    # lower quantile (percent)
QUANTILE_HI = 95   # upper quantile (percent)

# ------------------------------------------------------------------ #
#  1. Load dataset                                                    #
# ------------------------------------------------------------------ #

train_h_dicts = []
train_losses = []
test_rows = []

with open(DATASET_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['split'] == 'train' and row['filtered'] != 'True':
            train_h_dicts.append({
                'N': int(row['N']),
                'K': int(row['K']),
                'B': int(row['B']),
            })
            train_losses.append(float(row['loss']))
        elif row['split'] == 'test':
            test_rows.append(row)

print(f"Train: {len(train_h_dicts)} runs (after filtering)")
print(f"Test:  {len(test_rows)} runs")
print(f"Subsampling: {N_SAMPLES} samples, each using {SUBSAMPLE_FRAC*100:.0f}% of train data")

# ------------------------------------------------------------------ #
#  2. Fit NQS on subsampled train data                                #
# ------------------------------------------------------------------ #

param_ranges = {
    'p': (1.05, 2.5),
    'q': (0.6, 2.5),
    'P': (10.0, 100.0),
    'Q': (0.05, 20.0),
    'e_irr': (1.0, 1.5),
    'R': (0.1**2, 10.0**2),
    'r': (0.6, 2.5),
}

init_weight_norm_squared_fn = lambda N: N * 0.02**2
param_names = ['p', 'q', 'P', 'Q', 'e_irr', 'R', 'r']

rng = np.random.RandomState(SEED)
n_train = len(train_h_dicts)
n_sub = int(n_train * SUBSAMPLE_FRAC)

all_fitted_params = []  # list of dicts
all_test_preds = []     # list of arrays (one per sample)
all_train_preds = []    # list of arrays (one per sample)
all_sampled_idx = []    # list of index arrays (one per sample)

for i in range(N_SAMPLES):
    print(f"\n--- Sample {i+1}/{N_SAMPLES} ---")
    idx = rng.choice(n_train, size=n_sub, replace=False)
    all_sampled_idx.append(set(idx.tolist()))
    sub_h = [train_h_dicts[j] for j in idx]
    sub_losses = [train_losses[j] for j in idx]

    fitted, eval_metric, _ = fit_nqs(
        h_dicts=sub_h,
        nn_losses=sub_losses,
        seed=SEED + i,
        number_of_initializations=2000,
        param_ranges=param_ranges,
        gtol=1e-8,
        max_steps=5000,
        loss='mse',
    )
    all_fitted_params.append(fitted)
    print(f"  Eval metric (RMSLE %): {eval_metric:.4f}")
    for k in param_names:
        print(f"  {k}: {fitted[k]:.6f}")

    # Predict on train set
    train_preds = []
    for h in train_h_dicts:
        risk = float(compute_nqs_regularized(fitted, h, init_weight_norm_squared_fn))
        train_preds.append(risk)
    all_train_preds.append(train_preds)

    # Predict on test set
    preds = []
    for row in test_rows:
        h = {'N': int(row['N']), 'K': int(row['K']), 'B': int(row['B'])}
        risk = float(compute_nqs_regularized(fitted, h, init_weight_norm_squared_fn))
        preds.append(risk)
    all_test_preds.append(preds)

# ------------------------------------------------------------------ #
#  3. Error bounds on inferred parameters                             #
# ------------------------------------------------------------------ #

print(f"\n{'='*60}")
print("Error Bounds on Inferred Parameters")
print(f"{'='*60}")
print(f"  {'Param':<8s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
print(f"  {'-'*52}")

for k in param_names:
    vals = [p[k] for p in all_fitted_params]
    print(f"  {k:<8s} {np.mean(vals):>12.6f} {np.std(vals):>12.6f} {np.min(vals):>12.6f} {np.max(vals):>12.6f}")

# ------------------------------------------------------------------ #
#  4. Error bounds on test predictions                                #
# ------------------------------------------------------------------ #

test_preds_arr = np.array(all_test_preds)  # (N_SAMPLES, n_test)
pred_mean = test_preds_arr.mean(axis=0)
pred_std = test_preds_arr.std(axis=0)
pred_qlo = np.percentile(test_preds_arr, QUANTILE_LO, axis=0)
pred_qhi = np.percentile(test_preds_arr, QUANTILE_HI, axis=0)

print(f"\n{'='*60}")
print("Error Bounds on Test Predictions")
print(f"{'='*60}")
print(f"  {'N':>10s} {'B':>8s} {'K':>8s} {'NN loss':>10s} {'Mean':>10s} {'Std':>10s} {'Q_lo':>10s} {'Q_hi':>10s}")
print(f"  {'-'*72}")

for j, row in enumerate(test_rows):
    nn_loss = float(row['loss'])
    print(f"  {row['N']:>10s} {row['B']:>8s} {row['K']:>8s} {nn_loss:>10.4f} "
          f"{pred_mean[j]:>10.4f} {pred_std[j]:>10.4f} {pred_qlo[j]:>10.4f} {pred_qhi[j]:>10.4f}")

# Summary statistics
print(f"\n  Avg relative std of test predictions: {np.mean(pred_std / pred_mean) * 100:.2f}%")
print(f"  Max relative std of test predictions: {np.max(pred_std / pred_mean) * 100:.2f}%")

# ------------------------------------------------------------------ #
#  5. Save results                                                    #
# ------------------------------------------------------------------ #

import os
OUTPUT_DIR = "error_bound"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output = {
    'config': {
        'n_samples': N_SAMPLES,
        'subsample_frac': SUBSAMPLE_FRAC,
        'seed': SEED,
    },
    'param_bounds': {
        k: {
            'mean': float(np.mean([p[k] for p in all_fitted_params])),
            'std': float(np.std([p[k] for p in all_fitted_params])),
            'min': float(np.min([p[k] for p in all_fitted_params])),
            'max': float(np.max([p[k] for p in all_fitted_params])),
        }
        for k in param_names
    },
    'trials': [
        {k: float(p[k]) for k in param_names}
        for p in all_fitted_params
    ],
}

PARAM_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "error_bounds.json")
with open(PARAM_OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nParam bounds saved to {PARAM_OUTPUT_FILE}")

train_preds_arr = np.array(all_train_preds)  # (N_SAMPLES, n_train)

PRED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "error_bounds_preds.csv")
with open(PRED_OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['split', 'N', 'B', 'K', 'nn_loss', 'pred_mean', 'pred_std', 'pred_qlo', 'pred_qhi']
    header += [f'trial_{s+1}_pred' for s in range(N_SAMPLES)]
    header += [f'trial_{s+1}_sampled' for s in range(N_SAMPLES)]
    writer.writerow(header)

    # Train rows
    for j in range(n_train):
        preds_j = train_preds_arr[:, j]
        csv_row = [
            'train',
            train_h_dicts[j]['N'], train_h_dicts[j]['B'], train_h_dicts[j]['K'],
            train_losses[j],
            float(np.mean(preds_j)), float(np.std(preds_j)),
            float(np.percentile(preds_j, QUANTILE_LO)), float(np.percentile(preds_j, QUANTILE_HI)),
        ]
        csv_row += [float(preds_j[s]) for s in range(N_SAMPLES)]
        csv_row += [int(j in all_sampled_idx[s]) for s in range(N_SAMPLES)]
        writer.writerow(csv_row)

    # Test rows
    for j, row in enumerate(test_rows):
        preds_j = test_preds_arr[:, j]
        csv_row = [
            'test',
            int(row['N']), int(row['B']), int(row['K']),
            float(row['loss']),
            float(pred_mean[j]), float(pred_std[j]),
            float(pred_qlo[j]), float(pred_qhi[j]),
        ]
        csv_row += [float(preds_j[s]) for s in range(N_SAMPLES)]
        csv_row += [''] * N_SAMPLES
        writer.writerow(csv_row)

print(f"Pred bounds saved to {PRED_OUTPUT_FILE}")
