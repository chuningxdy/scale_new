"""
Error bounds via loss perturbation: fit NQS on the full train set but with
Gaussian noise added to each training loss, and report variability in
(1) inferred parameters and (2) test predictions.

Usage:
    python error_bound_perturb.py
"""

import csv
import json
import os
import numpy as np
from nqs import fit_nqs, compute_nqs_regularized
from compute_loss_variance import compute_loss_mean_and_variance

import jax
jax.config.update("jax_enable_x64", True)

DATASET_CSV = "dataset.csv"
N_SAMPLES = 100
SEED = 42
QUANTILE_LO = 5     # lower quantile (percent)
QUANTILE_HI = 95    # upper quantile (percent)

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

n_train = len(train_h_dicts)
print(f"Train: {n_train} runs (after filtering)")
print(f"Test:  {len(test_rows)} runs")
print(f"Perturbation: {N_SAMPLES} trials, per-point analytical std")

# ------------------------------------------------------------------ #
#  2. Fit NQS on perturbed train losses                               #
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

# Full-dataset fitted parameters (for computing analytical variance)
NQS_PARAMS = {
    'p': 1.1169923126408567,
    'q': 0.5880068752176018,
    'P': 3.581383803474162,
    'Q': 0.9355347298617339,
    'e_irr': 0.44947765223894076,
    'R': 4.282677723527906,
    'r': 1.4850559677444017,
}

# Compute per-point analytical std (cached to disk)
STD_CACHE_FILE = os.path.join("error_bound_perturb", "analytical_stds.npy")
os.makedirs("error_bound_perturb", exist_ok=True)

if os.path.exists(STD_CACHE_FILE):
    train_stds_arr = np.load(STD_CACHE_FILE)
    print(f"Loaded cached analytical stds from {STD_CACHE_FILE}")
    if len(train_stds_arr) != n_train:
        print(f"  Cache size mismatch ({len(train_stds_arr)} vs {n_train}), recomputing...")
        train_stds_arr = None
else:
    train_stds_arr = None

if train_stds_arr is None:
    print("Computing per-point analytical variance...")
    train_stds = []
    for h in train_h_dicts:
        _, var = compute_loss_mean_and_variance(
            h['N'], h['B'], h['K'],
            NQS_PARAMS['p'], NQS_PARAMS['q'], NQS_PARAMS['P'], NQS_PARAMS['Q'],
            NQS_PARAMS['e_irr'], NQS_PARAMS['R'], NQS_PARAMS['r'],
        )
        train_stds.append(float(np.sqrt(var)))
    train_stds_arr = np.array(train_stds)
    np.save(STD_CACHE_FILE, train_stds_arr)
    print(f"  Saved analytical stds to {STD_CACHE_FILE}")

print(f"  Std range: [{train_stds_arr.min():.6f}, {train_stds_arr.max():.6f}]")

rng = np.random.RandomState(SEED)
train_losses_arr = np.array(train_losses)

all_fitted_params = []
all_test_preds = []
all_train_preds = []

for i in range(N_SAMPLES):
    print(f"\n--- Trial {i+1}/{N_SAMPLES} ---")
    noise = rng.normal(0, 1.0, size=n_train) * train_stds_arr
    perturbed_losses = (train_losses_arr + noise).tolist()

    fitted, eval_metric, _ = fit_nqs(
        h_dicts=train_h_dicts,
        nn_losses=perturbed_losses,
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
print("Error Bounds on Inferred Parameters (Perturbation)")
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
print("Error Bounds on Test Predictions (Perturbation)")
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

OUTPUT_DIR = "error_bound_perturb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

output = {
    'config': {
        'n_samples': N_SAMPLES,
        'perturb_std': 'analytical (per-point)',
        'seed': SEED,
        'quantile_lo': QUANTILE_LO,
        'quantile_hi': QUANTILE_HI,
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
        writer.writerow(csv_row)

print(f"Pred bounds saved to {PRED_OUTPUT_FILE}")
