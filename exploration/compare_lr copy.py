"""
Compare eigenvalues and validation loss across learning rates.
Analyzes outputs from Hydra pipeline runs with batch_size=256.
"""

import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# ============== CONFIGURATION ==============
OUTPUT_DIR = "outputs/run_pipeline"
TARGET_BS = 256
TARGET_SCHEDULE = "linear"  # "constant", "linear", etc.
TARGET_STEPS = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
# Eigenvalue indices and their sources: (index, source)
# source can be "lanczos" or "slq"
EIGENVALUE_CONFIG = [
    (1, "lanczos"),  # λ_1 from Lanczos
    (8, "slq"),      # λ_8 from SLQ
    (64, "slq"),     # λ_64 from SLQ
]

# ============== HELPER FUNCTIONS ==============

def parse_folder_name(folder_name):
    """Extract batch_size, learning_rate, and schedule from folder name like 'bs256_lr0.0001_constant'."""
    match = re.match(r'bs(\d+)_lr([\d.e-]+)_(\w+)', folder_name)
    if match:
        bs = int(match.group(1))
        lr = float(match.group(2))
        schedule = match.group(3)
        return bs, lr, schedule
    # Fallback for old format without schedule
    match = re.match(r'bs(\d+)_lr([\d.e-]+)', folder_name)
    if match:
        bs = int(match.group(1))
        lr = float(match.group(2))
        return bs, lr, None
    return None, None, None


def slq_interp_eigenvalue(evs, wts, total_params, target_index):
    """Interpolate eigenvalue at target_index using log-log interpolation (power-law aware)."""
    if len(evs) == 0 or len(wts) == 0:
        return np.nan

    evs = np.array(evs)
    wts = np.array(wts)

    # Sort by eigenvalue descending
    idx = np.argsort(evs)[::-1]
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]

    # Filter positive eigenvalues (needed for log)
    mask = evs_sorted > 1e-10
    evs_pos = evs_sorted[mask]
    wts_pos = wts_sorted[mask]

    if len(evs_pos) < 2:
        return np.nan

    # Compute effective indices (cumulative spectral mass)
    indices = np.cumsum(wts_pos) * total_params

    # Check if target_index is in range
    if target_index < indices[0] or target_index > indices[-1]:
        return np.nan

    # Log-log interpolation (power-law aware)
    log_indices = np.log(indices)
    log_evs = np.log(evs_pos)

    log_target = np.log(target_index)
    log_result = np.interp(log_target, log_indices, log_evs)

    return np.exp(log_result)


def get_eigenvalue_from_hessian(hessian_data, eigenvalue_index, source="slq"):
    """Extract eigenvalue at given index (1-indexed) from specified source."""
    if source == "lanczos":
        lanczos_evs = hessian_data.get("lanczos", {}).get("eigenvalues", [])
        if not lanczos_evs:
            return np.nan
        sorted_evs = sorted(lanczos_evs, reverse=True)
        idx = eigenvalue_index - 1  # Convert to 0-indexed
        if idx < len(sorted_evs):
            return sorted_evs[idx]
        return np.nan
    else:  # slq
        slq_evs = hessian_data.get("slq", {}).get("raw_eigenvalues", [])
        slq_wts = hessian_data.get("slq", {}).get("raw_weights", [])
        total_params = hessian_data.get("config", {}).get("total_params", None)

        if not slq_evs or not slq_wts or total_params is None:
            return np.nan

        return slq_interp_eigenvalue(slq_evs, slq_wts, total_params, eigenvalue_index)


def get_val_loss_at_step(loss_history, target_step):
    """Get validation loss closest to target_step."""
    eval_losses = loss_history.get("eval_losses", [])

    if not eval_losses:
        return np.nan

    # Extract steps and losses from list of dicts
    eval_steps = np.array([entry["step"] for entry in eval_losses])
    eval_loss_vals = np.array([entry["loss"] for entry in eval_losses])

    # Find closest step
    idx = np.argmin(np.abs(eval_steps - target_step))

    # Only return if reasonably close (within 100 steps)
    if abs(eval_steps[idx] - target_step) <= 100:
        return eval_loss_vals[idx]
    return np.nan


# ============== COLLECT DATA ==============
print(f"Scanning {OUTPUT_DIR} for bs={TARGET_BS}, schedule={TARGET_SCHEDULE} runs...")

# Find all run folders
run_folders = glob.glob(os.path.join(OUTPUT_DIR, "bs*_lr*"))

# Filter for target batch size and schedule, collect data
# {lr: {"eigenvalues": {ev_idx: {step: val}}, "val_loss": {step: val}}}
data = {}

for folder in sorted(run_folders):
    folder_name = os.path.basename(folder)
    bs, lr, schedule = parse_folder_name(folder_name)

    if bs != TARGET_BS:
        continue
    if TARGET_SCHEDULE is not None and schedule != TARGET_SCHEDULE:
        continue

    print(f"  Processing {folder_name} (lr={lr}, schedule={schedule})")

    data[lr] = {"eigenvalues": {ev_idx: {} for ev_idx, _ in EIGENVALUE_CONFIG}, "val_loss": {}}

    # Load eigenvalues for each target step
    for step in TARGET_STEPS:
        hessian_file = os.path.join(folder, f"hessian_step_{step}.json")
        if os.path.exists(hessian_file):
            with open(hessian_file, "r") as f:
                hessian_data = json.load(f)
            for ev_idx, source in EIGENVALUE_CONFIG:
                ev = get_eigenvalue_from_hessian(hessian_data, ev_idx, source)
                data[lr]["eigenvalues"][ev_idx][step] = ev
        else:
            for ev_idx, _ in EIGENVALUE_CONFIG:
                data[lr]["eigenvalues"][ev_idx][step] = np.nan

    # Load validation loss
    loss_file = os.path.join(folder, "loss_history.json")
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            loss_history = json.load(f)
        for step in TARGET_STEPS:
            val_loss = get_val_loss_at_step(loss_history, step)
            data[lr]["val_loss"][step] = val_loss
    else:
        for step in TARGET_STEPS:
            data[lr]["val_loss"][step] = np.nan

if not data:
    print(f"No data found for bs={TARGET_BS} in {OUTPUT_DIR}")
    exit(1)

# Sort learning rates
lrs = sorted(data.keys())
print(f"\nFound {len(lrs)} learning rates: {lrs}")

# ============== PLOT ==============
num_ev_cols = len(EIGENVALUE_CONFIG)
fig, axes = plt.subplots(3, num_ev_cols + 1, figsize=(6 * (num_ev_cols + 1), 15))

# Color map for steps (row 1)
colors_steps = plt.cm.viridis(np.linspace(0.2, 0.8, len(TARGET_STEPS)))
# Color map for learning rates (row 2, 3)
colors_lrs = plt.cm.plasma(np.linspace(0.2, 0.8, len(lrs)))

# ===== ROW 1: x-axis = LR, series = steps =====
# Plot eigenvalues for each index
for panel_idx, (ev_idx, source) in enumerate(EIGENVALUE_CONFIG):
    ax = axes[0, panel_idx]
    for i, step in enumerate(TARGET_STEPS):
        evs = [data[lr]["eigenvalues"][ev_idx].get(step, np.nan) for lr in lrs]
        ax.plot(lrs, evs, 'o-', color=colors_steps[i], label=f'Step {step}',
                linewidth=2, markersize=8)

    # Add reference line λ = 2/LR for λ_1
    if ev_idx == 1:
        lr_range = np.logspace(np.log10(min(lrs)), np.log10(max(lrs)), 100)
        ref_line = 2.0 / lr_range
        ax.plot(lr_range, ref_line, 'k--', linewidth=1.5, label=r'$2/\eta$', zorder=0)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Learning Rate", fontsize=12)
    source_label = "Lanczos" if source == "lanczos" else "SLQ"
    ax.set_ylabel(f"$\\lambda_{{{ev_idx}}}$ ({source_label})", fontsize=12)
    ax.set_title(f"$\\lambda_{{{ev_idx}}}$ vs Learning Rate (bs={TARGET_BS})", fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

# Plot validation loss (last panel, row 1)
ax_loss = axes[0, -1]
for i, step in enumerate(TARGET_STEPS):
    losses = [data[lr]["val_loss"].get(step, np.nan) for lr in lrs]
    ax_loss.plot(lrs, losses, 'o-', color=colors_steps[i], label=f'Step {step}',
                 linewidth=2, markersize=8)

ax_loss.set_xscale('log')
ax_loss.set_yscale('log')
ax_loss.set_xlabel("Learning Rate", fontsize=12)
ax_loss.set_ylabel("Validation Loss", fontsize=12)
ax_loss.set_title(f"Validation Loss vs Learning Rate (bs={TARGET_BS})", fontsize=14)
ax_loss.legend(loc='best', fontsize=9)
ax_loss.grid(True, alpha=0.3, which='both')

# ===== ROW 2: x-axis = steps, series = LR =====
# Plot eigenvalues for each index
for panel_idx, (ev_idx, source) in enumerate(EIGENVALUE_CONFIG):
    ax = axes[1, panel_idx]
    for i, lr in enumerate(lrs):
        evs = [data[lr]["eigenvalues"][ev_idx].get(step, np.nan) for step in TARGET_STEPS]
        ax.plot(TARGET_STEPS, evs, 'o-', color=colors_lrs[i], label=f'LR={lr:.0e}',
                linewidth=2, markersize=8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Training Step", fontsize=12)
    source_label = "Lanczos" if source == "lanczos" else "SLQ"
    ax.set_ylabel(f"$\\lambda_{{{ev_idx}}}$ ({source_label})", fontsize=12)
    ax.set_title(f"$\\lambda_{{{ev_idx}}}$ vs Step (bs={TARGET_BS})", fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

# Plot validation loss (last panel, row 2)
ax_loss2 = axes[1, -1]
for i, lr in enumerate(lrs):
    losses = [data[lr]["val_loss"].get(step, np.nan) for step in TARGET_STEPS]
    ax_loss2.plot(TARGET_STEPS, losses, 'o-', color=colors_lrs[i], label=f'LR={lr:.0e}',
                  linewidth=2, markersize=8)

ax_loss2.set_xscale('log')
ax_loss2.set_yscale('log')
ax_loss2.set_xlabel("Training Step", fontsize=12)
ax_loss2.set_ylabel("Validation Loss", fontsize=12)
ax_loss2.set_title(f"Validation Loss vs Step (bs={TARGET_BS})", fontsize=14)
ax_loss2.legend(loc='best', fontsize=9)
ax_loss2.grid(True, alpha=0.3, which='both')

# ===== ROW 3: x-axis = lr * steps, series = LR =====
# Plot eigenvalues for each index
for panel_idx, (ev_idx, source) in enumerate(EIGENVALUE_CONFIG):
    ax = axes[2, panel_idx]
    for i, lr in enumerate(lrs):
        x_vals = [lr * step for step in TARGET_STEPS]
        evs = [data[lr]["eigenvalues"][ev_idx].get(step, np.nan) for step in TARGET_STEPS]
        ax.plot(x_vals, evs, 'o-', color=colors_lrs[i], label=f'LR={lr:.0e}',
                linewidth=2, markersize=8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("LR $\\times$ Step", fontsize=12)
    source_label = "Lanczos" if source == "lanczos" else "SLQ"
    ax.set_ylabel(f"$\\lambda_{{{ev_idx}}}$ ({source_label})", fontsize=12)
    ax.set_title(f"$\\lambda_{{{ev_idx}}}$ vs LR$\\times$Step (bs={TARGET_BS})", fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

# Plot validation loss (last panel, row 3)
ax_loss3 = axes[2, -1]
for i, lr in enumerate(lrs):
    x_vals = [lr * step for step in TARGET_STEPS]
    losses = [data[lr]["val_loss"].get(step, np.nan) for step in TARGET_STEPS]
    ax_loss3.plot(x_vals, losses, 'o-', color=colors_lrs[i], label=f'LR={lr:.0e}',
                  linewidth=2, markersize=8)

ax_loss3.set_xscale('log')
ax_loss3.set_yscale('log')
ax_loss3.set_xlabel("LR $\\times$ Step", fontsize=12)
ax_loss3.set_ylabel("Validation Loss", fontsize=12)
ax_loss3.set_title(f"Validation Loss vs LR$\\times$Step (bs={TARGET_BS})", fontsize=14)
ax_loss3.legend(loc='best', fontsize=9)
ax_loss3.grid(True, alpha=0.3, which='both')

plt.tight_layout()

# Save
file_name = f"compare_lr_bs{TARGET_BS}_{TARGET_SCHEDULE}.png"
output_path = os.path.join(OUTPUT_DIR, file_name)
plt.savefig(output_path, dpi=150)
print(f"\nSaved plot to {output_path}")

plt.show()

# ============== PRINT SUMMARY ==============
for ev_idx, source in EIGENVALUE_CONFIG:
    source_label = "Lanczos" if source == "lanczos" else "SLQ"
    print("\n" + "="*70)
    print(f"SUMMARY: λ_{ev_idx} ({source_label}) for bs={TARGET_BS}")
    print("="*70)
    print(f"{'LR':<15}", end="")
    for step in TARGET_STEPS:
        print(f"{'Step '+str(step):<15}", end="")
    print()
    print("-"*70)
    for lr in lrs:
        print(f"{lr:<15.1e}", end="")
        for step in TARGET_STEPS:
            val = data[lr]["eigenvalues"][ev_idx].get(step, np.nan)
            if np.isnan(val):
                print(f"{'N/A':<15}", end="")
            else:
                print(f"{val:<15.4f}", end="")
        print()

print("\n" + "="*70)
print(f"SUMMARY: Validation Loss for bs={TARGET_BS}")
print("="*70)
print(f"{'LR':<15}", end="")
for step in TARGET_STEPS:
    print(f"{'Step '+str(step):<15}", end="")
print()
print("-"*70)
for lr in lrs:
    print(f"{lr:<15.1e}", end="")
    for step in TARGET_STEPS:
        val = data[lr]["val_loss"].get(step, np.nan)
        if np.isnan(val):
            print(f"{'N/A':<15}", end="")
        else:
            print(f"{val:<15.4f}", end="")
    print()
