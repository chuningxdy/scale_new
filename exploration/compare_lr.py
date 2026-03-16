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
TARGET_BS_LIST = [1024, 4096]  # List of batch sizes to compare
BS_LINESTYLES = {1024: '-', 4096: ':'}  # solid for 256, dotted for 1024
TARGET_SCHEDULE = "constant"  # "constant", "linear", etc. Set to None for all schedules
TARGET_LRS = [3e-5, 1e-4, 3e-4, 6e-4, 1e-3] # None  # List of learning rates to plot, e.g. [1e-4, 3e-4, 1e-3]. Set to None for all
TARGET_STEPS = [0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
# Eigenvalue indices and their sources: (index, source)
# source can be "lanczos" or "slq"
EIGENVALUE_CONFIG = [
    (1, "lanczos"),  # λ_1 from Lanczos
    (16, "slq"),      # λ_8 from SLQ
    (128, "slq"),     # λ_64 from SLQ
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
print(f"Scanning {OUTPUT_DIR} for bs={TARGET_BS_LIST}, schedule={TARGET_SCHEDULE}, lrs={TARGET_LRS} ...")

# Find all run folders
run_folders = glob.glob(os.path.join(OUTPUT_DIR, "bs*_lr*"))

# Filter for target batch sizes, schedule, and learning rates, collect data
# {bs: {lr: {"eigenvalues": {ev_idx: {step: val}}, "val_loss": {step: val}}}}
data = {bs: {} for bs in TARGET_BS_LIST}

def lr_matches(lr, target_lrs):
    """Check if lr matches any in target_lrs (with tolerance for float comparison)."""
    if target_lrs is None:
        return True
    for target_lr in target_lrs:
        if abs(lr - target_lr) / max(abs(target_lr), 1e-10) < 0.01:  # 1% tolerance
            return True
    return False

for folder in sorted(run_folders):
    folder_name = os.path.basename(folder)
    bs, lr, schedule = parse_folder_name(folder_name)

    if bs not in TARGET_BS_LIST:
        continue
    if TARGET_SCHEDULE is not None and schedule != TARGET_SCHEDULE:
        continue
    if not lr_matches(lr, TARGET_LRS):
        continue

    print(f"  Processing {folder_name} (bs={bs}, lr={lr}, schedule={schedule})")

    data[bs][lr] = {"eigenvalues": {ev_idx: {} for ev_idx, _ in EIGENVALUE_CONFIG}, "val_loss": {}}

    # Load eigenvalues for each target step
    for step in TARGET_STEPS:
        hessian_file = os.path.join(folder, f"hessian_step_{step}.json")
        if os.path.exists(hessian_file):
            with open(hessian_file, "r") as f:
                hessian_data = json.load(f)
            for ev_idx, source in EIGENVALUE_CONFIG:
                ev = get_eigenvalue_from_hessian(hessian_data, ev_idx, source)
                data[bs][lr]["eigenvalues"][ev_idx][step] = ev
        else:
            for ev_idx, _ in EIGENVALUE_CONFIG:
                data[bs][lr]["eigenvalues"][ev_idx][step] = np.nan

    # Load validation loss
    loss_file = os.path.join(folder, "loss_history.json")
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            loss_history = json.load(f)
        for step in TARGET_STEPS:
            val_loss = get_val_loss_at_step(loss_history, step)
            data[bs][lr]["val_loss"][step] = val_loss
    else:
        for step in TARGET_STEPS:
            data[bs][lr]["val_loss"][step] = np.nan

# Get union of all learning rates across batch sizes
all_lrs = set()
for bs in TARGET_BS_LIST:
    all_lrs.update(data[bs].keys())
lrs = sorted(all_lrs)

if not lrs:
    print(f"No data found for bs={TARGET_BS_LIST} in {OUTPUT_DIR}")
    exit(1)

print(f"\nFound {len(lrs)} learning rates: {lrs}")
for bs in TARGET_BS_LIST:
    print(f"  bs={bs}: {sorted(data[bs].keys())}")

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
        for bs in TARGET_BS_LIST:
            bs_lrs = sorted(data[bs].keys())
            evs = [data[bs][lr]["eigenvalues"][ev_idx].get(step, np.nan) for lr in bs_lrs]
            linestyle = BS_LINESTYLES.get(bs, '-')
            label = f'Step {step}' if bs == TARGET_BS_LIST[0] else None  # Only label first bs
            ax.plot(bs_lrs, evs, linestyle, color=colors_steps[i], label=label,
                    linewidth=2, markersize=6, marker='o' if bs == TARGET_BS_LIST[0] else 's')

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
    bs_str = ','.join(str(b) for b in TARGET_BS_LIST)
    ax.set_title(f"$\\lambda_{{{ev_idx}}}$ vs LR (bs={bs_str})", fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

# Plot validation loss (last panel, row 1)
ax_loss = axes[0, -1]
for i, step in enumerate(TARGET_STEPS):
    for bs in TARGET_BS_LIST:
        bs_lrs = sorted(data[bs].keys())
        losses = [data[bs][lr]["val_loss"].get(step, np.nan) for lr in bs_lrs]
        linestyle = BS_LINESTYLES.get(bs, '-')
        label = f'Step {step}' if bs == TARGET_BS_LIST[0] else None
        ax_loss.plot(bs_lrs, losses, linestyle, color=colors_steps[i], label=label,
                     linewidth=2, markersize=6, marker='o' if bs == TARGET_BS_LIST[0] else 's')

ax_loss.set_xscale('log')
ax_loss.set_yscale('log')
ax_loss.set_xlabel("Learning Rate", fontsize=12)
ax_loss.set_ylabel("Validation Loss", fontsize=12)
bs_str = ','.join(str(b) for b in TARGET_BS_LIST)
ax_loss.set_title(f"Val Loss vs LR (bs={bs_str})", fontsize=14)
ax_loss.legend(loc='best', fontsize=9)
ax_loss.grid(True, alpha=0.3, which='both')

# ===== ROW 2: x-axis = steps, series = LR =====
# Plot eigenvalues for each index
for panel_idx, (ev_idx, source) in enumerate(EIGENVALUE_CONFIG):
    ax = axes[1, panel_idx]
    for i, lr in enumerate(lrs):
        for bs in TARGET_BS_LIST:
            if lr not in data[bs]:
                continue
            evs = [data[bs][lr]["eigenvalues"][ev_idx].get(step, np.nan) for step in TARGET_STEPS]
            linestyle = BS_LINESTYLES.get(bs, '-')
            label = f'LR={lr:.0e}' if bs == TARGET_BS_LIST[0] else None
            ax.plot(TARGET_STEPS, evs, linestyle, color=colors_lrs[i], label=label,
                    linewidth=2, markersize=6, marker='o' if bs == TARGET_BS_LIST[0] else 's')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Training Step", fontsize=12)
    source_label = "Lanczos" if source == "lanczos" else "SLQ"
    ax.set_ylabel(f"$\\lambda_{{{ev_idx}}}$ ({source_label})", fontsize=12)
    bs_str = ','.join(str(b) for b in TARGET_BS_LIST)
    ax.set_title(f"$\\lambda_{{{ev_idx}}}$ vs Step (bs={bs_str})", fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

# Plot validation loss (last panel, row 2)
ax_loss2 = axes[1, -1]
for i, lr in enumerate(lrs):
    for bs in TARGET_BS_LIST:
        if lr not in data[bs]:
            continue
        losses = [data[bs][lr]["val_loss"].get(step, np.nan) for step in TARGET_STEPS]
        linestyle = BS_LINESTYLES.get(bs, '-')
        label = f'LR={lr:.0e}' if bs == TARGET_BS_LIST[0] else None
        ax_loss2.plot(TARGET_STEPS, losses, linestyle, color=colors_lrs[i], label=label,
                      linewidth=2, markersize=6, marker='o' if bs == TARGET_BS_LIST[0] else 's')

ax_loss2.set_xscale('log')
ax_loss2.set_yscale('log')
ax_loss2.set_xlabel("Training Step", fontsize=12)
ax_loss2.set_ylabel("Validation Loss", fontsize=12)
bs_str = ','.join(str(b) for b in TARGET_BS_LIST)
ax_loss2.set_title(f"Val Loss vs Step (bs={bs_str})", fontsize=14)
ax_loss2.legend(loc='best', fontsize=9)
ax_loss2.grid(True, alpha=0.3, which='both')

# ===== ROW 3: x-axis = lr * steps, series = LR =====
# Plot eigenvalues for each index
for panel_idx, (ev_idx, source) in enumerate(EIGENVALUE_CONFIG):
    ax = axes[2, panel_idx]
    for i, lr in enumerate(lrs):
        for bs in TARGET_BS_LIST:
            if lr not in data[bs]:
                continue
            x_vals = [lr * step for step in TARGET_STEPS]
            evs = [data[bs][lr]["eigenvalues"][ev_idx].get(step, np.nan) for step in TARGET_STEPS]
            linestyle = BS_LINESTYLES.get(bs, '-')
            label = f'LR={lr:.0e}' if bs == TARGET_BS_LIST[0] else None
            ax.plot(x_vals, evs, linestyle, color=colors_lrs[i], label=label,
                    linewidth=2, markersize=6, marker='o' if bs == TARGET_BS_LIST[0] else 's')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("LR $\\times$ Step", fontsize=12)
    source_label = "Lanczos" if source == "lanczos" else "SLQ"
    ax.set_ylabel(f"$\\lambda_{{{ev_idx}}}$ ({source_label})", fontsize=12)
    bs_str = ','.join(str(b) for b in TARGET_BS_LIST)
    ax.set_title(f"$\\lambda_{{{ev_idx}}}$ vs LR$\\times$Step (bs={bs_str})", fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

# Plot validation loss (last panel, row 3)
ax_loss3 = axes[2, -1]
for i, lr in enumerate(lrs):
    for bs in TARGET_BS_LIST:
        if lr not in data[bs]:
            continue
        x_vals = [lr * step for step in TARGET_STEPS]
        losses = [data[bs][lr]["val_loss"].get(step, np.nan) for step in TARGET_STEPS]
        linestyle = BS_LINESTYLES.get(bs, '-')
        label = f'LR={lr:.0e}' if bs == TARGET_BS_LIST[0] else None
        ax_loss3.plot(x_vals, losses, linestyle, color=colors_lrs[i], label=label,
                      linewidth=2, markersize=6, marker='o' if bs == TARGET_BS_LIST[0] else 's')

ax_loss3.set_xscale('log')
ax_loss3.set_yscale('log')
ax_loss3.set_xlabel("LR $\\times$ Step", fontsize=12)
ax_loss3.set_ylabel("Validation Loss", fontsize=12)
bs_str = ','.join(str(b) for b in TARGET_BS_LIST)
ax_loss3.set_title(f"Val Loss vs LR$\\times$Step (bs={bs_str})", fontsize=14)
ax_loss3.legend(loc='best', fontsize=9)
ax_loss3.grid(True, alpha=0.3, which='both')

plt.tight_layout()

# Save
bs_str_file = '_'.join(str(b) for b in TARGET_BS_LIST)
file_name = f"compare_lr_bs{bs_str_file}_{TARGET_SCHEDULE}.png"
output_path = os.path.join(OUTPUT_DIR, file_name)
plt.savefig(output_path, dpi=150)
print(f"\nSaved plot to {output_path}")

plt.show()

# ============== PRINT SUMMARY ==============
for bs in TARGET_BS_LIST:
    for ev_idx, source in EIGENVALUE_CONFIG:
        source_label = "Lanczos" if source == "lanczos" else "SLQ"
        print("\n" + "="*70)
        print(f"SUMMARY: λ_{ev_idx} ({source_label}) for bs={bs}")
        print("="*70)
        print(f"{'LR':<15}", end="")
        for step in TARGET_STEPS[:5]:  # Limit columns for readability
            print(f"{'Step '+str(step):<15}", end="")
        print()
        print("-"*70)
        for lr in sorted(data[bs].keys()):
            print(f"{lr:<15.1e}", end="")
            for step in TARGET_STEPS[:5]:
                val = data[bs][lr]["eigenvalues"][ev_idx].get(step, np.nan)
                if np.isnan(val):
                    print(f"{'N/A':<15}", end="")
                else:
                    print(f"{val:<15.4f}", end="")
            print()

    print("\n" + "="*70)
    print(f"SUMMARY: Validation Loss for bs={bs}")
    print("="*70)
    print(f"{'LR':<15}", end="")
    for step in TARGET_STEPS[:5]:
        print(f"{'Step '+str(step):<15}", end="")
    print()
    print("-"*70)
    for lr in sorted(data[bs].keys()):
        print(f"{lr:<15.1e}", end="")
        for step in TARGET_STEPS[:5]:
            val = data[bs][lr]["val_loss"].get(step, np.nan)
            if np.isnan(val):
                print(f"{'N/A':<15}", end="")
            else:
                print(f"{val:<15.4f}", end="")
        print()
