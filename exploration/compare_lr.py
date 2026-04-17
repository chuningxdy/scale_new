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
TARGET_MODEL = "tinystories" #"pythia70m"  # Model filter, e.g. "pythia70m". Set to None for all
TARGET_DATASET = "tinystories" #"lm1b"  # Dataset filter, e.g. "lm1b", "tinystories". Set to None for all
TARGET_BS_LIST = [64, 256, 1024]  # List of batch sizes to compare
BS_LINESTYLES = {64:'-', 256: '--', 1024: ':'}  # solid for 256, dotted for 1024
TARGET_OPTIMIZER = "sgd"  # "adam" or "sgd". Set to None for all optimizers
TARGET_SCHEDULE = "constant"  # "constant", "linear", etc. Set to None for all schedules
TARGET_LRS = [3e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3,1e-2,3e-2,1e-1] # None  # List of learning rates to plot, e.g. [1e-4, 3e-4, 1e-3]. Set to None for all
STEP_INTERVAL = 20  # Step interval for analysis
MAX_STEPS = 10000  # Maximum step to consider
TARGET_STEPS = list(range(0, MAX_STEPS + 1, STEP_INTERVAL))  # [0, 20, 40, ...]
LR_STEP_INTERSECTIONS = [1.0, 2.0]  # x-values (LR × Step) for intersection analysis in Row 4, 5
BS_POWER = 1.0 # Power of BS in denominator for intersection plots (1.0 = LR/BS, 0.5 = LR/√BS)
REF_SLOPE = -1  # Slope of reference line on log-log scale in intersection plots
# Eigenvalue indices and their sources: (index, source)
# source can be "lanczos" or "slq"
EIGENVALUE_CONFIG = [
    (1, "lanczos"),  # λ_1 from Lanczos
    (16, "slq"),      # λ_8 from SLQ
    (128, "slq"),     # λ_64 from SLQ
]

# ============== HELPER FUNCTIONS ==============

def parse_folder_name(folder_name):
    """Extract model, dataset, batch_size, learning_rate, optimizer, and schedule from folder name.

    New format with optimizer: 'pythia70m_lm1b_bs256_lr0.0001_optadam_constant'
    Previous format: 'pythia70m_lm1b_bs256_lr0.0001_constant'
    Old format: 'bs256_lr0.0001_constant' or 'bs256_lr0.0001'
    """
    # New format with optimizer: model_dataset_bs{}_lr{}_opt{optimizer}_{schedule}
    match = re.match(r'([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_bs(\d+)_lr([\d.e-]+)_opt(\w+)_(\w+)', folder_name)
    if match:
        model = match.group(1)
        dataset = match.group(2)
        bs = int(match.group(3))
        lr = float(match.group(4))
        optimizer = match.group(5)
        schedule = match.group(6)
        return model, dataset, bs, lr, optimizer, schedule

    # Previous format without optimizer: model_dataset_bs{}_lr{}_{schedule}
    match = re.match(r'([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_bs(\d+)_lr([\d.e-]+)_(\w+)', folder_name)
    if match:
        model = match.group(1)
        dataset = match.group(2)
        bs = int(match.group(3))
        lr = float(match.group(4))
        schedule = match.group(5)
        return model, dataset, bs, lr, "adam", schedule  # Default to adam

    # Old format with schedule: bs{}_lr{}_schedule (default to tinystories)
    match = re.match(r'bs(\d+)_lr([\d.e-]+)_(\w+)', folder_name)
    if match:
        bs = int(match.group(1))
        lr = float(match.group(2))
        schedule = match.group(3)
        return "tinystories", "tinystories", bs, lr, "adam", schedule

    # Old format without schedule: bs{}_lr{} (default to tinystories)
    match = re.match(r'bs(\d+)_lr([\d.e-]+)', folder_name)
    if match:
        bs = int(match.group(1))
        lr = float(match.group(2))
        return "tinystories", "tinystories", bs, lr, "adam", None

    return None, None, None, None, None, None


def slq_interp_eigenvalue(evs, wts, total_params, target_index):
    """Interpolate eigenvalue at target_index using log-log interpolation (power-law aware).

    Uses updated formula:
    - Normalized weights (sum to 1)
    - Midpoint of cumulative-weight bin for index positioning
    """
    if len(evs) == 0 or len(wts) == 0:
        return np.nan

    evs = np.array(evs, dtype=float)
    wts = np.array(wts, dtype=float)

    # Sort by eigenvalue descending
    idx = np.argsort(evs)[::-1]
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]

    # Normalize weights
    wts_sorted = wts_sorted / wts_sorted.sum()

    # Compute indices using midpoint of cumulative-weight bin
    cum = np.cumsum(wts_sorted)
    indices_all = (cum - 0.5 * wts_sorted) * total_params

    # Filter positive eigenvalues (needed for log)
    mask = evs_sorted > 1e-10
    evs_pos = evs_sorted[mask]
    indices = indices_all[mask]

    if len(evs_pos) < 2:
        return np.nan

    # Check if target_index is in range
    if target_index < indices[0] or target_index > indices[-1]:
        # Debug: uncomment to see why NaN is returned
        # print(f"  target_index={target_index} out of range [{indices[0]:.1f}, {indices[-1]:.1f}]")
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
print(f"Scanning {OUTPUT_DIR} for model={TARGET_MODEL}, dataset={TARGET_DATASET}, bs={TARGET_BS_LIST}, opt={TARGET_OPTIMIZER}, schedule={TARGET_SCHEDULE}, lrs={TARGET_LRS} ...")

# Find all run folders (both old and new format)
run_folders = glob.glob(os.path.join(OUTPUT_DIR, "*_bs*_lr*")) + glob.glob(os.path.join(OUTPUT_DIR, "bs*_lr*"))
run_folders = list(set(run_folders))  # Remove duplicates

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
    model, dataset, bs, lr, optimizer, schedule = parse_folder_name(folder_name)

    if bs is None or bs not in TARGET_BS_LIST:
        continue
    if TARGET_MODEL is not None and model != TARGET_MODEL:
        continue
    if TARGET_DATASET is not None and dataset != TARGET_DATASET:
        continue
    if TARGET_OPTIMIZER is not None and optimizer != TARGET_OPTIMIZER:
        continue
    if TARGET_SCHEDULE is not None and schedule != TARGET_SCHEDULE:
        continue
    if not lr_matches(lr, TARGET_LRS):
        continue

    print(f"  Processing {folder_name} (model={model}, dataset={dataset}, bs={bs}, lr={lr}, opt={optimizer}, schedule={schedule})")

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

# ============== HELPER: Interpolate at x-value ==============
def interpolate_at_x(x_vals, y_vals, target_x):
    """Interpolate y at target_x using log-log interpolation."""
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)

    # Remove NaN and non-positive values
    valid = ~np.isnan(y_arr) & (x_arr > 0) & (y_arr > 0)
    x_valid = x_arr[valid]
    y_valid = y_arr[valid]

    if len(x_valid) < 2:
        return np.nan

    # Check if target_x is in range
    if target_x < x_valid.min() or target_x > x_valid.max():
        return np.nan

    # Log-log interpolation
    log_y = np.interp(np.log(target_x), np.log(x_valid), np.log(y_valid))
    return np.exp(log_y)

# ============== PLOT ==============
num_ev_cols = len(EIGENVALUE_CONFIG)
num_rows = 3 + len(LR_STEP_INTERSECTIONS) + 1  # 3 base rows + 1 row per intersection + 1 row for loss vs eigenvalue
fig, axes = plt.subplots(num_rows, num_ev_cols + 1, figsize=(6 * (num_ev_cols + 1), 5 * num_rows))

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
# Collect intersection points for each intersection value: {intersect_val: {ev_idx: [(lambda, bs, lr), ...]}}
all_intersection_data = {}
for intersect_val in LR_STEP_INTERSECTIONS:
    all_intersection_data[intersect_val] = {ev_idx: [] for ev_idx, _ in EIGENVALUE_CONFIG}
    all_intersection_data[intersect_val]["val_loss"] = []

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

            # Compute intersections at all LR_STEP_INTERSECTIONS
            for intersect_val in LR_STEP_INTERSECTIONS:
                y_intersect = interpolate_at_x(x_vals, evs, intersect_val)
                if not np.isnan(y_intersect):
                    all_intersection_data[intersect_val][ev_idx].append((y_intersect, bs, lr))

    # Add vertical lines at all intersection x-values
    for intersect_val in LR_STEP_INTERSECTIONS:
        ax.axvline(x=intersect_val, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

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

        # Compute intersections at all LR_STEP_INTERSECTIONS
        for intersect_val in LR_STEP_INTERSECTIONS:
            y_intersect = interpolate_at_x(x_vals, losses, intersect_val)
            if not np.isnan(y_intersect):
                all_intersection_data[intersect_val]["val_loss"].append((y_intersect, bs, lr))

# Add vertical lines at all intersection x-values
for intersect_val in LR_STEP_INTERSECTIONS:
    ax_loss3.axvline(x=intersect_val, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

ax_loss3.set_xscale('log')
ax_loss3.set_yscale('log')
ax_loss3.set_xlabel("LR $\\times$ Step", fontsize=12)
ax_loss3.set_ylabel("Validation Loss", fontsize=12)
bs_str = ','.join(str(b) for b in TARGET_BS_LIST)
ax_loss3.set_title(f"Val Loss vs LR$\\times$Step (bs={bs_str})", fontsize=14)
ax_loss3.legend(loc='best', fontsize=9)
ax_loss3.grid(True, alpha=0.3, which='both')

# ===== ROWS 4+: Eigenvalue/Loss at intersection vs LR/BS (one row per intersection value) =====
# Color map for batch sizes
colors_bs = plt.cm.Set1(np.linspace(0, 1, len(TARGET_BS_LIST)))

for row_offset, intersect_val in enumerate(LR_STEP_INTERSECTIONS):
    row_idx = 3 + row_offset
    intersection_data = all_intersection_data[intersect_val]

    for panel_idx, (ev_idx, source) in enumerate(EIGENVALUE_CONFIG):
        ax = axes[row_idx, panel_idx]
        points = intersection_data[ev_idx]

        # Collect all plotted points for reference line
        all_x_vals, all_y_vals, all_bs_vals = [], [], []
        for bs_idx, bs in enumerate(TARGET_BS_LIST):
            bs_points = [(y, lr) for y, b, lr in points if b == bs]
            if bs_points:
                y_vals = [p[0] for p in bs_points]
                lr_bs_vals = [p[1] / (bs ** BS_POWER) for p in bs_points]
                ax.scatter(lr_bs_vals, y_vals, color=colors_bs[bs_idx], s=80,
                           label=f'bs={bs}', marker='o' if bs_idx == 0 else 's', alpha=0.8)
                all_x_vals.extend(lr_bs_vals)
                all_y_vals.extend(y_vals)
                all_bs_vals.extend([bs] * len(lr_bs_vals))

        # Add reference line with slope -1 through leftmost point of largest BS
        if all_x_vals:
            max_bs = max(TARGET_BS_LIST)
            max_bs_points = [(x, y) for x, y, b in zip(all_x_vals, all_y_vals, all_bs_vals) if b == max_bs]
            if max_bs_points:
                # Find leftmost point (smallest x)
                anchor_x, anchor_y = min(max_bs_points, key=lambda p: p[0])
                # Reference line: y = anchor_y * (x / anchor_x)^slope
                x_range = np.logspace(np.log10(min(all_x_vals) * 0.5), np.log10(max(all_x_vals) * 2), 100)
                y_ref = anchor_y * (x_range / anchor_x) ** REF_SLOPE
                ax.plot(x_range, y_ref, 'k--', linewidth=1.5, alpha=0.7, label=f'slope={REF_SLOPE}')

        ax.set_xscale('log')
        ax.set_yscale('log')
        bs_label = r"LR / BS" if BS_POWER == 1.0 else (r"LR / $\sqrt{\mathrm{BS}}$" if BS_POWER == 0.5 else f"LR / BS$^{{{BS_POWER}}}$")
        ax.set_xlabel(bs_label, fontsize=12)
        source_label = "Lanczos" if source == "lanczos" else "SLQ"
        ax.set_ylabel(f"$\\lambda_{{{ev_idx}}}$ at LR$\\times$Step={intersect_val}", fontsize=12)
        ax.set_title(f"$\\lambda_{{{ev_idx}}}$ vs {bs_label} (LR$\\times$Step={intersect_val})", fontsize=14)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

    # Plot validation loss intersection vs LR/BS (last panel)
    ax_loss_row = axes[row_idx, -1]
    points = intersection_data["val_loss"]

    # Collect all plotted points for reference line
    all_x_vals, all_y_vals, all_bs_vals = [], [], []
    for bs_idx, bs in enumerate(TARGET_BS_LIST):
        bs_points = [(y, lr) for y, b, lr in points if b == bs]
        if bs_points:
            y_vals = [p[0] for p in bs_points]
            lr_bs_vals = [p[1] / (bs ** BS_POWER) for p in bs_points]
            ax_loss_row.scatter(lr_bs_vals, y_vals, color=colors_bs[bs_idx], s=80,
                                label=f'bs={bs}', marker='o' if bs_idx == 0 else 's', alpha=0.8)
            all_x_vals.extend(lr_bs_vals)
            all_y_vals.extend(y_vals)
            all_bs_vals.extend([bs] * len(lr_bs_vals))

    # Add reference line with slope -1 through leftmost point of largest BS
    if all_x_vals:
        max_bs = max(TARGET_BS_LIST)
        max_bs_points = [(x, y) for x, y, b in zip(all_x_vals, all_y_vals, all_bs_vals) if b == max_bs]
        if max_bs_points:
            # Find leftmost point (smallest x)
            anchor_x, anchor_y = min(max_bs_points, key=lambda p: p[0])
            # Horizontal reference line through the anchor point
            x_range = np.logspace(np.log10(min(all_x_vals) * 0.5), np.log10(max(all_x_vals) * 2), 100)
            ax_loss_row.axhline(y=anchor_y, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='horizontal')

    ax_loss_row.set_xscale('log')
    ax_loss_row.set_yscale('log')
    bs_label = r"LR / BS" if BS_POWER == 1.0 else (r"LR / $\sqrt{\mathrm{BS}}$" if BS_POWER == 0.5 else f"LR / BS$^{{{BS_POWER}}}$")
    ax_loss_row.set_xlabel(bs_label, fontsize=12)
    ax_loss_row.set_ylabel(f"Val Loss at LR$\\times$Step={intersect_val}", fontsize=12)
    ax_loss_row.set_title(f"Val Loss vs {bs_label} (LR$\\times$Step={intersect_val})", fontsize=14)
    ax_loss_row.legend(loc='best', fontsize=9)
    ax_loss_row.grid(True, alpha=0.3, which='both')

# ===== LAST ROW: Loss vs Eigenvalue at LR×Step=1.0 =====
loss_vs_ev_row = 3 + len(LR_STEP_INTERSECTIONS)
loss_vs_ev_intersect = 1.0  # Use LR×Step=1.0 for this analysis

if loss_vs_ev_intersect in all_intersection_data:
    intersection_data = all_intersection_data[loss_vs_ev_intersect]
    loss_points = intersection_data["val_loss"]  # [(loss, bs, lr), ...]

    for panel_idx, (ev_idx, source) in enumerate(EIGENVALUE_CONFIG):
        ax = axes[loss_vs_ev_row, panel_idx]
        ev_points = intersection_data[ev_idx]  # [(eigenvalue, bs, lr), ...]

        # Match eigenvalue and loss by (bs, lr) key
        ev_dict = {(b, lr): ev for ev, b, lr in ev_points}
        loss_dict = {(b, lr): loss for loss, b, lr in loss_points}

        for bs_idx, bs in enumerate(TARGET_BS_LIST):
            ev_vals = []
            loss_vals = []
            for key in ev_dict:
                if key[0] == bs and key in loss_dict:
                    ev_vals.append(ev_dict[key])
                    loss_vals.append(loss_dict[key])

            if ev_vals:
                ax.scatter(ev_vals, loss_vals, color=colors_bs[bs_idx], s=80,
                           label=f'bs={bs}', marker='o' if bs_idx == 0 else 's', alpha=0.8)

        ax.set_xscale('log')
        ax.set_yscale('log')
        source_label = "Lanczos" if source == "lanczos" else "SLQ"
        ax.set_xlabel(f"$\\lambda_{{{ev_idx}}}$ ({source_label})", fontsize=12)
        ax.set_ylabel("Validation Loss", fontsize=12)
        ax.set_title(f"Loss vs $\\lambda_{{{ev_idx}}}$ (LR$\\times$Step={loss_vs_ev_intersect})", fontsize=14)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

    # Leave last column empty
    axes[loss_vs_ev_row, -1].axis('off')

plt.tight_layout()

# Save
bs_str_file = '_'.join(str(b) for b in TARGET_BS_LIST)
model_str = TARGET_MODEL if TARGET_MODEL else "all"
dataset_str = TARGET_DATASET if TARGET_DATASET else "all"
optimizer_str = TARGET_OPTIMIZER if TARGET_OPTIMIZER else "all"
schedule_str = TARGET_SCHEDULE if TARGET_SCHEDULE else "all"
file_name = f"compare_lr_{model_str}_{dataset_str}_bs{bs_str_file}_opt{optimizer_str}_{schedule_str}.png"
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
