import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============== CONFIGURATION ==============
HESSIAN_OUTPUT_DIR = "./hessian_analysis"
# Select which eigenvalues to plot (1-indexed, e.g., [1, 2, 3] for top 3)
# Set to None to plot top NUM_EIGENVALUES
EIGENVALUE_INDICES = [1, 2, 4, 8, 16, 32, 64, 128]  # Example: plot 1st, 2nd, 3rd, 5th, 10th, 15th, 20th
NUM_EIGENVALUES = 3  # Used only if EIGENVALUE_INDICES is None

# Specify which indices use Lanczos vs SLQ (1-indexed)
# Indices in LANCZOS_INDICES use Lanczos; all others use SLQ or SLQ_INTERP
LANCZOS_INDICES = [1, 2, 3]  # e.g., first 3 eigenvalues from Lanczos, rest from SLQ
# Set to None to use EIGENVALUE_SOURCE for all indices
# Set to "all" to use Lanczos for all indices

# Source for non-Lanczos indices (or all if LANCZOS_INDICES is None):
#   "lanczos"    - exact top-k from power iteration
#   "slq"        - raw Ritz values (just sorted, pick index)
#   "slq_interp" - log-log interpolation using spectral mass (power-law aware)
EIGENVALUE_SOURCE = "slq_interp"

# ============== LOAD DATA ==============
# Find all hessian step files
pattern = os.path.join(HESSIAN_OUTPUT_DIR, "hessian_step_*.json")
files = glob.glob(pattern)

if not files:
    raise FileNotFoundError(f"No hessian files found in {HESSIAN_OUTPUT_DIR}")

# Parse step numbers and sort
data_by_step = {}
for filepath in files:
    filename = os.path.basename(filepath)
    step = int(filename.replace("hessian_step_", "").replace(".json", ""))
    with open(filepath, "r") as f:
        data_by_step[step] = json.load(f)

steps = sorted(data_by_step.keys())
print(f"Found data for steps: {steps}")

# ============== EXTRACT EIGENVALUES ==============
# Determine which eigenvalue indices to track (convert to 0-indexed)
if EIGENVALUE_INDICES is not None:
    indices_to_track = [i - 1 for i in EIGENVALUE_INDICES]  # Convert 1-indexed to 0-indexed
else:
    indices_to_track = list(range(NUM_EIGENVALUES))

# Determine which indices use Lanczos (0-indexed)
if LANCZOS_INDICES == "all":
    lanczos_indices_0 = set(indices_to_track)
elif LANCZOS_INDICES is not None:
    lanczos_indices_0 = set(i - 1 for i in LANCZOS_INDICES)  # Convert 1-indexed to 0-indexed
else:
    # Use legacy EIGENVALUE_SOURCE for all
    lanczos_indices_0 = set(indices_to_track) if EIGENVALUE_SOURCE == "lanczos" else set()

# Determine non-Lanczos source
non_lanczos_source = EIGENVALUE_SOURCE if EIGENVALUE_SOURCE in ["slq", "slq_interp"] else "slq"

# Get eigenvalues for each step
eigenvalue_trajectories = {i: [] for i in indices_to_track}
eigenvalue_sources = {i: "lanczos" if i in lanczos_indices_0 else non_lanczos_source for i in indices_to_track}
valid_steps = []

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

    # Compute effective indices
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

for step in steps:
    data = data_by_step[step]

    # Get Lanczos eigenvalues
    lanczos_evs = data.get("lanczos", {}).get("eigenvalues", [])
    sorted_lanczos = sorted(lanczos_evs, reverse=True) if lanczos_evs else []

    # Get SLQ eigenvalues and weights
    slq_evs = data.get("slq", {}).get("raw_eigenvalues", [])
    slq_wts = data.get("slq", {}).get("raw_weights", [])
    total_params = data.get("config", {}).get("total_params", None)

    # Sort SLQ (for non-interp mode)
    sorted_slq = sorted(slq_evs, reverse=True) if slq_evs else []

    if sorted_lanczos or sorted_slq:
        valid_steps.append(step)

        for i in indices_to_track:
            target_index = i + 1  # Convert 0-indexed to 1-indexed for interpolation

            if i in lanczos_indices_0:
                # Use Lanczos
                if i < len(sorted_lanczos):
                    eigenvalue_trajectories[i].append(sorted_lanczos[i])
                else:
                    eigenvalue_trajectories[i].append(np.nan)
            elif non_lanczos_source == "slq_interp" and total_params is not None:
                # Use SLQ with log-log interpolation
                ev = slq_interp_eigenvalue(slq_evs, slq_wts, total_params, target_index)
                eigenvalue_trajectories[i].append(ev)
            else:
                # Use raw SLQ (just pick index)
                if i < len(sorted_slq):
                    eigenvalue_trajectories[i].append(sorted_slq[i])
                else:
                    eigenvalue_trajectories[i].append(np.nan)

# Print source info
print(f"Eigenvalue sources:")
for i in indices_to_track:
    print(f"  λ{i+1}: {eigenvalue_sources[i]}")
print(f"Tracking {len(indices_to_track)} eigenvalues across {len(valid_steps)} steps")

# ============== PLOT EIGENVALUE TRAJECTORIES ==============
fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.viridis(np.linspace(0, 1, len(indices_to_track)))

for idx, i in enumerate(indices_to_track):
    evs = eigenvalue_trajectories[i]
    source = eigenvalue_sources[i]
    marker = 'o' if source == "lanczos" else 's'  # Circle for Lanczos, square for SLQ
    ax.semilogy(valid_steps, evs, marker=marker, color=colors[idx],
                label=f'$\\lambda_{{{i+1}}}$ ({source})', linewidth=2, markersize=6)

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Eigenvalue (log scale)", fontsize=12)
# Determine title based on sources used
sources_used = set(eigenvalue_sources.values())
if len(sources_used) == 1:
    source_label = list(sources_used)[0].capitalize()
else:
    source_label = "Mixed (Lanczos + SLQ)"
ax.set_title(f"Trajectory of Selected Hessian Eigenvalues ({source_label}) During Training", fontsize=14)
ax.legend(loc='best', ncol=2)
ax.grid(True, alpha=0.3, which='both')
ax.set_xticks(valid_steps)

plt.tight_layout()
output_path = os.path.join(HESSIAN_OUTPUT_DIR, "eigenvalue_trajectories.png")
plt.savefig(output_path, dpi=150)
plt.show()
print(f"Saved plot to {output_path}")

# ============== PRINT SUMMARY ==============
print("\n" + "="*60)
print("EIGENVALUE SUMMARY")
print("="*60)
display_indices = indices_to_track[:min(5, len(indices_to_track))]
print(f"{'Step':<10}", end="")
for i in display_indices:
    print(f"{'λ'+str(i+1):<12}", end="")
print()
print("-"*60)

for step_idx, step in enumerate(valid_steps):
    print(f"{step:<10}", end="")
    for i in display_indices:
        val = eigenvalue_trajectories[i][step_idx]
        print(f"{val:<12.4f}", end="")
    print()
