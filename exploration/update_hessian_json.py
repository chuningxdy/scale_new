"""
Update existing hessian_step_*.json files with new SLQ index computation.

The new computation uses:
- Normalized weights (sum to 1)
- Midpoint of cumulative-weight bin for index positioning

This script:
1. Backs up original JSON files to *.json.backup
2. Recomputes SLQ indices using the new formula
3. Recomputes power law fit
4. Updates the JSON files
5. Regenerates the plots
"""

import json
import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============== CONFIGURATION ==============
OUTPUT_DIR = "outputs/run_pipeline"
MIN_FIT_INDEX = 20  # Minimum index for power law fitting
PLOT_XLIM = [1e-1, 1e8]
PLOT_YLIM = [1e-3, 1e5]


def get_density(ev, weights, bins=100):
    """Smooth density estimation from eigenvalues and weights."""
    sigma = 0.1 * (max(ev) - min(ev))
    grid = np.linspace(min(ev), max(ev), bins)
    density = np.zeros_like(grid)
    for v, w in zip(ev, weights):
        density += w * np.exp(-(grid - v)**2 / (2 * sigma**2))
    return grid, density


def recompute_slq_indices(raw_eigenvalues, raw_weights, total_params):
    """
    Recompute SLQ indices using the new midpoint formula.

    Returns:
        evs_final: filtered eigenvalues (> 1e-6)
        indices: corresponding indices using midpoint of cumulative-weight bin
    """
    evs = np.array(raw_eigenvalues, dtype=float)
    wts = np.array(raw_weights, dtype=float)

    # Sort by eigenvalue descending
    idx = np.argsort(evs)[::-1]
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]

    # Normalize weights
    wts_sorted = wts_sorted / wts_sorted.sum()

    # Rank location = midpoint of each cumulative-weight bin
    cum = np.cumsum(wts_sorted)
    indices_all = (cum - 0.5 * wts_sorted) * total_params

    # Filter for positive eigenvalues
    mask = evs_sorted > 1e-6
    evs_final = evs_sorted[mask]
    indices = indices_all[mask]

    return evs_final, indices


def fit_power_law(indices, eigenvalues, min_fit_index=20):
    """Fit power law: lambda = c * i^(-p)"""
    fit_mask = indices >= min_fit_index
    c_fit, p_fit = None, None

    if np.sum(fit_mask) > 2:
        log_indices = np.log(indices[fit_mask])
        log_evs = np.log(eigenvalues[fit_mask])
        coeffs = np.polyfit(log_indices, log_evs, 1)
        p_fit = -coeffs[0]
        c_fit = np.exp(coeffs[1])

    return c_fit, p_fit


def regenerate_plot(json_data, output_dir, step):
    """Regenerate the hessian plot from JSON data."""
    # Extract data
    slq_evs = np.array(json_data["slq"]["eigenvalues"])
    slq_indices = np.array(json_data["slq"]["indices"])
    raw_evs = json_data["slq"]["raw_eigenvalues"]
    raw_wts = json_data["slq"]["raw_weights"]
    lanczos_evs = json_data.get("lanczos", {}).get("eigenvalues", [])
    c_fit = json_data.get("power_law_fit", {}).get("c")
    p_fit = json_data.get("power_law_fit", {}).get("p")
    total_params = json_data.get("config", {}).get("total_params", 1e6)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Density plot
    grid, density = get_density(raw_evs, raw_wts)
    ax1.plot(grid, density, color='blue', lw=2)
    ax1.set_title(f"Hessian Eigenvalue Density (Step {step})")
    ax1.set_xlabel(r"Eigenvalue $\lambda$")
    ax1.set_ylabel(r"Density $\rho(\lambda)$")
    ax1.grid(True, alpha=0.3)

    # Log-log plot
    ax2.loglog(slq_indices, slq_evs, marker='o', linestyle='None', alpha=0.6,
               markersize=4, label='SLQ Density')

    if lanczos_evs:
        top_evs = np.array(sorted(lanczos_evs, reverse=True))
        top_indices = np.arange(1, len(top_evs) + 1)
        ax2.loglog(top_indices, top_evs, marker='+', linestyle='None', color='red',
                   markersize=4, label='Top-k Lanczos', zorder=5, alpha=0.6)

    if c_fit is not None and p_fit is not None:
        fit_x = np.logspace(0, np.log10(total_params), 100)
        fit_y = c_fit * fit_x ** (-p_fit)
        ax2.loglog(fit_x, fit_y, 'g--', lw=2,
                   label=f'Fit: $\\lambda = {c_fit:.2f} \\cdot i^{{-{p_fit:.2f}}}$')

    ax2.set_title(f"Log-Log Spectrum (Step {step})")
    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_ylabel(r"Eigenvalue $\lambda$")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xlim(PLOT_XLIM[0], PLOT_XLIM[1])
    ax2.set_ylim(PLOT_YLIM[0], PLOT_YLIM[1])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"hessian_step_{step}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def process_hessian_file(json_path):
    """Process a single hessian JSON file."""
    # Load original data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check if we have the required data
    slq_data = data.get("slq", {})
    raw_evs = slq_data.get("raw_eigenvalues")
    raw_wts = slq_data.get("raw_weights")
    total_params = data.get("config", {}).get("total_params")

    if raw_evs is None or raw_wts is None or total_params is None:
        print(f"  Skipping {json_path}: missing raw data or total_params")
        return False

    # Backup original file
    backup_path = json_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(json_path, backup_path)
        print(f"  Backed up to {backup_path}")
    else:
        print(f"  Backup already exists: {backup_path}")

    # Recompute SLQ indices
    evs_final, indices = recompute_slq_indices(raw_evs, raw_wts, total_params)

    # Recompute power law fit
    c_fit, p_fit = fit_power_law(indices, evs_final, MIN_FIT_INDEX)

    # Update data
    data["slq"]["eigenvalues"] = evs_final.tolist()
    data["slq"]["indices"] = indices.tolist()
    data["power_law_fit"]["c"] = float(c_fit) if c_fit is not None else None
    data["power_law_fit"]["p"] = float(p_fit) if p_fit is not None else None

    # Save updated JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Updated {json_path}")

    # Regenerate plot
    output_dir = os.path.dirname(json_path)
    step = data.get("step", 0)
    plot_path = regenerate_plot(data, output_dir, step)
    print(f"  Regenerated {plot_path}")

    return True


def main():
    """Main function to process all hessian JSON files."""
    # Find all run folders
    run_folders = glob.glob(os.path.join(OUTPUT_DIR, "*"))
    run_folders = [f for f in run_folders if os.path.isdir(f)]

    print(f"Found {len(run_folders)} run folders in {OUTPUT_DIR}")

    total_processed = 0
    total_skipped = 0

    for folder in sorted(run_folders):
        folder_name = os.path.basename(folder)

        # Find all hessian JSON files in this folder
        json_files = glob.glob(os.path.join(folder, "hessian_step_*.json"))

        if not json_files:
            continue

        print(f"\nProcessing {folder_name} ({len(json_files)} hessian files)")

        for json_path in sorted(json_files):
            if json_path.endswith('.backup'):
                continue

            json_name = os.path.basename(json_path)
            print(f"  {json_name}:")

            if process_hessian_file(json_path):
                total_processed += 1
            else:
                total_skipped += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {total_processed}")
    print(f"Total files skipped: {total_skipped}")


if __name__ == "__main__":
    main()
