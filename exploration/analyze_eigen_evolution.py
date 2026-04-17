"""
Analyze evolution of top positive and negative Hessian eigenvalues across training.

Tracks eigenvalues at specific ranks (default: 1, 4, 16, 64, 256).
- Ranks <= 10: use Lanczos (exact) eigenvalues
- Ranks > 10: use SLQ density interpolation

Usage:
    python analyze_eigen_evolution.py <output_dir>
    python analyze_eigen_evolution.py <output_dir> --ranks 1 4 16 64 256
"""

import argparse
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def load_hessian_data(output_dir):
    """Load all hessian_step_*.json files, return dict keyed by step."""
    pattern = os.path.join(output_dir, "hessian_step_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No hessian files found in {output_dir}")

    data_by_step = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        step = int(filename.replace("hessian_step_", "").replace(".json", ""))
        with open(filepath) as f:
            data_by_step[step] = json.load(f)

    return data_by_step


def slq_interp_eigenvalue(evs, wts, total_params, target_index):
    """Interpolate eigenvalue at target_index (1-indexed) from SLQ density using log-log interpolation.
    Eigenvalues sorted descending (positive end). target_index=1 is largest."""
    if len(evs) == 0 or len(wts) == 0:
        return np.nan

    evs = np.array(evs, dtype=float)
    wts = np.array(wts, dtype=float)

    idx = np.argsort(evs)[::-1]
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]
    wts_sorted = wts_sorted / wts_sorted.sum()

    cum = np.cumsum(wts_sorted)
    indices_all = (cum - 0.5 * wts_sorted) * total_params

    mask = evs_sorted > 1e-10
    evs_pos = evs_sorted[mask]
    indices = indices_all[mask]

    if len(evs_pos) < 2:
        return np.nan
    if target_index < indices[0] or target_index > indices[-1]:
        return np.nan

    log_indices = np.log(indices)
    log_evs = np.log(evs_pos)
    log_result = np.interp(np.log(target_index), log_indices, log_evs)
    return np.exp(log_result)


def slq_interp_eigenvalue_neg(evs, wts, total_params, target_index):
    """Interpolate negative eigenvalue at target_index (1-indexed) from SLQ density.
    target_index=1 is most negative. Ranks within the negative portion only.
    Uses log-log interpolation on |eigenvalue|."""
    if len(evs) == 0 or len(wts) == 0:
        return np.nan

    evs = np.array(evs, dtype=float)
    wts = np.array(wts, dtype=float)

    # Sort ascending (most negative first)
    idx = np.argsort(evs)
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]
    wts_sorted = wts_sorted / wts_sorted.sum()

    # Filter negative eigenvalues
    mask = evs_sorted < -1e-10
    evs_neg = evs_sorted[mask]
    wts_neg = wts_sorted[mask]

    if len(evs_neg) < 2:
        return np.nan

    # Rank within negative portion: renormalize weights to sum to total negative count
    neg_count = wts_neg.sum() * total_params
    wts_neg_norm = wts_neg / wts_neg.sum()
    cum = np.cumsum(wts_neg_norm)
    indices = (cum - 0.5 * wts_neg_norm) * neg_count

    if target_index < indices[0] or target_index > indices[-1]:
        return np.nan

    # Log-log interpolation on magnitudes
    log_indices = np.log(indices)
    log_abs_evs = np.log(np.abs(evs_neg))
    log_result = np.interp(np.log(target_index), log_indices, log_abs_evs)
    return -np.exp(log_result)


def get_pos_eigenvalue(hessian_data, rank):
    """Get positive eigenvalue at given rank (1-indexed). Lanczos for rank<=10, SLQ otherwise."""
    if rank <= 10:
        lanczos_evs = sorted(hessian_data.get("lanczos", {}).get("eigenvalues", []), reverse=True)
        idx = rank - 1
        if idx < len(lanczos_evs):
            return lanczos_evs[idx]
        return np.nan
    else:
        slq_evs = hessian_data.get("slq", {}).get("raw_eigenvalues", [])
        slq_wts = hessian_data.get("slq", {}).get("raw_weights", [])
        total_params = hessian_data.get("config", {}).get("total_params")
        if not slq_evs or not slq_wts or total_params is None:
            return np.nan
        return slq_interp_eigenvalue(slq_evs, slq_wts, total_params, rank)


def get_neg_eigenvalue(hessian_data, rank):
    """Get negative eigenvalue at given rank (1-indexed, 1=most negative).
    Lanczos for rank<=10 if available, falls back to SLQ on -H density."""
    if rank <= 10:
        neg_evs = sorted(hessian_data.get("lanczos_bottom", {}).get("eigenvalues", []))
        idx = rank - 1
        if idx < len(neg_evs):
            return neg_evs[idx]
        # Fall back to SLQ if Lanczos didn't find enough negative eigenvalues
    # Use dedicated -H SLQ density if available (negated back to H eigenvalues)
    slq_bottom = hessian_data.get("slq_bottom", {})
    slq_neg_evs = slq_bottom.get("raw_eigenvalues", [])
    slq_neg_wts = slq_bottom.get("raw_weights", [])
    total_params = hessian_data.get("config", {}).get("total_params")
    if slq_neg_evs and slq_neg_wts and total_params:
        # These are already negated H eigenvalues; use positive SLQ interp on the negative values
        # The most negative eigenvalue of H = most positive eigenvalue of -H
        # slq_neg_evs are H eigenvalues (negated from -H density), so negative values are what we want
        neg_only = [ev for ev in slq_neg_evs if ev < -1e-10]
        neg_wts = [slq_neg_wts[i] for i, ev in enumerate(slq_neg_evs) if ev < -1e-10]
        if len(neg_only) >= 2:
            return slq_interp_eigenvalue_neg(neg_only, neg_wts, total_params, rank)
    # Fall back to original SLQ density
    slq_evs = hessian_data.get("slq", {}).get("raw_eigenvalues", [])
    slq_wts = hessian_data.get("slq", {}).get("raw_weights", [])
    if not slq_evs or not slq_wts or total_params is None:
        return np.nan
    return slq_interp_eigenvalue_neg(slq_evs, slq_wts, total_params, rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Path to run output directory")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 16, 64, 256],
                        help="Eigenvalue ranks to track (1-indexed)")
    args = parser.parse_args()

    data_by_step = load_hessian_data(args.output_dir)
    steps = sorted(data_by_step.keys())
    ranks = args.ranks

    print(f"Found {len(steps)} steps: {steps}")
    print(f"Tracking ranks: {ranks} (Lanczos for rank<=10, SLQ for rank>10)")

    # Extract eigenvalues at each rank and step
    pos_trajectories = {r: [] for r in ranks}
    neg_trajectories = {r: [] for r in ranks}
    valid_steps = []

    for step in steps:
        d = data_by_step[step]
        valid_steps.append(step)
        for r in ranks:
            pos_trajectories[r].append(get_pos_eigenvalue(d, r))
            neg_trajectories[r].append(get_neg_eigenvalue(d, r))

    valid_steps = np.array(valid_steps)
    # Replace step 0 with 0.5 for log-scale plotting
    plot_steps = np.where(valid_steps == 0, 0.5, valid_steps)

    # Plot
    fig, (ax_pos, ax_neg) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(ranks)))

    # Top panel: positive eigenvalues (log-log)
    for i, r in enumerate(ranks):
        vals = pos_trajectories[r]
        source = "L" if r <= 10 else "SLQ"
        ax_pos.loglog(plot_steps, vals, marker='o', markersize=4, color=colors[i],
                      label=f'rank {r} ({source})')
    ax_pos.set_ylabel("Eigenvalue")
    ax_pos.set_title("Top Positive Eigenvalues")
    ax_pos.legend(loc="best", ncol=2, fontsize=8)
    ax_pos.grid(True, which="both", alpha=0.3)

    # Bottom panel: negative eigenvalues (log-log, plot |eigenvalue|)
    for i, r in enumerate(ranks):
        vals = [abs(v) if not np.isnan(v) else np.nan for v in neg_trajectories[r]]
        source = "L" if r <= 10 else "SLQ"
        ax_neg.loglog(plot_steps, vals, marker='s', markersize=6, color=colors[i],
                      label=f'rank {r} ({source})', alpha=0.7, zorder=len(ranks) - i)
    ax_neg.set_xlabel("Training Step")
    ax_neg.set_ylabel(r"$|\lambda|$")
    ax_neg.set_title("Top Negative Eigenvalues (magnitude)")
    ax_neg.legend(loc="best", ncol=2, fontsize=8)
    ax_neg.grid(True, which="both", alpha=0.3)

    # Unify y-limits across both panels
    pos_ylim = ax_pos.get_ylim()
    neg_ylim = ax_neg.get_ylim()
    shared_ylim = (min(pos_ylim[0], neg_ylim[0]), max(pos_ylim[1], neg_ylim[1]))
    ax_pos.set_ylim(shared_ylim)
    ax_neg.set_ylim(shared_ylim)

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "eigen_evolution.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
