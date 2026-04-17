"""
Plot full SLQ eigenvalue spectrum as rank plots (eigenvalue vs rank).

Modes:
  --mode time:  One subplot per LR, showing spectrum evolution over training steps
  --mode lr:    One subplot per step, comparing spectra across learning rates
  --mode both:  Both plots

Usage:
    python analyze_spectral_cdf.py --optimizer sgd --mode time --lrs 0.001 0.01 0.1
    python analyze_spectral_cdf.py --optimizer sgd --mode lr --steps 0 100 1000
    python analyze_spectral_cdf.py --optimizer sgd --mode both
"""

import argparse
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def load_run(output_dir):
    pattern = os.path.join(output_dir, "hessian_step_*.json")
    files = glob.glob(pattern)
    data_by_step = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        step = int(filename.replace("hessian_step_", "").replace(".json", ""))
        with open(filepath) as f:
            data_by_step[step] = json.load(f)
    return data_by_step


def get_spectrum(hessian_data):
    """Extract positive and negative rank-eigenvalue curves from SLQ density.
    Returns (pos_ranks, pos_evs, neg_ranks, neg_evs) where:
      pos: sorted by rank from top (rank 1 = largest), eigenvalues positive
      neg: sorted by rank from bottom (rank 1 = most negative), eigenvalues as |lambda|
    """
    slq = hessian_data.get("slq", {})
    evs = np.array(slq.get("raw_eigenvalues", []), dtype=float)
    wts = np.array(slq.get("raw_weights", []), dtype=float)
    total_params = hessian_data.get("config", {}).get("total_params", 1)
    if len(evs) == 0:
        return None, None, None, None

    # Sort descending for positive spectrum
    idx = np.argsort(evs)[::-1]
    evs_desc = evs[idx]
    wts_desc = wts[idx]
    wts_desc = wts_desc / wts_desc.sum()
    cum_desc = np.cumsum(wts_desc)
    ranks_desc = (cum_desc - 0.5 * wts_desc) * total_params

    # Positive eigenvalues
    mask_pos = evs_desc > 1e-10
    pos_ranks = ranks_desc[mask_pos]
    pos_evs = evs_desc[mask_pos]

    # Sort ascending for negative spectrum
    idx_asc = np.argsort(evs)
    evs_asc = evs[idx_asc]
    wts_asc = wts[idx_asc]
    wts_asc = wts_asc / wts_asc.sum()
    cum_asc = np.cumsum(wts_asc)
    ranks_asc = (cum_asc - 0.5 * wts_asc) * total_params

    mask_neg = evs_asc < -1e-10
    neg_ranks = ranks_asc[mask_neg]
    neg_evs = np.abs(evs_asc[mask_neg])

    return pos_ranks, pos_evs, neg_ranks, neg_evs


def find_runs(base_dir, model_id, dataset_name, batch_size, optimizer):
    pattern = os.path.join(base_dir,
        f"{model_id}_{dataset_name}_bs{batch_size}_lr*_opt{optimizer}_constant")
    run_dirs = sorted(glob.glob(pattern))
    runs = {}
    for d in run_dirs:
        dirname = os.path.basename(d)
        lr_str = dirname.split("_lr")[1].split("_opt")[0]
        lr = float(lr_str)
        data = load_run(d)
        if data:
            runs[lr] = data
    return runs


def plot_spectrum(ax, hessian_data, color, label, sign="pos"):
    """Plot one spectrum curve on an axis."""
    pos_ranks, pos_evs, neg_ranks, neg_evs = get_spectrum(hessian_data)
    if sign == "pos" and pos_ranks is not None and len(pos_ranks) > 0:
        ax.loglog(pos_ranks, pos_evs, color=color, label=label, linewidth=1.5, alpha=0.8)
    elif sign == "neg" and neg_ranks is not None and len(neg_ranks) > 0:
        ax.loglog(neg_ranks, neg_evs, color=color, label=label, linewidth=1.5, alpha=0.8)


def plot_time_mode(runs, lrs, output_path, ylim):
    """2 rows (pos, neg) x n_lr cols, showing spectrum evolution over steps."""
    if lrs:
        runs = {lr: d for lr, d in runs.items()
                if any(abs(lr - l) / max(lr, 1e-10) < 0.01 for l in lrs)}
    sorted_lrs = sorted(runs.keys())
    n = len(sorted_lrs)
    if n == 0:
        print("No matching runs")
        return

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, lr in enumerate(sorted_lrs):
        data_by_step = runs[lr]
        steps = sorted(data_by_step.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))

        for i, step in enumerate(steps):
            plot_spectrum(axes[0, col], data_by_step[step], colors[i], f'{step}', "pos")
            plot_spectrum(axes[1, col], data_by_step[step], colors[i], f'{step}', "neg")

        axes[0, col].set_title(f"lr={lr}")
        for row in range(2):
            axes[row, col].grid(True, which="both", alpha=0.3)
            axes[row, col].legend(fontsize=5, loc="best", title="step", title_fontsize=6)
            if ylim:
                axes[row, col].set_ylim(ylim)

    axes[0, 0].set_ylabel("Eigenvalue")
    axes[1, 0].set_ylabel(r"$|\lambda|$ (negative)")
    for col in range(n):
        axes[1, col].set_xlabel("Rank")

    fig.suptitle("Eigenvalue Spectrum Evolution Over Training", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


def plot_lr_mode(runs, steps, output_path, ylim):
    """2 rows (pos, neg) x n_steps cols, comparing spectra across LRs."""
    sorted_lrs = sorted(runs.keys())
    if not steps:
        all_steps = set()
        for data in runs.values():
            all_steps.update(data.keys())
        steps = sorted(all_steps)

    valid_steps = [s for s in steps if any(s in data for data in runs.values())]
    n = len(valid_steps)
    if n == 0:
        print("No matching steps")
        return

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_lrs)))

    for col, step in enumerate(valid_steps):
        for i, lr in enumerate(sorted_lrs):
            if step not in runs[lr]:
                continue
            plot_spectrum(axes[0, col], runs[lr][step], colors[i], f'{lr}', "pos")
            plot_spectrum(axes[1, col], runs[lr][step], colors[i], f'{lr}', "neg")

        axes[0, col].set_title(f"Step {step}")
        for row in range(2):
            axes[row, col].grid(True, which="both", alpha=0.3)
            axes[row, col].legend(fontsize=6, loc="best", title="lr", title_fontsize=7)
            if ylim:
                axes[row, col].set_ylim(ylim)

    axes[0, 0].set_ylabel("Eigenvalue")
    axes[1, 0].set_ylabel(r"$|\lambda|$ (negative)")
    for col in range(n):
        axes[1, col].set_xlabel("Rank")

    fig.suptitle("Eigenvalue Spectrum Across Learning Rates", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--mode", type=str, default="both", choices=["time", "lr", "both"])
    parser.add_argument("--lrs", type=float, nargs="+", default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--model-id", type=str, default="pythia70m")
    parser.add_argument("--dataset-name", type=str, default="tinystories")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ylim", type=float, nargs=2, default=None, help="Y-axis limits, e.g. --ylim 1 1000")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    base_dir = "outputs/run_pipeline"
    runs = find_runs(base_dir, args.model_id, args.dataset_name, args.batch_size, args.optimizer)
    if not runs:
        print("No runs found")
        return
    print(f"Found {len(runs)} runs: {sorted(runs.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.model_id}_{args.optimizer}"
    ylim = tuple(args.ylim) if args.ylim else None

    if args.mode in ("time", "both"):
        path = os.path.join(args.output_dir, f"spectral_rank_time_{tag}.png")
        plot_time_mode(runs, args.lrs, path, ylim)

    if args.mode in ("lr", "both"):
        path = os.path.join(args.output_dir, f"spectral_rank_lr_{tag}.png")
        plot_lr_mode(runs, args.steps, path, ylim)


if __name__ == "__main__":
    main()
