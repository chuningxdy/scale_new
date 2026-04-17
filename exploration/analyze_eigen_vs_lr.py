"""
Plot eigenvalue evolution across training steps, with one subplot per eigenvalue rank
and one series per learning rate.

Usage:
    python analyze_eigen_vs_lr.py --optimizer sgd --ranks 1 4 16 64
    python analyze_eigen_vs_lr.py --optimizer adam --ranks 1 4 16 64
"""

import argparse
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def slq_interp_eigenvalue(evs, wts, total_params, target_index, return_bounds=False):
    """Interpolate eigenvalue at target_index (1-indexed) from SLQ density.
    If return_bounds=True, also returns (lo, hi) eigenvalues of the two bracketing SLQ buckets."""
    nan_result = (np.nan, np.nan, np.nan) if return_bounds else np.nan
    if len(evs) == 0 or len(wts) == 0:
        return nan_result
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
        return nan_result
    if target_index < indices[0] or target_index > indices[-1]:
        return nan_result
    log_result = np.interp(np.log(target_index), np.log(indices), np.log(evs_pos))
    val = np.exp(log_result)
    if not return_bounds:
        return val
    # Find bracketing buckets
    j = np.searchsorted(indices, target_index)  # indices is descending in eigenvalue, ascending in index
    j = np.clip(j, 1, len(indices) - 1)
    hi = evs_pos[j - 1]  # higher eigenvalue (lower rank index)
    lo = evs_pos[j]      # lower eigenvalue (higher rank index)
    return val, lo, hi


def get_pos_eigenvalue(hessian_data, rank, return_bounds=False):
    """Get positive eigenvalue at given rank. Lanczos for rank<=10, SLQ otherwise.
    If return_bounds=True, returns (val, lo, hi). Lanczos has no bounds (lo=hi=val)."""
    if rank <= 10:
        lanczos_evs = sorted(hessian_data.get("lanczos", {}).get("eigenvalues", []), reverse=True)
        idx = rank - 1
        if idx < len(lanczos_evs):
            v = lanczos_evs[idx]
            return (v, v, v) if return_bounds else v
        return (np.nan, np.nan, np.nan) if return_bounds else np.nan
    else:
        slq_evs = hessian_data.get("slq", {}).get("raw_eigenvalues", [])
        slq_wts = hessian_data.get("slq", {}).get("raw_weights", [])
        total_params = hessian_data.get("config", {}).get("total_params")
        if not slq_evs or not slq_wts or total_params is None:
            return (np.nan, np.nan, np.nan) if return_bounds else np.nan
        return slq_interp_eigenvalue(slq_evs, slq_wts, total_params, rank, return_bounds=return_bounds)


def slq_interp_eigenvalue_neg(evs, wts, total_params, target_index, return_bounds=False):
    """Interpolate negative eigenvalue at target_index (1-indexed, 1=most negative) from SLQ density.
    Ranks within the negative portion only."""
    nan_result = (np.nan, np.nan, np.nan) if return_bounds else np.nan
    if len(evs) == 0 or len(wts) == 0:
        return nan_result
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
        return nan_result
    # Rank within negative portion: renormalize weights to sum to total negative count
    neg_count = wts_neg.sum() * total_params
    wts_neg_norm = wts_neg / wts_neg.sum()
    cum = np.cumsum(wts_neg_norm)
    indices = (cum - 0.5 * wts_neg_norm) * neg_count
    if target_index < indices[0] or target_index > indices[-1]:
        return nan_result
    # Interpolate in log-log on magnitudes
    log_abs = np.log(np.abs(evs_neg))
    log_result = np.interp(np.log(target_index), np.log(indices), log_abs)
    val = -np.exp(log_result)
    if not return_bounds:
        return val
    j = np.searchsorted(indices, target_index)
    j = np.clip(j, 1, len(indices) - 1)
    hi_abs = np.abs(evs_neg[j - 1])
    lo_abs = np.abs(evs_neg[j])
    return val, lo_abs, hi_abs


def get_neg_eigenvalue(hessian_data, rank, return_bounds=False):
    """Get negative eigenvalue at given rank (1=most negative).
    Lanczos for rank<=10 if available, falls back to SLQ on -H density.
    Returns magnitude (positive). If return_bounds=True, returns (val, lo, hi) all as magnitudes."""
    nan_result = (np.nan, np.nan, np.nan) if return_bounds else np.nan
    if rank <= 10:
        neg_evs = sorted(hessian_data.get("lanczos_bottom", {}).get("eigenvalues", []))
        idx = rank - 1
        if idx < len(neg_evs):
            v = abs(neg_evs[idx])
            return (v, v, v) if return_bounds else v
        # Fall back to SLQ if Lanczos didn't find enough negative eigenvalues
    total_params = hessian_data.get("config", {}).get("total_params")
    # Use dedicated -H SLQ density if available
    slq_bottom = hessian_data.get("slq_bottom", {})
    slq_neg_evs = slq_bottom.get("raw_eigenvalues", [])
    slq_neg_wts = slq_bottom.get("raw_weights", [])
    if slq_neg_evs and slq_neg_wts and total_params:
        neg_only = [ev for ev in slq_neg_evs if ev < -1e-10]
        neg_wts = [slq_neg_wts[i] for i, ev in enumerate(slq_neg_evs) if ev < -1e-10]
        if len(neg_only) >= 2:
            result = slq_interp_eigenvalue_neg(neg_only, neg_wts, total_params, rank, return_bounds=return_bounds)
            if return_bounds:
                return (abs(result[0]), result[1], result[2])
            return abs(result) if not np.isnan(result) else np.nan
    # Fall back to original SLQ density
    slq_evs = hessian_data.get("slq", {}).get("raw_eigenvalues", [])
    slq_wts = hessian_data.get("slq", {}).get("raw_weights", [])
    if not slq_evs or not slq_wts or total_params is None:
        return nan_result
    result = slq_interp_eigenvalue_neg(slq_evs, slq_wts, total_params, rank, return_bounds=return_bounds)
    if return_bounds:
        return (abs(result[0]), result[1], result[2])
    return abs(result) if not np.isnan(result) else np.nan


def load_run(output_dir):
    """Load all hessian_step_*.json from a run directory."""
    pattern = os.path.join(output_dir, "hessian_step_*.json")
    files = glob.glob(pattern)
    data_by_step = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        step = int(filename.replace("hessian_step_", "").replace(".json", ""))
        with open(filepath) as f:
            data_by_step[step] = json.load(f)
    return data_by_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--lrs", type=float, nargs="+", default=None, help="Filter to these LRs only")
    parser.add_argument("--model-id", type=str, default="pythia70m")
    parser.add_argument("--dataset-name", type=str, default="tinystories")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--x-axis", type=str, default="step", choices=["step", "step_x_lr"],
                        help="X-axis: 'step' or 'step_x_lr' (step * learning_rate)")
    args = parser.parse_args()

    # Find all runs for this optimizer
    base_dir = "outputs/run_pipeline"
    pattern = os.path.join(base_dir, f"{args.model_id}_{args.dataset_name}_bs{args.batch_size}_lr*_opt{args.optimizer}_constant")
    run_dirs = sorted(glob.glob(pattern))

    if not run_dirs:
        print(f"No runs found for optimizer={args.optimizer}")
        return

    # Load all runs, extract LR from directory name
    runs = {}
    for d in run_dirs:
        dirname = os.path.basename(d)
        # Extract LR from dirname: ...bs256_lr{LR}_opt...
        lr_str = dirname.split("_lr")[1].split("_opt")[0]
        lr = float(lr_str)
        if args.lrs and not any(abs(lr - l) / max(lr, 1e-10) < 0.01 for l in args.lrs):
            continue
        data = load_run(d)
        if data:
            runs[lr] = data
            print(f"Loaded lr={lr}: {sorted(data.keys())}")

    if not runs:
        print("No data loaded")
        return

    # Load loss histories
    loss_histories = {}
    for lr in sorted(runs.keys()):
        base_dir_run = "outputs/run_pipeline"
        loss_path = os.path.join(base_dir_run,
            f"{args.model_id}_{args.dataset_name}_bs{args.batch_size}_lr{lr}_opt{args.optimizer}_constant",
            "loss_history.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_histories[lr] = json.load(f)

    # Plot: 3 rows (positive, negative, loss) x n_ranks columns
    n_ranks = len(args.ranks)
    fig, axes = plt.subplots(3, n_ranks, figsize=(5 * n_ranks, 13))
    if n_ranks == 1:
        axes = axes.reshape(3, 1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(runs)))
    sorted_lrs = sorted(runs.keys())

    x_scale = args.x_axis  # "step" or "step_x_lr"
    x_label = "Training Step" if x_scale == "step" else r"Step $\times$ LR"

    def make_plot_steps(steps, lr):
        if x_scale == "step_x_lr":
            return [0.5 * lr if s == 0 else s * lr for s in steps]
        return [0.5 if s == 0 else s for s in steps]

    for rank_idx, rank in enumerate(args.ranks):
        # Top row: positive eigenvalues
        ax_pos = axes[0, rank_idx]
        for lr_idx, lr in enumerate(sorted_lrs):
            data_by_step = runs[lr]
            steps = sorted(data_by_step.keys())
            plot_steps = make_plot_steps(steps, lr)
            results = [get_pos_eigenvalue(data_by_step[s], rank, return_bounds=True) for s in steps]
            vals = [r[0] for r in results]
            los = [r[1] for r in results]
            his = [r[2] for r in results]

            ax_pos.loglog(plot_steps, vals, marker='o', markersize=4, color=colors[lr_idx],
                          label=f'lr={lr}')
            if rank > 10:
                ax_pos.fill_between(plot_steps, los, his, color=colors[lr_idx], alpha=0.15)

        if rank_idx == 0:
            ax_pos.set_ylabel("Eigenvalue")
        ax_pos.set_title(f"Rank {rank}")
        ax_pos.grid(True, which="both", alpha=0.3)
        ax_pos.legend(fontsize=7, loc="best")

        # Bottom row: negative eigenvalues (plotted as magnitude)
        ax_neg = axes[1, rank_idx]
        for lr_idx, lr in enumerate(sorted_lrs):
            data_by_step = runs[lr]
            steps = sorted(data_by_step.keys())
            plot_steps = make_plot_steps(steps, lr)
            results = [get_neg_eigenvalue(data_by_step[s], rank, return_bounds=True) for s in steps]
            vals = [r[0] for r in results]
            los = [r[1] for r in results]
            his = [r[2] for r in results]

            ax_neg.loglog(plot_steps, vals, marker='s', markersize=4, color=colors[lr_idx],
                          label=f'lr={lr}')
            if rank > 10:
                ax_neg.fill_between(plot_steps, los, his, color=colors[lr_idx], alpha=0.15)

        ax_neg.set_xlabel(x_label)
        if rank_idx == 0:
            ax_neg.set_ylabel(r"$|\lambda|$")
        ax_neg.grid(True, which="both", alpha=0.3)
        ax_neg.legend(fontsize=7, loc="best")

    # Fixed y-limits for eigenvalue rows
    for ax in axes[0, :].flat:
        ax.set_ylim(1, 1000)
    for ax in axes[1, :].flat:
        ax.set_ylim(1, 1000)

    axes[0, 0].annotate("Positive", xy=(0, 0.5), xytext=(-50, 0),
                         xycoords='axes fraction', textcoords='offset points',
                         fontsize=12, ha='right', va='center', rotation=90)
    axes[1, 0].annotate("Negative (magnitude)", xy=(0, 0.5), xytext=(-50, 0),
                         xycoords='axes fraction', textcoords='offset points',
                         fontsize=12, ha='right', va='center', rotation=90)

    # Loss curves in last column of row 3, hide others
    for c in range(n_ranks - 1):
        axes[2, c].set_visible(False)
    ax_loss = axes[2, n_ranks - 1]
    for lr_idx, lr in enumerate(sorted_lrs):
        if lr in loss_histories:
            eval_losses = loss_histories[lr].get("eval_losses", [])
            if eval_losses:
                steps_l = [x["step"] for x in eval_losses if x["step"] > 0]
                vals_l = [x["loss"] for x in eval_losses if x["step"] > 0]
                if x_scale == "step_x_lr":
                    steps_l = [s * lr for s in steps_l]
                ax_loss.loglog(steps_l, vals_l, marker='o', markersize=4, color=colors[lr_idx],
                             label=f'lr={lr}')
    ax_loss.set_xlabel("Training Step")
    ax_loss.set_ylabel("Eval Loss")
    ax_loss.set_title("Loss Curves")
    ax_loss.legend(fontsize=7, loc="best")
    ax_loss.grid(True, alpha=0.3)

    fig.suptitle(f"{args.optimizer.upper()} — Eigenvalue Evolution by Rank", fontsize=13)
    plt.tight_layout()

    suffix = f"_{x_scale}" if x_scale != "step" else ""
    save_path = args.output or f"outputs/eigen_vs_lr_{args.model_id}_{args.optimizer}{suffix}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
