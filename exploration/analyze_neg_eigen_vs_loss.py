"""
Plot rank-64 negative eigenvalue magnitude vs final eval loss.
3 panels: Pythia+Adam, Pythia+SGD, TinyStories+SGD.
Each point is one LR. Missing negative eigenvalues are set to 0.01.

Usage:
    python analyze_neg_eigen_vs_loss.py
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def slq_interp_eigenvalue_neg(evs, wts, total_params, target_index):
    if len(evs) == 0 or len(wts) == 0:
        return np.nan
    evs = np.array(evs, dtype=float)
    wts = np.array(wts, dtype=float)
    idx = np.argsort(evs)
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]
    wts_sorted = wts_sorted / wts_sorted.sum()
    mask = evs_sorted < -1e-10
    evs_neg = evs_sorted[mask]
    wts_neg = wts_sorted[mask]
    if len(evs_neg) < 2:
        return np.nan
    neg_count = wts_neg.sum() * total_params
    wts_neg_norm = wts_neg / wts_neg.sum()
    cum = np.cumsum(wts_neg_norm)
    indices = (cum - 0.5 * wts_neg_norm) * neg_count
    if target_index < indices[0] or target_index > indices[-1]:
        return np.nan
    log_abs = np.log(np.abs(evs_neg))
    log_result = np.interp(np.log(target_index), np.log(indices), log_abs)
    return -np.exp(log_result)


def get_neg_eigenvalue(hessian_data, rank):
    if rank <= 10:
        neg_evs = sorted(hessian_data.get("lanczos_bottom", {}).get("eigenvalues", []))
        idx = rank - 1
        if idx < len(neg_evs):
            return neg_evs[idx]
    # Fall back to -H SLQ density
    slq_bottom = hessian_data.get("slq_bottom", {})
    slq_neg_evs = slq_bottom.get("raw_eigenvalues", [])
    slq_neg_wts = slq_bottom.get("raw_weights", [])
    total_params = hessian_data.get("config", {}).get("total_params")
    if slq_neg_evs and slq_neg_wts and total_params:
        neg_only = [ev for ev in slq_neg_evs if ev < -1e-10]
        neg_wts = [slq_neg_wts[i] for i, ev in enumerate(slq_neg_evs) if ev < -1e-10]
        if len(neg_only) >= 2:
            return slq_interp_eigenvalue_neg(neg_only, neg_wts, total_params, rank)
    # Fall back to original SLQ
    slq_evs = hessian_data.get("slq", {}).get("raw_eigenvalues", [])
    slq_wts = hessian_data.get("slq", {}).get("raw_weights", [])
    if slq_evs and slq_wts and total_params:
        return slq_interp_eigenvalue_neg(slq_evs, slq_wts, total_params, rank)
    return np.nan


def slq_interp_eigenvalue(evs, wts, total_params, target_index):
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
    log_result = np.interp(np.log(target_index), np.log(indices), np.log(evs_pos))
    return np.exp(log_result)


def get_pos_eigenvalue(hessian_data, rank):
    if rank <= 10:
        lanczos_evs = sorted(hessian_data.get("lanczos", {}).get("eigenvalues", []), reverse=True)
        idx = rank - 1
        if idx < len(lanczos_evs):
            return lanczos_evs[idx]
        return np.nan
    slq_evs = hessian_data.get("slq", {}).get("raw_eigenvalues", [])
    slq_wts = hessian_data.get("slq", {}).get("raw_weights", [])
    total_params = hessian_data.get("config", {}).get("total_params")
    if not slq_evs or not slq_wts or total_params is None:
        return np.nan
    return slq_interp_eigenvalue(slq_evs, slq_wts, total_params, rank)


def load_panel_data(model_id, dataset_name, batch_size, optimizer):
    """Load final loss, rank-64 positive and negative eigenvalue for each LR."""
    base_dir = "outputs/run_pipeline"
    pattern = os.path.join(base_dir,
        f"{model_id}_{dataset_name}_bs{batch_size}_lr*_opt{optimizer}_constant")
    run_dirs = sorted(glob.glob(pattern))

    results = []
    for d in run_dirs:
        dirname = os.path.basename(d)
        lr_str = dirname.split("_lr")[1].split("_opt")[0]
        lr = float(lr_str)

        # Get final loss
        loss_file = os.path.join(d, "loss_history.json")
        if not os.path.exists(loss_file):
            continue
        with open(loss_file) as f:
            loss_data = json.load(f)
        eval_losses = loss_data.get("eval_losses", [])
        if not eval_losses:
            continue
        final_loss = eval_losses[-1]["loss"]
        if np.isnan(final_loss):
            continue

        # Get final hessian step
        hessian_files = glob.glob(os.path.join(d, "hessian_step_*.json"))
        if not hessian_files:
            continue
        last_step = max(int(os.path.basename(f).replace("hessian_step_", "").replace(".json", ""))
                        for f in hessian_files)
        with open(os.path.join(d, f"hessian_step_{last_step}.json")) as f:
            hessian_data = json.load(f)

        pos_ev = get_pos_eigenvalue(hessian_data, 64)
        pos_val = pos_ev if not np.isnan(pos_ev) else 0.01

        neg_ev = get_neg_eigenvalue(hessian_data, 64)
        neg_mag = abs(neg_ev) if not np.isnan(neg_ev) else 0.01

        results.append((lr, final_loss, pos_val, neg_mag))

    return results


def main():
    panels = [
        ("Pythia + Adam", "pythia70m", "tinystories", 256, "adam"),
        ("Pythia + SGD", "pythia70m", "tinystories", 256, "sgd"),
        ("TinyStories + SGD", "tinystories", "tinystories", 256, "sgd"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col, (title, model_id, dataset_name, batch_size, optimizer) in enumerate(panels):
        data = load_panel_data(model_id, dataset_name, batch_size, optimizer)
        if not data:
            axes[0, col].set_title(f"{title}\n(no data)")
            axes[1, col].set_title(f"{title}\n(no data)")
            continue

        lrs, losses, pos_vals, neg_mags = zip(*data)

        # Top row: positive eigenvalues
        ax_pos = axes[0, col]
        ax_pos.scatter(losses, pos_vals, c=np.log10(lrs), cmap='viridis',
                       s=80, edgecolors='k', linewidths=0.5, zorder=5)
        for lr, loss, pos in zip(lrs, losses, pos_vals):
            ax_pos.annotate(f'{lr}', (loss, pos), fontsize=6, ha='left',
                            xytext=(5, 3), textcoords='offset points')
        ax_pos.set_ylabel(r"$\lambda_{64}^+$")
        ax_pos.set_title(title)
        ax_pos.set_yscale('log')
        ax_pos.grid(True, alpha=0.3)
        ax_pos.axhline(0.01, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Bottom row: negative eigenvalues
        ax_neg = axes[1, col]
        ax_neg.scatter(losses, neg_mags, c=np.log10(lrs), cmap='viridis',
                       s=80, edgecolors='k', linewidths=0.5, zorder=5)
        for lr, loss, neg in zip(lrs, losses, neg_mags):
            ax_neg.annotate(f'{lr}', (loss, neg), fontsize=6, ha='left',
                            xytext=(5, 3), textcoords='offset points')
        ax_neg.set_xlabel("Final Eval Loss")
        ax_neg.set_ylabel(r"$|\lambda_{64}^-|$")
        ax_neg.set_yscale('log')
        ax_neg.grid(True, alpha=0.3)
        ax_neg.axhline(0.01, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    save_path = "outputs/eigen64_vs_loss.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
