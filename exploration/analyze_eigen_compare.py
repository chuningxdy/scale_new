"""
Compare eigenvalue geomean across optimizers on the same plot.
Solid lines for first optimizer, dotted for second.

Usage:
    python analyze_eigen_compare.py --opt1 adam --opt2 signsgd --lr-scheduler-type constant
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


def get_pos_eigenvalue(hessian_data, rank):
    if rank <= 10:
        lanczos_evs = sorted(hessian_data.get("lanczos", {}).get("eigenvalues", []), reverse=True)
        if rank - 1 < len(lanczos_evs):
            return lanczos_evs[rank - 1]
    return np.nan


def get_neg_eigenvalue(hessian_data, rank):
    if rank <= 10:
        neg_evs = sorted(hessian_data.get("lanczos_bottom", {}).get("eigenvalues", []))
        if rank - 1 < len(neg_evs):
            return abs(neg_evs[rank - 1])
    return np.nan


def get_geomean(hessian_data, rank_lo, rank_hi, sign="pos"):
    product = 1.0
    n = rank_hi - rank_lo + 1
    for r in range(rank_lo, rank_hi + 1):
        ev = get_pos_eigenvalue(hessian_data, r) if sign == "pos" else get_neg_eigenvalue(hessian_data, r)
        if np.isnan(ev):
            return np.nan
        product *= ev
    return product ** (1.0 / n)


def find_runs(base_dir, model_id, dataset_name, batch_size, optimizer, lr_scheduler_type, lrs=None):
    pattern = os.path.join(base_dir,
        f"{model_id}_{dataset_name}_bs{batch_size}_lr*_opt{optimizer}_{lr_scheduler_type}")
    run_dirs = sorted(glob.glob(pattern))
    runs = {}
    loss_histories = {}
    for d in run_dirs:
        dirname = os.path.basename(d)
        lr_str = dirname.split("_lr")[1].split("_opt")[0]
        lr = float(lr_str)
        if lrs and not any(abs(lr - l) / max(lr, 1e-10) < 0.01 for l in lrs):
            continue
        data = load_run(d)
        if data:
            runs[lr] = data
        loss_path = os.path.join(d, "loss_history.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_histories[lr] = json.load(f)
    return runs, loss_histories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt1", type=str, default="adam", help="First optimizer (solid lines)")
    parser.add_argument("--opt2", type=str, default="signsgd", help="Second optimizer (dotted lines)")
    parser.add_argument("--model-id", type=str, default="pythia70m")
    parser.add_argument("--dataset-name", type=str, default="tinystories")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lrs", type=float, nargs="+", default=None)
    parser.add_argument("--rank-range", type=int, nargs=2, default=[5, 10])
    parser.add_argument("--x-axis", type=str, default="step_x_lr", choices=["step", "step_x_lr"])
    parser.add_argument("--cross-section", type=float, nargs="+", default=None)
    parser.add_argument("--lr-scheduler-type", type=str, default="constant")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rank_lo, rank_hi = args.rank_range
    base_dir = "outputs/run_pipeline"

    runs1, loss1 = find_runs(base_dir, args.model_id, args.dataset_name, args.batch_size,
                              args.opt1, args.lr_scheduler_type, args.lrs)
    runs2, loss2 = find_runs(base_dir, args.model_id, args.dataset_name, args.batch_size,
                              args.opt2, args.lr_scheduler_type, args.lrs)

    all_lrs = sorted(set(list(runs1.keys()) + list(runs2.keys())))
    if not all_lrs:
        print("No runs found")
        return
    print(f"{args.opt1}: {sorted(runs1.keys())}")
    print(f"{args.opt2}: {sorted(runs2.keys())}")

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_lrs)))
    lr_to_color = {lr: colors[i] for i, lr in enumerate(all_lrs)}

    def make_plot_steps(steps, lr):
        if args.x_axis == "step_x_lr":
            return [0.5 * lr if s == 0 else s * lr for s in steps]
        return [0.5 if s == 0 else s for s in steps]

    x_label = "Training Step" if args.x_axis == "step" else r"Step $\times$ LR"
    label = f"GM$_{{i={rank_lo}}}^{{{rank_hi}}}$"

    n_rows = 3
    if args.cross_section is not None:
        n_rows += 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4.3 * n_rows))
    ax_pos, ax_neg, ax_loss = axes[0], axes[1], axes[2]

    # Plot helper
    def plot_opt(runs, loss_hist, optimizer, linestyle, marker):
        for lr in sorted(runs.keys()):
            data_by_step = runs[lr]
            steps = sorted(data_by_step.keys())
            plot_steps = make_plot_steps(steps, lr)

            vals_pos = [get_geomean(data_by_step[s], rank_lo, rank_hi, "pos") for s in steps]
            ax_pos.loglog(plot_steps, vals_pos, marker=marker, markersize=4,
                          color=lr_to_color[lr], linestyle=linestyle,
                          label=f'{optimizer} lr={lr}')

            vals_neg = [get_geomean(data_by_step[s], rank_lo, rank_hi, "neg") for s in steps]
            ax_neg.loglog(plot_steps, vals_neg, marker=marker, markersize=4,
                          color=lr_to_color[lr], linestyle=linestyle,
                          label=f'{optimizer} lr={lr}')

            if lr in loss_hist:
                eval_losses = loss_hist[lr].get("eval_losses", [])
                if eval_losses:
                    steps_l = [x["step"] for x in eval_losses if x["step"] > 0]
                    vals_l = [x["loss"] for x in eval_losses if x["step"] > 0]
                    if args.x_axis == "step_x_lr":
                        steps_l = [s * lr for s in steps_l]
                    ax_loss.loglog(steps_l, vals_l, marker=marker, markersize=4,
                                   color=lr_to_color[lr], linestyle=linestyle,
                                   label=f'{optimizer} lr={lr}')

    plot_opt(runs1, loss1, args.opt1, '-', 'o')
    plot_opt(runs2, loss2, args.opt2, '--', 's')

    ax_pos.set_ylabel(f"{label} $\\lambda_i^+$")
    ax_pos.set_title(f"Positive Eigenvalue Geometric Mean (rank {rank_lo}-{rank_hi})")
    ax_pos.legend(fontsize=5, loc="best", ncol=2)
    ax_pos.grid(True, which="both", alpha=0.3)

    ax_neg.set_ylabel(f"{label} $|\\lambda_i^-|$")
    ax_neg.set_title(f"Negative Eigenvalue Geometric Mean (rank {rank_lo}-{rank_hi})")
    ax_neg.legend(fontsize=5, loc="best", ncol=2)
    ax_neg.grid(True, which="both", alpha=0.3)

    ax_loss.set_xlabel(x_label)
    ax_loss.set_ylabel("Eval Loss")
    ax_loss.set_title("Loss Curves")
    ax_loss.legend(fontsize=5, loc="best", ncol=2)
    ax_loss.grid(True, alpha=0.3)

    # Cross-section panels
    if args.cross_section is not None:
        log_cs = np.log10(args.cross_section)
        cs_norm = (log_cs - log_cs.min()) / (log_cs.max() - log_cs.min() + 1e-10)
        cs_colors = plt.cm.plasma(cs_norm)

        for panel_idx, sign in enumerate(["pos", "neg"]):
            ax_cs = axes[3 + panel_idx]
            for ci, target_slr in enumerate(args.cross_section):
                for runs_dict, optimizer, linestyle, marker in [
                    (runs1, args.opt1, '-', 'o'), (runs2, args.opt2, '--', 's')]:
                    cs_lrs = []
                    cs_vals = []
                    for lr in sorted(runs_dict.keys()):
                        data_by_step = runs_dict[lr]
                        steps = sorted(data_by_step.keys())
                        slr_vals = [s * lr for s in steps]
                        products = [get_geomean(data_by_step[s], rank_lo, rank_hi, sign) for s in steps]
                        valid = [(x, y) for x, y in zip(slr_vals, products)
                                 if not np.isnan(y) and y > 0 and x > 0]
                        if len(valid) < 2:
                            continue
                        xs, ys = zip(*valid)
                        if target_slr < min(xs) or target_slr > max(xs):
                            continue
                        log_interp = np.interp(np.log(target_slr), np.log(xs), np.log(ys))
                        cs_lrs.append(lr)
                        cs_vals.append(np.exp(log_interp))
                    if cs_lrs:
                        ax_cs.loglog(cs_lrs, cs_vals, marker=marker, markersize=8,
                                     color=cs_colors[ci], linestyle=linestyle,
                                     linewidth=2, zorder=5,
                                     label=f'{optimizer} t·lr={target_slr}')

            ylabel = f"{label} $\\lambda_i^+$" if sign == "pos" else f"{label} $|\\lambda_i^-|$"
            title = f"{'Positive' if sign == 'pos' else 'Negative'} Geo. Mean vs LR"
            ax_cs.set_xlabel("Learning Rate")
            ax_cs.set_ylabel(ylabel)
            ax_cs.set_title(title)
            ax_cs.legend(fontsize=5, loc="best", ncol=2)
            ax_cs.grid(True, which="both", alpha=0.3)

    sched_suffix = f"_{args.lr_scheduler_type}" if args.lr_scheduler_type != "constant" else ""
    suffix = f"_{args.x_axis}" if args.x_axis != "step" else ""
    fig.suptitle(f"{args.opt1.upper()} vs {args.opt2.upper()} — Eigenvalue Geometric Mean (rank {rank_lo}-{rank_hi})", fontsize=13)
    plt.tight_layout()

    save_path = args.output or f"outputs/eigen_geomean_compare_{args.opt1}_{args.opt2}{sched_suffix}{suffix}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
