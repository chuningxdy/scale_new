"""
Plot product of eigenvalues (partial determinant) vs training step across LRs.
Top row: product of positive eigenvalues at specified ranks.
Bottom row: product of |negative eigenvalues| at specified ranks.
If any eigenvalue in the product is missing, the product is NaN.

Usage:
    python analyze_eigen_product.py --optimizer adam --model-id pythia70m --rank-range 2 5
    python analyze_eigen_product.py --optimizer sgd --model-id tinystories --rank-range 1 4 --x-axis step_x_lr
"""

import argparse
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml


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


def build_cumulative_lr_fn(run_dir, max_lr):
    """Return a function step -> sum_{k=0}^{step-1} lr(k).

    lr(k) follows the schedule used by HuggingFace Trainer (matches
    run_pipeline.py): linear warmup for `warmup_steps`, then either
    constant, cosine (num_cycles=0.5), or linear decay to 0 over
    `total_steps`. Uses 0.5 * lr(0) at step 0 so values work on a
    log x-axis.
    """
    cfg_path = os.path.join(run_dir, "config.yaml")
    total_steps = None
    warmup_steps = 0
    scheduler = "constant"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        total_steps = cfg.get("total_steps")
        warmup_steps = int(cfg.get("warmup_steps", 0) or 0)
        scheduler = cfg.get("lr_scheduler_type", "constant")

    def factors(n):
        """Factor array of length n for steps 0..n-1."""
        if n <= 0:
            return np.zeros(0)
        ks = np.arange(n)
        if scheduler == "constant" or total_steps is None:
            return np.ones(n)
        f = np.ones(n)
        if warmup_steps > 0:
            warm = ks < warmup_steps
            f[warm] = ks[warm] / float(warmup_steps)
            post_ks = ks[~warm]
        else:
            post_ks = ks
        denom = max(1, total_steps - warmup_steps)
        progress = (post_ks - warmup_steps) / float(denom)
        if scheduler == "cosine":
            post_vals = np.maximum(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        elif scheduler == "linear":
            post_vals = np.maximum(0.0, 1.0 - progress)
        else:
            post_vals = np.ones_like(progress)
        if warmup_steps > 0:
            f[ks >= warmup_steps] = post_vals
        else:
            f = post_vals
        return f

    def cumulative(step):
        if step <= 0:
            f0 = factors(1)
            f0_val = float(f0[0]) if len(f0) else 1.0
            return 0.5 * max_lr * f0_val
        return float(max_lr * factors(step).sum())

    return cumulative


def get_pos_eigenvalue(hessian_data, rank):
    if rank <= 10:
        lanczos_evs = sorted(hessian_data.get("lanczos", {}).get("eigenvalues", []), reverse=True)
        if rank - 1 < len(lanczos_evs):
            return lanczos_evs[rank - 1]
        return np.nan
    # SLQ fallback
    slq_evs = hessian_data.get("slq", {}).get("raw_eigenvalues", [])
    slq_wts = hessian_data.get("slq", {}).get("raw_weights", [])
    total_params = hessian_data.get("config", {}).get("total_params")
    if not slq_evs or not slq_wts or total_params is None:
        return np.nan
    return _slq_interp(slq_evs, slq_wts, total_params, rank)


def get_neg_eigenvalue(hessian_data, rank):
    """Returns magnitude (positive value) of the rank-th most negative eigenvalue."""
    if rank <= 10:
        neg_evs = sorted(hessian_data.get("lanczos_bottom", {}).get("eigenvalues", []))
        if rank - 1 < len(neg_evs):
            return abs(neg_evs[rank - 1])
    # SLQ fallback - try -H density first
    slq_bottom = hessian_data.get("slq_bottom", {})
    slq_neg_evs = slq_bottom.get("raw_eigenvalues", [])
    slq_neg_wts = slq_bottom.get("raw_weights", [])
    total_params = hessian_data.get("config", {}).get("total_params")
    if slq_neg_evs and slq_neg_wts and total_params:
        neg_only = [ev for ev in slq_neg_evs if ev < -1e-10]
        neg_wts = [slq_neg_wts[i] for i, ev in enumerate(slq_neg_evs) if ev < -1e-10]
        if len(neg_only) >= 2:
            val = _slq_interp_neg(neg_only, neg_wts, total_params, rank)
            if not np.isnan(val):
                return abs(val)
    # Fall back to original SLQ
    slq_evs = hessian_data.get("slq", {}).get("raw_eigenvalues", [])
    slq_wts = hessian_data.get("slq", {}).get("raw_weights", [])
    if slq_evs and slq_wts and total_params:
        val = _slq_interp_neg(slq_evs, slq_wts, total_params, rank)
        if not np.isnan(val):
            return abs(val)
    return np.nan


def _slq_interp(evs, wts, total_params, target_index):
    evs = np.array(evs, dtype=float)
    wts = np.array(wts, dtype=float)
    idx = np.argsort(evs)[::-1]
    evs_sorted = evs[idx]
    wts_sorted = wts[idx] / wts[idx].sum()
    cum = np.cumsum(wts_sorted)
    indices = (cum - 0.5 * wts_sorted) * total_params
    mask = evs_sorted > 1e-10
    evs_pos = evs_sorted[mask]
    ind = indices[mask]
    if len(evs_pos) < 2 or target_index < ind[0] or target_index > ind[-1]:
        return np.nan
    return np.exp(np.interp(np.log(target_index), np.log(ind), np.log(evs_pos)))


def _slq_interp_neg(evs, wts, total_params, target_index):
    evs = np.array(evs, dtype=float)
    wts = np.array(wts, dtype=float)
    idx = np.argsort(evs)
    evs_sorted = evs[idx]
    wts_sorted = wts[idx] / wts[idx].sum()
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
    log_result = np.interp(np.log(target_index), np.log(indices), np.log(np.abs(evs_neg)))
    return -np.exp(log_result)


def get_product(hessian_data, rank_lo, rank_hi, sign="pos"):
    """Geometric mean of eigenvalues from rank_lo to rank_hi (inclusive). NaN if any missing."""
    product = 1.0
    n = rank_hi - rank_lo + 1
    for r in range(rank_lo, rank_hi + 1):
        if sign == "pos":
            ev = get_pos_eigenvalue(hessian_data, r)
        else:
            ev = get_neg_eigenvalue(hessian_data, r)
        if np.isnan(ev):
            return np.nan
        product *= ev
    return product ** (1.0 / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--model-id", type=str, default="pythia70m")
    parser.add_argument("--dataset-name", type=str, default="tinystories")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lrs", type=float, nargs="+", default=None)
    parser.add_argument("--rank-range", type=int, nargs=2, default=[2, 5],
                        help="Rank range for product (inclusive), e.g. --rank-range 2 5")
    parser.add_argument("--x-axis", type=str, default="step", choices=["step", "step_x_lr"])
    parser.add_argument("--cross-section", type=float, nargs="+", default=None,
                        help="Add panel showing product vs LR at cumulative-LR = these values, e.g. --cross-section 1.0 10.0")
    parser.add_argument("--lr-scheduler-type", type=str, default="constant")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rank_lo, rank_hi = args.rank_range

    base_dir = "outputs/run_pipeline"
    pattern = os.path.join(base_dir,
        f"{args.model_id}_{args.dataset_name}_bs{args.batch_size}_lr*_opt{args.optimizer}_{args.lr_scheduler_type}")
    run_dirs = sorted(glob.glob(pattern))

    runs = {}
    cum_lr_fns = {}
    for d in run_dirs:
        dirname = os.path.basename(d)
        lr_str = dirname.split("_lr")[1].split("_opt")[0]
        lr = float(lr_str)
        if args.lrs and not any(abs(lr - l) / max(lr, 1e-10) < 0.01 for l in args.lrs):
            continue
        data = load_run(d)
        if data:
            runs[lr] = data
            cum_lr_fns[lr] = build_cumulative_lr_fn(d, lr)
            print(f"Loaded lr={lr}: {len(data)} steps")

    if not runs:
        print("No runs found")
        return

    # Load loss histories
    loss_histories = {}
    for lr in sorted(runs.keys()):
        loss_path = os.path.join(base_dir,
            f"{args.model_id}_{args.dataset_name}_bs{args.batch_size}_lr{lr}_opt{args.optimizer}_{args.lr_scheduler_type}",
            "loss_history.json")
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                loss_histories[lr] = json.load(f)

    sorted_lrs = sorted(runs.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_lrs)))

    def make_plot_steps(steps, lr):
        if args.x_axis == "step_x_lr":
            fn = cum_lr_fns[lr]
            return [fn(s) for s in steps]
        return [0.5 if s == 0 else s for s in steps]

    x_label = "Training Step" if args.x_axis == "step" else r"$\sum_{k} \eta_k$ (cumulative LR)"
    label = f"GM$_{{i={rank_lo}}}^{{{rank_hi}}}$"

    n_rows = 3  # pos, neg, loss
    if args.cross_section is not None:
        n_rows += 5  # pos vs lr, neg vs lr, train-loss vs lr, eval-loss vs lr, gen-gap vs lr
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4.3 * n_rows))
    ax_pos, ax_neg, ax_loss = axes[0], axes[1], axes[2]

    # Top: product of positive eigenvalues
    for i, lr in enumerate(sorted_lrs):
        data_by_step = runs[lr]
        steps = sorted(data_by_step.keys())
        plot_steps = make_plot_steps(steps, lr)
        vals = [get_product(data_by_step[s], rank_lo, rank_hi, "pos") for s in steps]
        ax_pos.loglog(plot_steps, vals, marker='o', markersize=4, color=colors[i], label=f'lr={lr}')

    ax_pos.set_ylabel(f"{label} $\\lambda_i^+$")
    ax_pos.set_title(f"Geometric Mean of Positive Eigenvalues (rank {rank_lo}-{rank_hi})")
    ax_pos.legend(fontsize=7, loc="best")
    ax_pos.grid(True, which="both", alpha=0.3)

    # Middle: product of |negative eigenvalues|
    for i, lr in enumerate(sorted_lrs):
        data_by_step = runs[lr]
        steps = sorted(data_by_step.keys())
        plot_steps = make_plot_steps(steps, lr)
        vals = [get_product(data_by_step[s], rank_lo, rank_hi, "neg") for s in steps]
        ax_neg.loglog(plot_steps, vals, marker='s', markersize=4, color=colors[i], label=f'lr={lr}')

    ax_neg.set_ylabel(f"{label} $|\\lambda_i^-|$")
    ax_neg.set_title(f"Geometric Mean of |Negative Eigenvalues| (rank {rank_lo}-{rank_hi})")
    ax_neg.legend(fontsize=7, loc="best")
    ax_neg.grid(True, which="both", alpha=0.3)

    # Bottom: loss curves (train = dashed, eval = solid)
    for i, lr in enumerate(sorted_lrs):
        if lr not in loss_histories:
            continue
        train_losses = loss_histories[lr].get("train_losses", [])
        if train_losses:
            steps_t = [x["step"] for x in train_losses if x["step"] > 0]
            vals_t = [x["loss"] for x in train_losses if x["step"] > 0]
            if args.x_axis == "step_x_lr":
                fn = cum_lr_fns[lr]
                steps_t = [fn(s) for s in steps_t]
            ax_loss.loglog(steps_t, vals_t, linestyle='--', linewidth=1, color=colors[i],
                           alpha=0.6)
        eval_losses = loss_histories[lr].get("eval_losses", [])
        if eval_losses:
            steps_l = [x["step"] for x in eval_losses if x["step"] > 0]
            vals_l = [x["loss"] for x in eval_losses if x["step"] > 0]
            if args.x_axis == "step_x_lr":
                fn = cum_lr_fns[lr]
                steps_l = [fn(s) for s in steps_l]
            ax_loss.loglog(steps_l, vals_l, marker='o', markersize=4, color=colors[i],
                           label=f'lr={lr}')

    # Fit per-LR power law to loss (cumulative LR >= 2 or step >= 20)
    for i, lr in enumerate(sorted_lrs):
        if lr not in loss_histories:
            continue
        eval_losses = loss_histories[lr].get("eval_losses", [])
        if args.x_axis == "step_x_lr":
            fn = cum_lr_fns[lr]
            fit_pairs = [(fn(x["step"]), x["loss"]) for x in eval_losses
                         if fn(x["step"]) >= 2.0 and not np.isnan(x["loss"]) and x["loss"] > 0]
        else:
            fit_pairs = [(x["step"], x["loss"]) for x in eval_losses
                         if x["step"] >= 20 and not np.isnan(x["loss"]) and x["loss"] > 0]
        if len(fit_pairs) >= 2:
            fx, fy = zip(*fit_pairs)
            coeffs = np.polyfit(np.log10(fx), np.log10(fy), 1)
            # Update legend label with slope
            for line in ax_loss.get_lines():
                if line.get_label() == f'lr={lr}':
                    line.set_label(f'lr={lr} (slope={coeffs[0]:.3f})')

    ax_loss.set_xlabel(x_label)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss Curves (solid + markers = eval, dashed = train)")
    ax_loss.legend(fontsize=7, loc="best")
    ax_loss.grid(True, alpha=0.3)

    # Cross-section panel: product vs LR at fixed step*lr values (log-log interpolated)
    if args.cross_section is not None:
        ax_cs = axes[3]
        log_cs = np.log10(args.cross_section)
        cs_norm = (log_cs - log_cs.min()) / (log_cs.max() - log_cs.min() + 1e-10)
        cs_colors = plt.cm.viridis(cs_norm)
        for ci, target_slr in enumerate(args.cross_section):
            cs_lrs = []
            cs_vals = []
            for lr in sorted_lrs:
                data_by_step = runs[lr]
                steps = sorted(data_by_step.keys())
                fn = cum_lr_fns[lr]
                slr_vals = [fn(s) for s in steps]
                products = [get_product(data_by_step[s], rank_lo, rank_hi, "pos") for s in steps]
                valid = [(x, y) for x, y in zip(slr_vals, products) if not np.isnan(y) and y > 0 and x > 0]
                if len(valid) < 2:
                    continue
                xs, ys = zip(*valid)
                if target_slr < min(xs) or target_slr > max(xs):
                    continue
                log_interp = np.interp(np.log(target_slr), np.log(xs), np.log(ys))
                cs_lrs.append(lr)
                cs_vals.append(np.exp(log_interp))
            if cs_lrs:
                ax_cs.loglog(cs_lrs, cs_vals, marker='o', markersize=8, color=cs_colors[ci],
                             linewidth=2, zorder=5, label=rf'$\sum\eta$={target_slr}')
                for lr, val in zip(cs_lrs, cs_vals):
                    ax_cs.annotate(f'{lr}', (lr, val), fontsize=6, ha='left',
                                   xytext=(5, 3), textcoords='offset points')

        # Individual eigenvalues (dotted) at the highest cross-section value
        target_slr_top = max(args.cross_section)
        rank_colors_pos = plt.cm.copper(np.linspace(0.15, 0.85, rank_hi - rank_lo + 1))
        for ri, rank in enumerate(range(rank_lo, rank_hi + 1)):
            r_lrs = []
            r_vals = []
            for lr in sorted_lrs:
                data_by_step = runs[lr]
                steps = sorted(data_by_step.keys())
                fn = cum_lr_fns[lr]
                slr_vals = [fn(s) for s in steps]
                evs = [get_pos_eigenvalue(data_by_step[s], rank) for s in steps]
                valid = [(x, y) for x, y in zip(slr_vals, evs) if not np.isnan(y) and y > 0 and x > 0]
                if len(valid) < 2:
                    continue
                xs, ys = zip(*valid)
                if target_slr_top < min(xs) or target_slr_top > max(xs):
                    continue
                r_lrs.append(lr)
                r_vals.append(np.exp(np.interp(np.log(target_slr_top), np.log(xs), np.log(ys))))
            if r_lrs:
                ax_cs.loglog(r_lrs, r_vals, marker='x', markersize=5, linestyle=':',
                             color=rank_colors_pos[ri], linewidth=1.2, alpha=0.8,
                             label=rf'rank {rank} @ $\sum\eta$={target_slr_top}')

        # Reference line y = 2 / (x / (1 - beta)) for adam (beta = momentum, HF default 0.9);
        # falls back to y = 2/x otherwise.
        if args.optimizer.lower() in ("adam", "adamw"):
            beta = 0.9  # HF Trainer adam_beta1 default; not overridden in run_pipeline.py
            ref_label = rf'y = 2/(x/(1-{beta}))'
        else:
            beta = 0.0
            ref_label = 'y = 2/x'
        ref_x = np.logspace(np.log10(min(sorted_lrs) * 0.5), np.log10(max(sorted_lrs) * 2), 50)
        ref_y = 2.0 * (1.0 - beta) / ref_x
        ax_cs.loglog(ref_x, ref_y, 'k--', linewidth=1, alpha=0.5, label=ref_label)

        ax_cs.set_xlabel("Learning Rate")
        ax_cs.set_ylabel(f"{label} $\\lambda_i^+$")
        ax_cs.set_title(f"Positive Geo. Mean vs LR (cross-sections)")
        ax_cs.legend(fontsize=8)
        ax_cs.grid(True, which="both", alpha=0.3)

        # Negative cross-section panel
        ax_cs_neg = axes[4]
        for ci, target_slr in enumerate(args.cross_section):
            cs_lrs = []
            cs_vals = []
            for lr in sorted_lrs:
                data_by_step = runs[lr]
                steps = sorted(data_by_step.keys())
                fn = cum_lr_fns[lr]
                slr_vals = [fn(s) for s in steps]
                products = [get_product(data_by_step[s], rank_lo, rank_hi, "neg") for s in steps]
                valid = [(x, y) for x, y in zip(slr_vals, products) if not np.isnan(y) and y > 0 and x > 0]
                if len(valid) < 2:
                    continue
                xs, ys = zip(*valid)
                if target_slr < min(xs) or target_slr > max(xs):
                    continue
                log_interp = np.interp(np.log(target_slr), np.log(xs), np.log(ys))
                cs_lrs.append(lr)
                cs_vals.append(np.exp(log_interp))
            if cs_lrs:
                ax_cs_neg.loglog(cs_lrs, cs_vals, marker='s', markersize=8, color=cs_colors[ci],
                                 linewidth=2, zorder=5, label=rf'$\sum\eta$={target_slr}')
                for lr, val in zip(cs_lrs, cs_vals):
                    ax_cs_neg.annotate(f'{lr}', (lr, val), fontsize=6, ha='left',
                                       xytext=(5, 3), textcoords='offset points')

        # Individual |negative eigenvalues| (dotted) at the highest cross-section value
        for ri, rank in enumerate(range(rank_lo, rank_hi + 1)):
            r_lrs = []
            r_vals = []
            for lr in sorted_lrs:
                data_by_step = runs[lr]
                steps = sorted(data_by_step.keys())
                fn = cum_lr_fns[lr]
                slr_vals = [fn(s) for s in steps]
                evs = [get_neg_eigenvalue(data_by_step[s], rank) for s in steps]
                valid = [(x, y) for x, y in zip(slr_vals, evs) if not np.isnan(y) and y > 0 and x > 0]
                if len(valid) < 2:
                    continue
                xs, ys = zip(*valid)
                if target_slr_top < min(xs) or target_slr_top > max(xs):
                    continue
                r_lrs.append(lr)
                r_vals.append(np.exp(np.interp(np.log(target_slr_top), np.log(xs), np.log(ys))))
            if r_lrs:
                ax_cs_neg.loglog(r_lrs, r_vals, marker='x', markersize=5, linestyle=':',
                                 color=rank_colors_pos[ri], linewidth=1.2, alpha=0.8,
                                 label=rf'rank {rank} @ $\sum\eta$={target_slr_top}')

        # Same reference line as positive panel
        ax_cs_neg.loglog(ref_x, ref_y, 'k--', linewidth=1, alpha=0.5, label=ref_label)
        ax_cs_neg.set_xlabel("Learning Rate")
        ax_cs_neg.set_ylabel(f"{label} $|\\lambda_i^-|$")
        ax_cs_neg.set_title(f"Negative Geo. Mean vs LR (cross-sections)")
        ax_cs_neg.legend(fontsize=8)
        ax_cs_neg.grid(True, which="both", alpha=0.3)

        # Generalization-gap cross-section panel: eval_loss - train_loss vs LR
        # HF Trainer's train_loss[S] is the mean over steps [S-W+1, S] (logging window).
        # Shift each train_loss timestamp to the log-step midpoint sqrt((S-W+1)*S) so
        # the value better represents the instantaneous loss at the window's effective
        # center (rather than at the right edge).
        W_LOG = 50  # logging_steps in run_pipeline.py
        def train_step_shift(s):
            lo = max(1, s - W_LOG + 1)
            return max(1, int(round(np.sqrt(lo * s))))

        # Train-loss cross-section panel
        ax_cs_train = axes[5]
        for ci, target_slr in enumerate(args.cross_section):
            tl_lrs = []
            tl_vals = []
            for lr in sorted_lrs:
                if lr not in loss_histories:
                    continue
                fn = cum_lr_fns[lr]
                train_pairs = [(fn(train_step_shift(x["step"])), x["loss"])
                               for x in loss_histories[lr].get("train_losses", [])
                               if fn(train_step_shift(x["step"])) > 0
                               and not np.isnan(x["loss"]) and x["loss"] > 0]
                if len(train_pairs) < 2:
                    continue
                tx, ty = zip(*sorted(train_pairs))
                if target_slr < min(tx) or target_slr > max(tx):
                    continue
                tl_lrs.append(lr)
                tl_vals.append(np.exp(np.interp(np.log(target_slr), np.log(tx), np.log(ty))))
            if tl_lrs:
                ax_cs_train.loglog(tl_lrs, tl_vals, marker='o', markersize=8, color=cs_colors[ci],
                                   linewidth=2, zorder=5, label=rf'$\sum\eta$={target_slr}')
                for lr, val in zip(tl_lrs, tl_vals):
                    ax_cs_train.annotate(f'{lr}', (lr, val), fontsize=6, ha='left',
                                         xytext=(5, 3), textcoords='offset points')
        ax_cs_train.set_xlabel("Learning Rate")
        ax_cs_train.set_ylabel("Train Loss")
        ax_cs_train.set_title("Train Loss vs LR (cross-sections)")
        ax_cs_train.legend(fontsize=8)
        ax_cs_train.grid(True, which="both", alpha=0.3)

        # Eval-loss cross-section panel
        ax_cs_eval = axes[6]
        for ci, target_slr in enumerate(args.cross_section):
            el_lrs = []
            el_vals = []
            for lr in sorted_lrs:
                if lr not in loss_histories:
                    continue
                fn = cum_lr_fns[lr]
                eval_pairs = [(fn(x["step"]), x["loss"])
                              for x in loss_histories[lr].get("eval_losses", [])
                              if fn(x["step"]) > 0
                              and not np.isnan(x["loss"]) and x["loss"] > 0]
                if len(eval_pairs) < 2:
                    continue
                ex, ey = zip(*sorted(eval_pairs))
                if target_slr < min(ex) or target_slr > max(ex):
                    continue
                el_lrs.append(lr)
                el_vals.append(np.exp(np.interp(np.log(target_slr), np.log(ex), np.log(ey))))
            if el_lrs:
                ax_cs_eval.loglog(el_lrs, el_vals, marker='o', markersize=8, color=cs_colors[ci],
                                  linewidth=2, zorder=5, label=rf'$\sum\eta$={target_slr}')
                for lr, val in zip(el_lrs, el_vals):
                    ax_cs_eval.annotate(f'{lr}', (lr, val), fontsize=6, ha='left',
                                        xytext=(5, 3), textcoords='offset points')
        ax_cs_eval.set_xlabel("Learning Rate")
        ax_cs_eval.set_ylabel("Eval Loss")
        ax_cs_eval.set_title("Eval Loss vs LR (cross-sections)")
        ax_cs_eval.legend(fontsize=8)
        ax_cs_eval.grid(True, which="both", alpha=0.3)

        ax_cs_gap = axes[7]
        for ci, target_slr in enumerate(args.cross_section):
            gap_lrs = []
            gap_vals = []
            for lr in sorted_lrs:
                if lr not in loss_histories:
                    continue
                fn = cum_lr_fns[lr]
                train_pairs = [(fn(train_step_shift(x["step"])), x["loss"])
                               for x in loss_histories[lr].get("train_losses", [])
                               if fn(train_step_shift(x["step"])) > 0
                               and not np.isnan(x["loss"]) and x["loss"] > 0]
                eval_pairs = [(fn(x["step"]), x["loss"]) for x in loss_histories[lr].get("eval_losses", [])
                              if fn(x["step"]) > 0 and not np.isnan(x["loss"]) and x["loss"] > 0]
                if len(train_pairs) < 2 or len(eval_pairs) < 2:
                    continue
                tx, ty = zip(*sorted(train_pairs))
                ex, ey = zip(*sorted(eval_pairs))
                if target_slr < max(min(tx), min(ex)) or target_slr > min(max(tx), max(ex)):
                    continue
                train_at = np.exp(np.interp(np.log(target_slr), np.log(tx), np.log(ty)))
                eval_at = np.exp(np.interp(np.log(target_slr), np.log(ex), np.log(ey)))
                gap_lrs.append(lr)
                gap_vals.append(eval_at - train_at)
            if gap_lrs:
                ax_cs_gap.plot(gap_lrs, gap_vals, marker='o', markersize=8, color=cs_colors[ci],
                               linewidth=2, zorder=5, label=rf'$\sum\eta$={target_slr}')
                for lr, val in zip(gap_lrs, gap_vals):
                    ax_cs_gap.annotate(f'{lr}', (lr, val), fontsize=6, ha='left',
                                       xytext=(5, 3), textcoords='offset points')
        ax_cs_gap.set_xscale("log")
        ax_cs_gap.set_yscale("symlog", linthresh=0.01)
        ax_cs_gap.axhline(0, color='k', linewidth=0.5, alpha=0.4)
        ax_cs_gap.set_xlabel("Learning Rate")
        ax_cs_gap.set_ylabel("eval_loss $-$ train_loss")
        ax_cs_gap.set_title("Generalization Gap vs LR (cross-sections)")
        ax_cs_gap.legend(fontsize=8)
        ax_cs_gap.grid(True, which="both", alpha=0.3)

    suffix = f"_{args.x_axis}" if args.x_axis != "step" else ""
    fig.suptitle(f"{args.optimizer.upper()} — Eigenvalue Geometric Mean (rank {rank_lo}-{rank_hi})", fontsize=13)
    plt.tight_layout()

    sched_suffix = f"_{args.lr_scheduler_type}" if args.lr_scheduler_type != "constant" else ""
    save_path = args.output or f"outputs/eigen_geomean_{args.model_id}_{args.optimizer}{sched_suffix}{suffix}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
