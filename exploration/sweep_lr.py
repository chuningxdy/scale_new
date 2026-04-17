"""
Automated learning rate sweep. Runs increasing LRs until loss diverges.

Usage:
    python sweep_lr.py --optimizer sgd --lrs 0.001 0.003 0.01 0.03 0.1 0.3 1.0 3.0
    python sweep_lr.py --optimizer adam --lrs 0.001 0.003 0.01 0.03
    python sweep_lr.py --optimizer sgd  # uses default LR list
"""

import subprocess
import json
import os
import argparse
import numpy as np


DEFAULT_LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# Divergence detection: if eval loss at step 200 is worse than init, or final > init, it diverged
DIVERGENCE_THRESHOLD = 1.05  # final_loss > init_loss * threshold => diverged


def get_output_dir(model_id, dataset_name, batch_size, optimizer, lr, lr_scheduler_type="constant"):
    return f"outputs/run_pipeline/{model_id}_{dataset_name}_bs{batch_size}_lr{lr}_opt{optimizer}_{lr_scheduler_type}"


def check_already_done(model_id, dataset_name, batch_size, optimizer, lr, total_steps, lr_scheduler_type="constant"):
    """Check if a run already completed by looking for the final hessian file."""
    output_dir = get_output_dir(model_id, dataset_name, batch_size, optimizer, lr, lr_scheduler_type)
    loss_file = os.path.join(output_dir, "loss_history.json")
    hessian_file = os.path.join(output_dir, f"hessian_step_{total_steps}.json")
    return os.path.exists(loss_file) and os.path.exists(hessian_file)


def get_results(model_id, dataset_name, batch_size, optimizer, lr, total_steps, lr_scheduler_type="constant"):
    """Load results from a completed run."""
    output_dir = get_output_dir(model_id, dataset_name, batch_size, optimizer, lr, lr_scheduler_type)
    loss_file = os.path.join(output_dir, "loss_history.json")
    hessian_file = os.path.join(output_dir, f"hessian_step_{total_steps}.json")

    if not os.path.exists(loss_file):
        return None

    with open(loss_file) as f:
        loss_data = json.load(f)

    eval_losses = loss_data.get("eval_losses", [])
    if not eval_losses:
        return None

    init_loss = eval_losses[0]["loss"]
    final_loss = eval_losses[-1]["loss"]

    hessian_data = None
    if os.path.exists(hessian_file):
        with open(hessian_file) as f:
            hessian_data = json.load(f)

    top_pos = []
    top_neg = []
    if hessian_data:
        top_pos = sorted(hessian_data.get("lanczos", {}).get("eigenvalues", []), reverse=True)[:5]
        top_neg = sorted(hessian_data.get("lanczos_bottom", {}).get("eigenvalues", []))[:5]

    return {
        "init_loss": init_loss,
        "final_loss": final_loss,
        "eval_losses": eval_losses,
        "top_pos": top_pos,
        "top_neg": top_neg,
    }


def is_diverged(results):
    """Check if the run diverged."""
    if results is None:
        return True
    # Check if loss is NaN
    if np.isnan(results["final_loss"]):
        return True
    # Check if final loss is worse than init
    if results["final_loss"] > results["init_loss"] * DIVERGENCE_THRESHOLD:
        return True
    return False


def run_experiment(optimizer, lr, extra_args):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running {optimizer.upper()} lr={lr}")
    print(f"{'='*60}")

    cmd = [
        "python", "run_pipeline.py",
        f"learning_rate={lr}",
        f"optimizer={optimizer}",
    ] + extra_args

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def print_summary(optimizer, results_by_lr):
    """Print summary table."""
    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY: {optimizer.upper()}")
    print(f"{'='*60}")
    print(f"{'LR':>10} {'Init Loss':>10} {'Final Loss':>11} {'Top Pos':>10} {'Top Neg':>10} {'Status':>10}")
    print("-" * 65)
    for lr, res in results_by_lr.items():
        if res is None:
            print(f"{lr:>10} {'—':>10} {'—':>11} {'—':>10} {'—':>10} {'FAILED':>10}")
        else:
            top_p = f"{res['top_pos'][0]:.1f}" if res['top_pos'] else "—"
            top_n = f"{res['top_neg'][0]:.1f}" if res['top_neg'] else "—"
            status = "DIVERGED" if is_diverged(res) else "OK"
            print(f"{lr:>10} {res['init_loss']:>10.4f} {res['final_loss']:>11.4f} {top_p:>10} {top_n:>10} {status:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--model-id", type=str, default="pythia70m")
    parser.add_argument("--dataset-name", type=str, default="tinystories")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--hessian-eval-interval", type=int, default=20)
    parser.add_argument("--hessian-log-base", type=int, default=2)
    parser.add_argument("--hessian-linear-interval", type=int, default=0)
    parser.add_argument("--hessian-explicit-steps", type=int, nargs="+", default=None)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--extra-args", type=str, nargs="*", default=[],
                        help="Additional hydra overrides passed to run_pipeline.py")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip runs that already have results")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--stop-on-diverge", action="store_true", default=True,
                        help="Stop sweep when loss diverges")
    parser.add_argument("--no-stop-on-diverge", action="store_false", dest="stop_on_diverge")
    args = parser.parse_args()

    common_args = [
        f"total_steps={args.total_steps}",
        f"hessian_eval_interval={args.hessian_eval_interval}",
        f"hessian_log_base={args.hessian_log_base}",
        f"hessian_linear_interval={args.hessian_linear_interval}",
        "hessian_spacing=log",
        "top_k=30",
        "bottom_k=30",
        "slq_iter=10",
        "hessian_batch_size=16",
        f"batch_size={args.batch_size}",
        f"eval_steps={args.eval_steps}",
        "eval_samples=100",
        "max_length=128",
        f"dataset_name={args.dataset_name}",
        f"model_name={args.model_name}",
        f"model_id={args.model_id}",
    ]
    if args.hessian_explicit_steps:
        steps_str = "[" + ",".join(str(s) for s in args.hessian_explicit_steps) + "]"
        common_args.append(f"hessian_explicit_steps={steps_str}")
    common_args += args.extra_args

    # Determine lr_scheduler_type from extra_args (for correct output dir lookup)
    lr_scheduler_type = "constant"
    for arg in args.extra_args:
        if arg.startswith("lr_scheduler_type="):
            lr_scheduler_type = arg.split("=", 1)[1]

    results_by_lr = {}

    for lr in sorted(args.lrs):
        # Check if already done
        if args.skip_existing and check_already_done(
                args.model_id, args.dataset_name, args.batch_size, args.optimizer, lr, args.total_steps,
                lr_scheduler_type=lr_scheduler_type):
            print(f"\nSkipping {args.optimizer} lr={lr} (already done)")
            results_by_lr[lr] = get_results(
                args.model_id, args.dataset_name, args.batch_size, args.optimizer, lr, args.total_steps,
                lr_scheduler_type=lr_scheduler_type)
            if args.stop_on_diverge and is_diverged(results_by_lr[lr]):
                print(f"  -> Previously diverged, stopping sweep.")
                break
            continue

        # Run experiment
        success = run_experiment(args.optimizer, lr, common_args)
        results_by_lr[lr] = get_results(
            args.model_id, args.dataset_name, args.batch_size, args.optimizer, lr, args.total_steps,
            lr_scheduler_type=lr_scheduler_type)

        if not success or is_diverged(results_by_lr[lr]):
            print(f"\n*** {args.optimizer.upper()} lr={lr} DIVERGED — stopping sweep ***")
            if args.stop_on_diverge:
                break

    print_summary(args.optimizer, results_by_lr)

    # Save summary
    summary_path = f"outputs/sweep_summary_{args.optimizer}_{lr_scheduler_type}.json"
    os.makedirs("outputs", exist_ok=True)
    summary = {
        "optimizer": args.optimizer,
        "results": {
            str(lr): {
                "final_loss": res["final_loss"] if res else None,
                "top_pos": res["top_pos"] if res else None,
                "top_neg": res["top_neg"] if res else None,
                "diverged": is_diverged(res),
            }
            for lr, res in results_by_lr.items()
        }
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
