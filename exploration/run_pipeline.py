"""
Hessian Training and Analysis Pipeline

Usage:
    # Single run
    python run_pipeline.py batch_size=256 learning_rate=1e-4

    # Sweep over hyperparameters
    python run_pipeline.py --multirun batch_size=128,256 learning_rate=1e-4,5e-4

Results are saved to: outputs/run_pipeline/bs{batch_size}_lr{learning_rate}/
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import gc
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset
from pyhessian import hessian


# ============== HELPER FUNCTIONS ==============

class ModelWrapper(torch.nn.Module):
    """Wrapper for PyHessian compatibility."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        return outputs.logits, outputs.loss


def criterion(outputs, labels):
    """Loss function for PyHessian."""
    logits, loss = outputs
    return loss


def flatten(lst):
    """Recursively flatten nested lists, extracting only scalar values."""
    result = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        elif hasattr(item, 'numel'):
            if item.numel() == 1:
                result.append(item.item())
        elif hasattr(item, 'item'):
            result.append(item.item())
        elif isinstance(item, (int, float)):
            result.append(item)
    return result


def get_density(ev, weights, bins=100):
    """Smooth density estimation from eigenvalues and weights."""
    sigma = 0.1 * (max(ev) - min(ev))
    grid = np.linspace(min(ev), max(ev), bins)
    density = np.zeros_like(grid)
    for v, w in zip(ev, weights):
        density += w * np.exp(-(grid - v)**2 / (2 * sigma**2))
    return grid, density


def compute_and_save_hessian(model, step, hessian_batch, cfg, output_dir, device):
    """Compute Hessian spectrum and save results."""
    print(f"\n{'='*50}")
    print(f"Computing Hessian analysis at step {step}")
    print(f"{'='*50}")

    model.eval()
    model.gradient_checkpointing_enable()
    wrapped_model = ModelWrapper(model)

    # Ensure hessian_batch is on the correct device
    model_device = next(model.parameters()).device
    hessian_batch_device = hessian_batch.to(model_device)

    hessian_comp = hessian(
        wrapped_model, criterion,
        data=(hessian_batch_device, hessian_batch_device),
        cuda=(str(model_device).startswith("cuda"))
    )

    # SLQ density estimation
    print(f"Computing SLQ density (iter={cfg.slq_iter})...")
    density_eigenvalues, density_weights = hessian_comp.density(iter=cfg.slq_iter)

    # Exact Lanczos for top-k
    max_iter = cfg.top_k * cfg.max_iter_multiplier
    print(f"Computing top {cfg.top_k} eigenvalues (maxIter={max_iter})...")
    top_eigenvalues_raw = hessian_comp.eigenvalues(top_n=cfg.top_k, maxIter=max_iter)
    top_eigenvalues = flatten(top_eigenvalues_raw)
    top_eigenvalues = [ev for ev in top_eigenvalues if ev > 0]
    print(f"Top {len(top_eigenvalues)} positive eigenvalues: {top_eigenvalues}")

    # Process SLQ results
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    evs = np.array(density_eigenvalues[0])
    wts = np.array(density_weights[0])
    idx = np.argsort(evs)[::-1]
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]
    mask = evs_sorted > 1e-6
    evs_final = evs_sorted[mask]
    ranks = np.cumsum(wts_sorted[mask])
    indices = ranks * total_params

    # Fit power law
    fit_mask = indices >= cfg.min_fit_index
    c_fit, p_fit = None, None
    if np.sum(fit_mask) > 2:
        log_indices = np.log(indices[fit_mask])
        log_evs = np.log(evs_final[fit_mask])
        coeffs = np.polyfit(log_indices, log_evs, 1)
        p_fit = -coeffs[0]
        c_fit = np.exp(coeffs[1])
        print(f"Power law fit: lambda = {c_fit:.4f} * i^(-{p_fit:.4f})")

    # Save data
    save_data = {
        "step": step,
        "slq": {
            "eigenvalues": evs_final.tolist(),
            "indices": indices.tolist(),
            "raw_eigenvalues": [float(x) for x in density_eigenvalues[0]],
            "raw_weights": [float(x) for x in density_weights[0]],
        },
        "lanczos": {
            "eigenvalues": [float(x) for x in top_eigenvalues] if top_eigenvalues else [],
        },
        "power_law_fit": {
            "c": float(c_fit) if c_fit is not None else None,
            "p": float(p_fit) if p_fit is not None else None,
        },
        "config": {
            "top_k": cfg.top_k,
            "slq_iter": cfg.slq_iter,
            "batch_size": cfg.hessian_batch_size,
            "total_params": total_params,
        }
    }

    data_path = os.path.join(output_dir, f"hessian_step_{step}.json")
    with open(data_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved data to {data_path}")

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Density plot
    grid, density = get_density(density_eigenvalues[0], density_weights[0])
    ax1.plot(grid, density, color='blue', lw=2)
    ax1.set_title(f"Hessian Eigenvalue Density (Step {step})")
    ax1.set_xlabel(r"Eigenvalue $\lambda$")
    ax1.set_ylabel(r"Density $\rho(\lambda)$")
    ax1.grid(True, alpha=0.3)

    # Log-log plot
    ax2.loglog(indices, evs_final, marker='o', linestyle='None', alpha=0.6,
               markersize=4, label='SLQ Density')

    if top_eigenvalues:
        top_evs = np.array(sorted(top_eigenvalues, reverse=True))
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
    ax2.set_xlim(cfg.plot_xlim[0], cfg.plot_xlim[1])
    ax2.set_ylim(cfg.plot_ylim[0], cfg.plot_ylim[1])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"hessian_step_{step}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Cleanup
    model.gradient_checkpointing_disable()
    model.train()
    del hessian_comp
    del wrapped_model
    gc.collect()
    torch.cuda.empty_cache()

    return save_data


class LossRecorderCallback(TrainerCallback):
    """Callback to record training and evaluation losses."""
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append({"step": state.global_step, "loss": logs["loss"]})
            if "eval_loss" in logs:
                self.eval_losses.append({"step": state.global_step, "loss": logs["eval_loss"]})


class HessianTrainer(Trainer):
    """Custom Trainer that computes Hessian at specified steps."""
    def __init__(self, *args, hessian_batch=None, hessian_steps=None,
                 hessian_cfg=None, hessian_output_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hessian_batch = hessian_batch
        self.hessian_steps = hessian_steps or []
        self.hessian_cfg = hessian_cfg
        self.hessian_output_dir = hessian_output_dir
        self.hessian_results = {}

    def training_step(self, model, inputs, num_items_in_batch=None):
        current_step = self.state.global_step
        if current_step in self.hessian_steps and current_step not in self.hessian_results:
            device = next(model.parameters()).device
            result = compute_and_save_hessian(
                model, current_step, self.hessian_batch,
                self.hessian_cfg, self.hessian_output_dir, str(device)
            )
            self.hessian_results[current_step] = result
        return super().training_step(model, inputs, num_items_in_batch)


def slq_interp_eigenvalue(evs, wts, total_params, target_index):
    """Interpolate eigenvalue at target_index using log-log interpolation."""
    if len(evs) == 0 or len(wts) == 0:
        return np.nan
    evs = np.array(evs)
    wts = np.array(wts)
    idx = np.argsort(evs)[::-1]
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]
    mask = evs_sorted > 1e-10
    evs_pos = evs_sorted[mask]
    wts_pos = wts_sorted[mask]
    if len(evs_pos) < 2:
        return np.nan
    indices = np.cumsum(wts_pos) * total_params
    if target_index < indices[0] or target_index > indices[-1]:
        return np.nan
    log_indices = np.log(indices)
    log_evs = np.log(evs_pos)
    log_target = np.log(target_index)
    log_result = np.interp(log_target, log_indices, log_evs)
    return np.exp(log_result)


def run_analysis(cfg, output_dir):
    """Run eigenvalue trajectory analysis."""
    print("\n" + "="*60)
    print("RUNNING EIGENVALUE TRAJECTORY ANALYSIS")
    print("="*60)

    # Load all hessian data
    import glob
    pattern = os.path.join(output_dir, "hessian_step_*.json")
    files = glob.glob(pattern)

    if not files:
        print(f"No hessian files found in {output_dir}")
        return

    data_by_step = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        step = int(filename.replace("hessian_step_", "").replace(".json", ""))
        with open(filepath, "r") as f:
            data_by_step[step] = json.load(f)

    steps = sorted(data_by_step.keys())
    print(f"Found data for steps: {steps}")

    # Determine indices to track
    indices_to_track = [i - 1 for i in cfg.eigenvalue_indices]
    lanczos_indices_0 = set(i - 1 for i in cfg.lanczos_indices)

    eigenvalue_trajectories = {i: [] for i in indices_to_track}
    eigenvalue_sources = {i: "lanczos" if i in lanczos_indices_0 else cfg.eigenvalue_source
                          for i in indices_to_track}
    valid_steps = []

    for step in steps:
        data = data_by_step[step]
        lanczos_evs = data.get("lanczos", {}).get("eigenvalues", [])
        slq_evs = data.get("slq", {}).get("raw_eigenvalues", [])
        slq_wts = data.get("slq", {}).get("raw_weights", [])
        total_params = data.get("config", {}).get("total_params", None)

        sorted_lanczos = sorted(lanczos_evs, reverse=True) if lanczos_evs else []
        sorted_slq = sorted(slq_evs, reverse=True) if slq_evs else []

        if sorted_lanczos or sorted_slq:
            valid_steps.append(step)
            for i in indices_to_track:
                target_index = i + 1
                if i in lanczos_indices_0:
                    if i < len(sorted_lanczos):
                        eigenvalue_trajectories[i].append(sorted_lanczos[i])
                    else:
                        eigenvalue_trajectories[i].append(np.nan)
                elif cfg.eigenvalue_source == "slq_interp" and total_params:
                    ev = slq_interp_eigenvalue(slq_evs, slq_wts, total_params, target_index)
                    eigenvalue_trajectories[i].append(ev)
                else:
                    if i < len(sorted_slq):
                        eigenvalue_trajectories[i].append(sorted_slq[i])
                    else:
                        eigenvalue_trajectories[i].append(np.nan)

    # Plot trajectories
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices_to_track)))

    for idx, i in enumerate(indices_to_track):
        evs = eigenvalue_trajectories[i]
        source = eigenvalue_sources[i]
        marker = 'o' if source == "lanczos" else 's'
        ax.semilogy(valid_steps, evs, marker=marker, color=colors[idx],
                    label=f'$\\lambda_{{{i+1}}}$ ({source})', linewidth=2, markersize=6)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Eigenvalue (log scale)", fontsize=12)
    ax.set_title(f"Eigenvalue Trajectories (bs={cfg.batch_size}, lr={cfg.learning_rate})")
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(valid_steps)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "eigenvalue_trajectories.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot to {plot_path}")

    # Save trajectory data (convert OmegaConf types to Python types for JSON)
    trajectory_data = {
        "steps": valid_steps,
        "eigenvalue_indices": list(cfg.eigenvalue_indices),
        "trajectories": {str(i+1): eigenvalue_trajectories[i] for i in indices_to_track},
        "sources": {str(i+1): eigenvalue_sources[i] for i in indices_to_track},
    }
    traj_path = os.path.join(output_dir, "eigenvalue_trajectories.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory_data, f, indent=2)
    print(f"Saved trajectory data to {traj_path}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main pipeline: train with Hessian analysis, then run trajectory analysis."""

    print("="*60)
    print("HESSIAN TRAINING PIPELINE")
    print("="*60)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("No GPU available")

    output_dir = os.getcwd()  # Hydra sets this to the run directory
    original_dir = hydra.utils.get_original_cwd()  # Original directory for caching
    cache_dir = os.path.join(original_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")

    # Disable efficient attention for Hessian computation
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Load tokenizer
    print(f"\nLoading model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if cfg.train_from_scratch:
        print("Initializing model from scratch (random weights)")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(cfg.model_name)
        model = AutoModelForCausalLM.from_config(config).to(device)
    else:
        print("Loading pretrained model weights")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load datasets with fixed cache directory
    print("Loading datasets...")
    train_dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=cache_dir)
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation", cache_dir=cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True,
                        max_length=cfg.max_length, padding="max_length")

    # Use fixed cache file names for tokenized datasets
    train_cache_file = os.path.join(cache_dir, f"tokenized_train_maxlen{cfg.max_length}.arrow")
    val_cache_file = os.path.join(cache_dir, f"tokenized_val_maxlen{cfg.max_length}.arrow")

    print("Tokenizing datasets (using cache if available)...")
    tokenized_train = train_dataset.map(
        tokenize_function, batched=True,
        remove_columns=train_dataset.column_names, num_proc=12,
        cache_file_name=train_cache_file,
        load_from_cache_file=True,
    )
    tokenized_val = val_dataset.map(
        tokenize_function, batched=True,
        remove_columns=val_dataset.column_names, num_proc=12,
        cache_file_name=val_cache_file,
        load_from_cache_file=True,
    )
    tokenized_val = tokenized_val.select(range(min(cfg.eval_samples, len(tokenized_val))))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Prepare Hessian batch
    print("Preparing Hessian batch...")
    hessian_samples = tokenized_val.select(range(cfg.hessian_batch_size))
    hessian_batch = torch.tensor([s["input_ids"] for s in hessian_samples]).to(device)

    # Hessian evaluation steps
    hessian_steps = [0] + list(range(cfg.hessian_eval_interval,
                                      cfg.total_steps + 1,
                                      cfg.hessian_eval_interval))

    # Compute Hessian at step 0
    print("\n" + "="*60)
    print("COMPUTING HESSIAN AT STEP 0")
    print("="*60)
    compute_and_save_hessian(model, 0, hessian_batch, cfg, output_dir, device)

    # Calculate gradient accumulation for large batch sizes
    import math
    MAX_PHYSICAL_BATCH_SIZE = 256

    # Find minimum accumulation steps needed to keep physical batch size <= 256
    gradient_accumulation_steps = math.ceil(cfg.batch_size / MAX_PHYSICAL_BATCH_SIZE)
    physical_batch_size = cfg.batch_size // gradient_accumulation_steps
    effective_batch_size = physical_batch_size * gradient_accumulation_steps

    if effective_batch_size != cfg.batch_size:
        print(f"Warning: batch_size {cfg.batch_size} is not divisible by {gradient_accumulation_steps}")
        print(f"Effective batch size will be {effective_batch_size} (instead of {cfg.batch_size})")

    print(f"Batch size config: physical={physical_batch_size}, "
          f"accumulation_steps={gradient_accumulation_steps}, "
          f"effective={effective_batch_size}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        max_steps=cfg.total_steps,
        per_device_train_batch_size=physical_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=50,
        save_steps=cfg.hessian_eval_interval,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Setup callbacks and trainer
    loss_recorder = LossRecorderCallback()

    # Evaluate at step 0
    trainer = HessianTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        hessian_batch=hessian_batch,
        hessian_steps=[s for s in hessian_steps if s > 0],
        hessian_cfg=cfg,
        hessian_output_dir=output_dir,
        callbacks=[loss_recorder],
    )

    # Evaluate at step 0
    print("\nEvaluating at step 0...")
    eval_results = trainer.evaluate()
    loss_recorder.eval_losses.append({"step": 0, "loss": eval_results["eval_loss"]})
    print(f"Step 0 eval loss: {eval_results['eval_loss']:.4f}")

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    trainer.train()

    # Final Hessian if needed
    if cfg.total_steps in hessian_steps and cfg.total_steps not in trainer.hessian_results:
        compute_and_save_hessian(model, cfg.total_steps, hessian_batch, cfg, output_dir, device)

    # Save loss data
    loss_data = {
        "train_losses": loss_recorder.train_losses,
        "eval_losses": loss_recorder.eval_losses,
    }
    loss_path = os.path.join(output_dir, "loss_history.json")
    with open(loss_path, "w") as f:
        json.dump(loss_data, f, indent=2)

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    if loss_recorder.train_losses:
        train_steps = [x["step"] for x in loss_recorder.train_losses]
        train_vals = [x["loss"] for x in loss_recorder.train_losses]
        ax.plot(train_steps, train_vals, 'b-', alpha=0.7, label='Train Loss')
    if loss_recorder.eval_losses:
        eval_steps = [x["step"] for x in loss_recorder.eval_losses]
        eval_vals = [x["loss"] for x in loss_recorder.eval_losses]
        ax.plot(eval_steps, eval_vals, 'r-', marker='o', markersize=4, label='Eval Loss')
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves (bs={cfg.batch_size}, lr={cfg.learning_rate})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close()

    # Run trajectory analysis
    run_analysis(cfg, output_dir)

    # Save config
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
