"""
Hessian Training and Analysis Pipeline

Usage:
    # Single run
    python run_pipeline.py batch_size=256 learning_rate=1e-4

    # Sweep over hyperparameters
    python run_pipeline.py --multirun batch_size=128,256 learning_rate=1e-4,5e-4

Results are saved to: outputs/run_pipeline/bs{batch_size}_lr{learning_rate}/
"""

import os
# Set HuggingFace cache directories before importing HF libraries
# This is needed for lm1b dataset which is cached at a specific location
os.environ.setdefault("HF_HOME", "/mfs1/datasets/pile/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/mfs1/datasets/pile/huggingface/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/mfs1/datasets/pile/huggingface/datasets")

import torch
import matplotlib.pyplot as plt
import numpy as np
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


def neg_criterion(outputs, labels):
    """Negated loss function for computing bottom eigenvalues via -H."""
    logits, loss = outputs
    return -loss


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
    is_cuda = str(model_device).startswith("cuda")

    hessian_comp = hessian(
        wrapped_model, criterion,
        data=(hessian_batch_device, hessian_batch_device),
        cuda=is_cuda
    )

    # SLQ density estimation
    print(f"Computing SLQ density (iter={cfg.slq_iter})...")
    density_eigenvalues, density_weights = hessian_comp.density(iter=cfg.slq_iter)

    # Power iteration on H: finds eigenvalues by decreasing |magnitude|
    # Split by sign to get top positive and top negative
    bottom_k = cfg.get("bottom_k", 0)
    total_k = cfg.top_k + bottom_k
    max_iter = total_k * cfg.max_iter_multiplier
    print(f"Computing top {total_k} eigenvalues by magnitude (maxIter={max_iter})...")
    all_eigenvalues_raw = hessian_comp.eigenvalues(top_n=total_k, maxIter=max_iter)
    all_eigenvalues = flatten(all_eigenvalues_raw)
    top_eigenvalues = sorted([ev for ev in all_eigenvalues if ev > 0], reverse=True)
    bottom_eigenvalues = sorted([ev for ev in all_eigenvalues if ev < 0])
    print(f"Found {len(top_eigenvalues)} positive, {len(bottom_eigenvalues)} negative eigenvalues (of {len(all_eigenvalues)} total)")
    if top_eigenvalues:
        print(f"  Top positive: {top_eigenvalues[:5]}...")
    if bottom_eigenvalues:
        print(f"  Top negative: {bottom_eigenvalues[:5]}...")

    del hessian_comp
    gc.collect()
    torch.cuda.empty_cache()

    # SLQ density on -H for better negative eigenvalue resolution
    density_neg_eigenvalues = None
    density_neg_weights = None
    if bottom_k > 0:
        hessian_neg = hessian(
            wrapped_model, neg_criterion,
            data=(hessian_batch_device, hessian_batch_device),
            cuda=is_cuda
        )
        print(f"Computing SLQ density on -H (iter={cfg.slq_iter})...")
        density_neg_eigenvalues, density_neg_weights = hessian_neg.density(iter=cfg.slq_iter)
        del hessian_neg

    # Process SLQ results ----- 



    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    evs = np.array(density_eigenvalues[0], dtype=float)
    wts = np.array(density_weights[0], dtype=float)

    idx = np.argsort(evs)[::-1]
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]

    # normalize weights just in case
    wts_sorted = wts_sorted / wts_sorted.sum()

    # rank location = midpoint of each cumulative-weight bin
    cum = np.cumsum(wts_sorted)
    indices_all = (cum - 0.5 * wts_sorted) * total_params

    mask = evs_sorted > 1e-6
    evs_final = evs_sorted[mask]
    indices = indices_all[mask]

    if False:
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

    # Process negative eigenvalues from SLQ density
    evs_all = np.array(density_eigenvalues[0], dtype=float)
    wts_all = np.array(density_weights[0], dtype=float)
    idx_neg = np.argsort(evs_all)  # ascending: most negative first
    evs_neg_sorted = evs_all[idx_neg]
    wts_neg_sorted = wts_all[idx_neg]
    wts_neg_sorted = wts_neg_sorted / wts_neg_sorted.sum()

    # Bottom (negative) spectrum from SLQ
    mask_neg = evs_neg_sorted < -1e-6
    evs_neg_final = evs_neg_sorted[mask_neg]
    cum_neg = np.cumsum(wts_neg_sorted)
    indices_neg_all = (cum_neg - 0.5 * wts_neg_sorted) * total_params
    indices_neg = indices_neg_all[mask_neg]

    # Save data
    save_data = {
        "step": step,
        "slq": {
            "eigenvalues": evs_final.tolist(),
            "indices": indices.tolist(),
            "raw_eigenvalues": [float(x) for x in density_eigenvalues[0]],
            "raw_weights": [float(x) for x in density_weights[0]],
        },
        "slq_bottom": {
            "eigenvalues": evs_neg_final.tolist(),
            "indices": indices_neg.tolist(),
            # SLQ density computed on -H (negated back to H eigenvalues)
            "raw_eigenvalues": [-float(x) for x in density_neg_eigenvalues[0]] if density_neg_eigenvalues else [],
            "raw_weights": [float(x) for x in density_neg_weights[0]] if density_neg_weights else [],
        },
        "lanczos": {
            "eigenvalues": [float(x) for x in top_eigenvalues] if top_eigenvalues else [],
        },
        "lanczos_bottom": {
            "eigenvalues": [float(x) for x in bottom_eigenvalues] if bottom_eigenvalues else [],
        },
        "power_law_fit": {
            "c": float(c_fit) if c_fit is not None else None,
            "p": float(p_fit) if p_fit is not None else None,
        },
        "config": {
            "top_k": cfg.top_k,
            "bottom_k": bottom_k,
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
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax1, ax2, ax3 = axes

    # Density plot
    grid, density = get_density(density_eigenvalues[0], density_weights[0])
    ax1.plot(grid, density, color='blue', lw=2)
    ax1.set_title(f"Hessian Eigenvalue Density (Step {step})")
    ax1.set_xlabel(r"Eigenvalue $\lambda$")
    ax1.set_ylabel(r"Density $\rho(\lambda)$")
    ax1.grid(True, alpha=0.3)

    # Log-log plot: positive (top) spectrum
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

    ax2.set_title(f"Top Eigenvalue Spectrum (Step {step})")
    ax2.set_xlabel("Eigenvalue Index (from top)")
    ax2.set_ylabel(r"Eigenvalue $\lambda$")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xlim(cfg.plot_xlim[0], cfg.plot_xlim[1])
    ax2.set_ylim(cfg.plot_ylim[0], cfg.plot_ylim[1])

    # Log-log plot: negative (bottom) spectrum — plot |eigenvalue| vs index from bottom
    if len(evs_neg_final) > 0:
        ax3.loglog(indices_neg, np.abs(evs_neg_final), marker='o', linestyle='None',
                   alpha=0.6, markersize=4, color='purple', label='SLQ Density (neg)')

    if bottom_eigenvalues:
        bot_evs = np.array(sorted(bottom_eigenvalues))  # most negative first
        bot_indices = np.arange(1, len(bot_evs) + 1)
        ax3.loglog(bot_indices, np.abs(bot_evs), marker='+', linestyle='None',
                   color='orange', markersize=4, label='Bottom-k Lanczos', zorder=5, alpha=0.6)

    ax3.set_title(f"Bottom Eigenvalue Spectrum (Step {step})")
    ax3.set_xlabel("Eigenvalue Index (from bottom)")
    ax3.set_ylabel(r"$|\lambda|$ (negative eigenvalues)")
    ax3.legend()
    ax3.grid(True, which="both", alpha=0.3)
    ax3.set_xlim(cfg.plot_xlim[0], cfg.plot_xlim[1])
    ax3.set_ylim(cfg.plot_ylim[0], cfg.plot_ylim[1])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"hessian_step_{step}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Cleanup
    model.gradient_checkpointing_disable()
    model.train()
    del wrapped_model
    gc.collect()
    torch.cuda.empty_cache()

    return save_data


class SignSGD(torch.optim.Optimizer):
    """Sign descent optimizer: w = w - lr * sign(grad)."""
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])
                p.add_(p.grad.sign(), alpha=-group['lr'])
        return loss


class LossRecorderCallback(TrainerCallback):
    """Callback to record training and evaluation losses. Saves incrementally."""
    def __init__(self, output_dir=None):
        self.train_losses = []
        self.eval_losses = []
        self.output_dir = output_dir

    def _save(self):
        if self.output_dir:
            loss_data = {
                "train_losses": self.train_losses,
                "eval_losses": self.eval_losses,
            }
            loss_path = os.path.join(self.output_dir, "loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_data, f, indent=2)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append({"step": state.global_step, "loss": logs["loss"]})
            if "eval_loss" in logs:
                self.eval_losses.append({"step": state.global_step, "loss": logs["eval_loss"]})
                self._save()


class HessianTrainer(Trainer):
    """Custom Trainer that computes Hessian and eval at specified steps."""
    def __init__(self, *args, hessian_batch=None, hessian_steps=None,
                 hessian_cfg=None, hessian_output_dir=None,
                 loss_recorder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hessian_batch = hessian_batch
        self.hessian_steps = hessian_steps or []
        self.hessian_cfg = hessian_cfg
        self.hessian_output_dir = hessian_output_dir
        self.hessian_results = {}
        self.loss_recorder = loss_recorder
        self._eval_done_steps = set()

    def training_step(self, model, inputs, num_items_in_batch=None):
        current_step = self.state.global_step
        if current_step in self.hessian_steps and current_step not in self.hessian_results:
            # Run eval at hessian steps
            if current_step not in self._eval_done_steps:
                eval_results = self.evaluate()
                if self.loss_recorder:
                    self.loss_recorder.eval_losses.append(
                        {"step": current_step, "loss": eval_results["eval_loss"]})
                    self.loss_recorder._save()
                print(f"Step {current_step} eval loss: {eval_results['eval_loss']:.4f}")
                self._eval_done_steps.add(current_step)
            device = next(model.parameters()).device
            try:
                result = compute_and_save_hessian(
                    model, current_step, self.hessian_batch,
                    self.hessian_cfg, self.hessian_output_dir, str(device)
                )
                self.hessian_results[current_step] = result
            except RuntimeError as e:
                print(f"Warning: Hessian computation at step {current_step} failed: {e}")
                print("Skipping this step.")
                # Ensure model is in correct state for training
                model.gradient_checkpointing_disable()
                model.train()
                torch.cuda.empty_cache()
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
    original_dir = hydra.utils.get_original_cwd()
    # Set cache directory based on dataset
    if cfg.dataset_name == "lm1b":
        cache_dir = cfg.dataset_cache_dir  # Shared directory for lm1b
    else:
        cache_dir = os.path.join(original_dir, ".cache")  # Local for tinystories
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
        from transformers import AutoConfig, GPTNeoXForCausalLM
        config = AutoConfig.from_pretrained(cfg.model_name)
        # Use GPTNeoXForCausalLM directly for Pythia (same as hf_utils_train_model.py)
        if "pythia" in cfg.model_name.lower() or "neox" in cfg.model_name.lower():
            model = GPTNeoXForCausalLM(config).to(device)
        else:
            model = AutoModelForCausalLM.from_config(config).to(device)
    else:
        print("Loading pretrained model weights")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load datasets based on config
    print(f"Loading dataset: {cfg.dataset_name}...")

    if cfg.dataset_name == "tinystories":
        # Check for pre-tokenized cached datasets first
        from datasets import load_from_disk
        tokenized_train_path = os.path.join(cache_dir, f"tokenized_tinystories_{cfg.model_id}_train_maxlen{cfg.max_length}")
        tokenized_val_path = os.path.join(cache_dir, f"tokenized_tinystories_{cfg.model_id}_val_maxlen{cfg.max_length}")
        # Fall back to old cache path if model-specific doesn't exist (backward compat for pythia)
        if not os.path.exists(tokenized_train_path):
            old_train = os.path.join(cache_dir, f"tokenized_tinystories_train_maxlen{cfg.max_length}")
            old_val = os.path.join(cache_dir, f"tokenized_tinystories_val_maxlen{cfg.max_length}")
            if os.path.exists(old_train) and cfg.model_id == "pythia70m":
                tokenized_train_path = old_train
                tokenized_val_path = old_val

        if os.path.exists(tokenized_train_path) and os.path.exists(tokenized_val_path):
            print(f"Loading pre-tokenized TinyStories from cache...")
            tokenized_train = load_from_disk(tokenized_train_path)
            tokenized_val = load_from_disk(tokenized_val_path)
            tokenized_val = tokenized_val.select(range(min(cfg.eval_samples, len(tokenized_val))))
            # Skip the tokenization step below
            train_dataset = None
            val_dataset = None
        else:
            print("Tokenized cache not found, will create...")
            train_dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=cache_dir)
            val_dataset = load_dataset("roneneldan/TinyStories", split="validation", cache_dir=cache_dir)
        dataset_cache_prefix = "tinystories"

    elif cfg.dataset_name == "lm1b":
        # lm1b: use the configured cache directory
        lm1b_cache = cfg.dataset_cache_dir
        print(f"Using lm1b cache directory: {lm1b_cache}")

        # Check for pre-tokenized cached datasets first
        from datasets import load_from_disk
        tokenized_train_path = os.path.join(lm1b_cache, f"tokenized_lm1b_train_maxlen{cfg.max_length}")
        tokenized_val_path = os.path.join(lm1b_cache, f"tokenized_lm1b_val_maxlen{cfg.max_length}")

        if os.path.exists(tokenized_train_path) and os.path.exists(tokenized_val_path):
            print(f"Loading pre-tokenized datasets from cache...")
            tokenized_train = load_from_disk(tokenized_train_path)
            tokenized_val = load_from_disk(tokenized_val_path)
            tokenized_val = tokenized_val.select(range(min(cfg.eval_samples, len(tokenized_val))))
            # Skip the tokenization step below
            train_dataset = None
            val_dataset = None
            dataset_cache_prefix = "lm1b"
        else:
            print("Tokenized cache not found, will create...")
            dataset = load_dataset("lm1b", cache_dir=lm1b_cache, trust_remote_code=True)

            # Combine all splits (lm1b has train and test)
            from datasets import concatenate_datasets
            available_splits = list(dataset.keys())
            print(f"Available splits: {available_splits}")
            full_dataset = dataset[available_splits[0]]
            for split in available_splits[1:]:
                full_dataset = concatenate_datasets([full_dataset, dataset[split]])

            # Create train/val split (use last 10k for validation)
            val_size = min(10000, len(full_dataset) // 10)
            train_dataset = full_dataset.select(range(len(full_dataset) - val_size))
            val_dataset = full_dataset.select(range(len(full_dataset) - val_size, len(full_dataset)))
            print(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")
            dataset_cache_prefix = "lm1b"

    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset_name}. Use 'tinystories' or 'lm1b'")

    # Tokenize if not already loaded from cache
    if train_dataset is not None:
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True,
                            max_length=cfg.max_length, padding="max_length")

        print("Tokenizing datasets...")
        tokenized_train = train_dataset.map(
            tokenize_function, batched=True,
            remove_columns=train_dataset.column_names, num_proc=12,
        )
        tokenized_val = val_dataset.map(
            tokenize_function, batched=True,
            remove_columns=val_dataset.column_names, num_proc=12,
        )

        # Save to disk (so we can load quickly next time)
        if cfg.dataset_name == "lm1b":
            tokenized_train_path = os.path.join(cache_dir, f"tokenized_lm1b_train_maxlen{cfg.max_length}")
            tokenized_val_path = os.path.join(cache_dir, f"tokenized_lm1b_val_maxlen{cfg.max_length}")
        elif cfg.dataset_name == "tinystories":
            tokenized_train_path = os.path.join(cache_dir, f"tokenized_tinystories_{cfg.model_id}_train_maxlen{cfg.max_length}")
            tokenized_val_path = os.path.join(cache_dir, f"tokenized_tinystories_{cfg.model_id}_val_maxlen{cfg.max_length}")
        else:
            tokenized_train_path = None
            tokenized_val_path = None

        if tokenized_train_path:
            print(f"Saving tokenized datasets to {tokenized_train_path}...")
            tokenized_train.save_to_disk(tokenized_train_path)
            tokenized_val.save_to_disk(tokenized_val_path)

        tokenized_val = tokenized_val.select(range(min(cfg.eval_samples, len(tokenized_val))))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Prepare Hessian batch
    print("Preparing Hessian batch...")
    hessian_samples = tokenized_val.select(range(cfg.hessian_batch_size))
    hessian_batch = torch.tensor([s["input_ids"] for s in hessian_samples]).to(device)

    # Hessian evaluation steps
    hessian_spacing = cfg.get("hessian_spacing", "linear")
    hessian_log_base = cfg.get("hessian_log_base", 2)
    hessian_linear_interval = cfg.get("hessian_linear_interval", 0)
    hessian_explicit_steps = list(cfg.get("hessian_explicit_steps", []))
    if hessian_explicit_steps:
        hessian_steps = sorted(set(int(s) for s in hessian_explicit_steps if s <= cfg.total_steps))
    elif hessian_spacing == "log":
        # Log-spaced from interval with configurable base
        hessian_steps = [0]
        s = cfg.hessian_eval_interval
        while s <= cfg.total_steps:
            hessian_steps.append(s)
            s *= hessian_log_base
        # Also add linear-spaced steps if configured
        if hessian_linear_interval > 0:
            for s in range(hessian_linear_interval, cfg.total_steps + 1, hessian_linear_interval):
                hessian_steps.append(s)
        if hessian_steps[-1] != cfg.total_steps:
            hessian_steps.append(cfg.total_steps)
        hessian_steps = sorted(set(int(s) for s in hessian_steps if s <= cfg.total_steps))
    else:
        hessian_steps = [0] + list(range(cfg.hessian_eval_interval,
                                          cfg.total_steps + 1,
                                          cfg.hessian_eval_interval))
    print(f"Hessian steps ({hessian_spacing}): {hessian_steps}")

    # Check for existing checkpoints to resume from
    import glob as glob_module
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    last_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(glob_module.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
                            key=lambda x: int(x.split("-")[-1]))
        # Find the latest valid checkpoint (has trainer_state.json)
        for ckpt in reversed(checkpoints):
            trainer_state_file = os.path.join(ckpt, "trainer_state.json")
            if os.path.exists(trainer_state_file):
                last_checkpoint = ckpt
                print(f"\nFound valid checkpoint: {last_checkpoint}")
                print("Will resume training from this checkpoint.")
                break
            else:
                print(f"\nSkipping corrupted checkpoint: {ckpt} (missing trainer_state.json)")

    # Compute Hessian at step 0 (skip if resuming)
    if last_checkpoint is None:
        print("\n" + "="*60)
        print("COMPUTING HESSIAN AT STEP 0")
        print("="*60)
        try:
            compute_and_save_hessian(model, 0, hessian_batch, cfg, output_dir, device)
        except RuntimeError as e:
            print(f"Warning: Hessian computation at step 0 failed: {e}")
            print("Skipping step 0 Hessian (common with random initialization). Will compute at later steps.")
            # Ensure model is in correct state for training
            model.gradient_checkpointing_disable()
            model.train()
            torch.cuda.empty_cache()
    else:
        print("\nSkipping step 0 Hessian (resuming from checkpoint)")

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

    # Map optimizer config to HuggingFace optim string
    use_custom_optimizer = False
    optimizer_map = {
        "adam": "adamw_torch",
        "adamw": "adamw_torch",
        "sgd": "sgd",
    }
    if cfg.optimizer.lower() == "signsgd":
        optim_str = "adamw_torch"  # placeholder, will be overridden
        use_custom_optimizer = True
        print(f"Using optimizer: SignSGD (custom)")
    else:
        optim_str = optimizer_map.get(cfg.optimizer.lower(), "adamw_torch")
        print(f"Using optimizer: {cfg.optimizer} (HF: {optim_str})")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        max_steps=cfg.total_steps,
        per_device_train_batch_size=physical_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=50,
        save_steps=cfg.get("save_steps", 1000),
        save_total_limit=2,  # Keep last 2 checkpoints (fallback if latest corrupts)
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        learning_rate=cfg.learning_rate,
        optim=optim_str,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Setup callbacks and trainer
    loss_recorder = LossRecorderCallback(output_dir=output_dir)

    # Create custom optimizer if needed
    custom_optimizers = (None, None)
    if use_custom_optimizer:
        if cfg.optimizer.lower() == "signsgd":
            custom_opt = SignSGD(model.parameters(), lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)
            custom_optimizers = (custom_opt, None)

    # Evaluate at step 0
    trainer = HessianTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        optimizers=custom_optimizers,
        hessian_batch=hessian_batch,
        hessian_steps=[s for s in hessian_steps if s > 0],
        hessian_cfg=cfg,
        hessian_output_dir=output_dir,
        loss_recorder=loss_recorder,
        callbacks=[loss_recorder],
    )

    # Evaluate at step 0 (skip if resuming)
    if last_checkpoint is None:
        print("\nEvaluating at step 0...")
        eval_results = trainer.evaluate()
        loss_recorder.eval_losses.append({"step": 0, "loss": eval_results["eval_loss"]})
        print(f"Step 0 eval loss: {eval_results['eval_loss']:.4f}")
    else:
        print("\nSkipping step 0 evaluation (resuming from checkpoint)")

    # Train
    print("\n" + "="*60)
    if last_checkpoint:
        print(f"RESUMING TRAINING FROM {last_checkpoint}")
    else:
        print("STARTING TRAINING")
    print("="*60)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save loss data (before final Hessian so it's saved even if Hessian crashes)
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

    # Final Hessian if needed
    if cfg.total_steps in hessian_steps and cfg.total_steps not in trainer.hessian_results:
        try:
            compute_and_save_hessian(model, cfg.total_steps, hessian_batch, cfg, output_dir, device)
        except RuntimeError as e:
            print(f"Warning: Final Hessian computation failed: {e}")

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
