import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import gc
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

# ============== CONFIGURATION ==============
MODEL_NAME = "roneneldan/TinyStories-8M"
OUTPUT_DIR = "./tinystories_training"
HESSIAN_OUTPUT_DIR = "./hessian_analysis"
TOTAL_STEPS = 5000

HESSIAN_EVAL_INTERVAL = 500  # Compute Hessian every N steps
HESSIAN_EVAL_STEPS = [0] + list(range(HESSIAN_EVAL_INTERVAL, TOTAL_STEPS + 1, HESSIAN_EVAL_INTERVAL))
TOP_K = 10  # Number of top eigenvalues to compute with exact Lanczos
MAX_ITER = 4 * TOP_K
BATCH_SIZE = 256  # Optimal from tuning
HESSIAN_BATCH_SIZE = 64  # Batch size for Hessian computation (H100 can handle larger)
SLQ_ITER = 200  # Lanczos iterations for stochastic estimation
MIN_FIT_INDEX = 20 

# ============== SETUP ==============
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise ValueError("no GPU")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HESSIAN_OUTPUT_DIR, exist_ok=True)

# Disable efficient/flash attention (they don't support second-order gradients)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# ============== LOAD MODEL & TOKENIZER ==============
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# ============== LOAD DATASET ==============
print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )

print("Tokenizing train dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=4,
)

# Load and tokenize validation dataset
print("Loading validation dataset...")
val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
print("Tokenizing validation dataset...")
tokenized_val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    num_proc=4,
)
# Select subset for faster evaluation
EVAL_SAMPLES = 1000
tokenized_val_dataset = tokenized_val_dataset.select(range(EVAL_SAMPLES))
print(f"Using {EVAL_SAMPLES} samples for evaluation")

# Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============== HESSIAN ANALYSIS UTILITIES ==============

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
        elif hasattr(item, 'numel'):  # torch tensor
            if item.numel() == 1:
                result.append(item.item())
        elif hasattr(item, 'item'):  # numpy scalar
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


def compute_and_save_hessian(model, step, hessian_batch):
    """
    Compute Hessian spectrum and save results.

    Args:
        model: The model to analyze
        step: Current training step
        hessian_batch: Input tensor for Hessian computation
    """
    print(f"\n{'='*50}")
    print(f"Computing Hessian analysis at step {step}")
    print(f"{'='*50}")

    # Prepare model for Hessian computation
    model.eval()
    model.gradient_checkpointing_enable()
    wrapped_model = ModelWrapper(model)

    # Compute Hessian
    hessian_comp = hessian(
        wrapped_model, criterion,
        data=(hessian_batch, hessian_batch),
        cuda=(device == "cuda")
    )

    # Stochastic Lanczos Quadrature for density estimation
    print(f"Computing SLQ density (iter={SLQ_ITER})...")
    density_eigenvalues, density_weights = hessian_comp.density(iter=SLQ_ITER)

    # Exact Lanczos for top-k eigenvalues
    print(f"Computing top {TOP_K} eigenvalues...")
    top_eigenvalues_raw = hessian_comp.eigenvalues(top_n=TOP_K, maxIter=MAX_ITER)
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
    indices = ranks * total_params  # Convert to index

    # Fit power law: lambda = c * i^(-p) for indices >= 1
    fit_mask = indices >= MIN_FIT_INDEX
    if np.sum(fit_mask) > 2:
        log_indices = np.log(indices[fit_mask])
        log_evs = np.log(evs_final[fit_mask])
        # Linear fit: log(lambda) = log(c) - p * log(i)
        coeffs = np.polyfit(log_indices, log_evs, 1)
        p_fit = -coeffs[0]  # power
        c_fit = np.exp(coeffs[1])  # constant
        print(f"Power law fit: lambda = {c_fit:.4f} * i^(-{p_fit:.4f})")
    else:
        p_fit, c_fit = None, None

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
            "top_k": TOP_K,
            "slq_iter": SLQ_ITER,
            "batch_size": HESSIAN_BATCH_SIZE,
            "total_params": total_params,
        }
    }

    data_path = os.path.join(HESSIAN_OUTPUT_DIR, f"hessian_step_{step}.json")
    with open(data_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved data to {data_path}")

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # A. Density Plot
    grid, density = get_density(density_eigenvalues[0], density_weights[0])
    ax1.plot(grid, density, color='blue', lw=2)
    ax1.set_title(f"Hessian Eigenvalue Density (Step {step})")
    ax1.set_xlabel(r"Eigenvalue $\lambda$")
    ax1.set_ylabel(r"Density $\rho(\lambda)$")
    ax1.grid(True, alpha=0.3)

    # B. Log-Log Plot
    ax2.loglog(indices, evs_final, marker='o', linestyle='None', alpha=0.6,
               markersize=4, label='SLQ Density')

    if top_eigenvalues:
        top_evs = np.array(sorted(top_eigenvalues, reverse=True))
        top_indices = np.arange(1, len(top_evs) + 1)  # Simple integer indices
        ax2.loglog(top_indices, top_evs, marker='+', linestyle='None', color='red',
                   markersize=4, label='Top-k Lanczos', zorder=5, alpha=0.6)

    # Plot fitted power law
    if c_fit is not None and p_fit is not None:
        fit_x = np.logspace(0, np.log10(total_params), 100)  # From 1 to total_params
        fit_y = c_fit * fit_x ** (-p_fit)
        ax2.loglog(fit_x, fit_y, 'g--', lw=2, label=f'Fit: $\\lambda = {c_fit:.2f} \\cdot i^{{-{p_fit:.2f}}}$')

    ax2.set_title(f"Log-Log Spectrum (Step {step})")
    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_ylabel(r"Eigenvalue $\lambda$")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    # Fixed axis ranges for consistent comparison across steps
    ax2.set_xlim(1e-1, 1e7)
    ax2.set_ylim(1e-3, 1e5)

    plt.tight_layout()
    plot_path = os.path.join(HESSIAN_OUTPUT_DIR, f"hessian_step_{step}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Print summary
    print(f"Analysis complete at step {step}:")
    print(f"  Max Eigenvalue (SLQ): {max(evs_final):.4f}")
    if top_eigenvalues:
        print(f"  Max Eigenvalue (Lanczos): {max(top_eigenvalues):.4f}")

    # Return model to training mode
    model.gradient_checkpointing_disable()
    model.train()

    # Clean up GPU memory
    del hessian_comp
    del wrapped_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory cleared. Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    return save_data


# ============== PREPARE HESSIAN BATCH ==============
# Use a fixed batch for consistent Hessian evaluation across steps
print("Preparing fixed batch for Hessian computation...")
hessian_samples = tokenized_dataset.select(range(HESSIAN_BATCH_SIZE))
hessian_batch = torch.tensor([s["input_ids"] for s in hessian_samples]).to(device)

# ============== CUSTOM TRAINER WITH HESSIAN CALLBACKS ==============

class LossRecorderCallback(TrainerCallback):
    """Callback to record training and evaluation losses."""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_losses.append({
                    "step": state.global_step,
                    "loss": logs["loss"]
                })
            if "eval_loss" in logs:
                self.eval_losses.append({
                    "step": state.global_step,
                    "loss": logs["eval_loss"]
                })


class HessianTrainer(Trainer):
    """Custom Trainer that computes Hessian at specified steps."""

    def __init__(self, *args, hessian_batch=None, hessian_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hessian_batch = hessian_batch
        self.hessian_steps = hessian_steps or []
        self.hessian_results = {}

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Check if we should compute Hessian before this step
        current_step = self.state.global_step

        if current_step in self.hessian_steps and current_step not in self.hessian_results:
            result = compute_and_save_hessian(model, current_step, self.hessian_batch)
            self.hessian_results[current_step] = result

        return super().training_step(model, inputs, num_items_in_batch)


# ============== TRAINING ==============

# Compute Hessian at step 0 (before training)
print("\n" + "="*60)
print("COMPUTING HESSIAN AT STEP 0 (BEFORE TRAINING)")
print("="*60)
compute_and_save_hessian(model, 0, hessian_batch)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=TOTAL_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=50,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    learning_rate=1e-4,  # Optimal from tuning
    warmup_steps=100,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False,
)

# Create loss recorder callback
loss_recorder = LossRecorderCallback()

# Create trainer
trainer = HessianTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    hessian_batch=hessian_batch,
    hessian_steps=[s for s in HESSIAN_EVAL_STEPS if s > 0],  # Exclude 0, already done
    callbacks=[loss_recorder],
)

# Evaluate at step 0 (before training)
print("\n" + "="*60)
print("EVALUATING AT STEP 0 (BEFORE TRAINING)")
print("="*60)
eval_results = trainer.evaluate()
loss_recorder.eval_losses.append({"step": 0, "loss": eval_results["eval_loss"]})
print(f"Step 0 eval loss: {eval_results['eval_loss']:.4f}")

# Train
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
trainer.train()

# Compute final Hessian if not already done
if TOTAL_STEPS in HESSIAN_EVAL_STEPS and TOTAL_STEPS not in trainer.hessian_results:
    print("\n" + "="*60)
    print(f"COMPUTING HESSIAN AT STEP {TOTAL_STEPS} (AFTER TRAINING)")
    print("="*60)
    compute_and_save_hessian(model, TOTAL_STEPS, hessian_batch)

# ============== SAVE LOSS DATA AND PLOT ==============

# Save loss data to JSON
loss_data = {
    "train_losses": loss_recorder.train_losses,
    "eval_losses": loss_recorder.eval_losses,
}
loss_data_path = os.path.join(HESSIAN_OUTPUT_DIR, "loss_history.json")
with open(loss_data_path, "w") as f:
    json.dump(loss_data, f, indent=2)
print(f"Saved loss history to {loss_data_path}")

# Create loss curve plot
fig, ax = plt.subplots(figsize=(10, 6))

if loss_recorder.train_losses:
    train_steps = [x["step"] for x in loss_recorder.train_losses]
    train_loss_vals = [x["loss"] for x in loss_recorder.train_losses]
    ax.plot(train_steps, train_loss_vals, 'b-', alpha=0.7, label='Train Loss')

if loss_recorder.eval_losses:
    eval_steps = [x["step"] for x in loss_recorder.eval_losses]
    eval_loss_vals = [x["loss"] for x in loss_recorder.eval_losses]
    ax.plot(eval_steps, eval_loss_vals, 'r-', marker='o', markersize=4, label='Eval Loss')

ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training and Evaluation Loss Curves")
ax.legend()
ax.grid(True, alpha=0.3)

loss_plot_path = os.path.join(HESSIAN_OUTPUT_DIR, "loss_curves.png")
plt.savefig(loss_plot_path, dpi=150)
plt.close()
print(f"Saved loss plot to {loss_plot_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Hessian analysis saved to: {HESSIAN_OUTPUT_DIR}")
print(f"Loss history saved to: {loss_data_path}")
print(f"Model checkpoints saved to: {OUTPUT_DIR}")
