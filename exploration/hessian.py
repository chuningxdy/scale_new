import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyhessian import hessian

# ============== CONFIGURATION ==============
model_name = "roneneldan/TinyStories-8M"

# Choose which methods to compute
COMPUTE_SLQ = True   # Stochastic Lanczos Quadrature for density estimation
COMPUTE_LANCZOS = True # Exact Lanczos for top-k eigenvalues

# SLQ parameters
SLQ_ITER = 200  # Lanczos iterations for stochastic estimation

# Exact Lanczos parameters
TOP_K = 200     # Number of top eigenvalues to compute
MAX_ITER = 200  # Power iteration steps per eigenvalue

# ============== SETUP ==============
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    raise ValueError("no GPU")

# Disable efficient/flash attention (they don't support second-order gradients)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# ============== LOAD MODEL & TOKENIZER ==============
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.gradient_checkpointing_enable()
model.eval()

# ============== PREPARE CALIBRATION BATCH ==============
base_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold, but it shines bright.",
    "The only thing we have to fear is fear itself.",
    "In the beginning, there was nothing but darkness.",
    "Knowledge is power, and power corrupts absolutely.",
    "Time flies like an arrow, fruit flies like a banana.",
    "The early bird catches the worm, but the second mouse gets the cheese.",
    "Actions speak louder than words, but silence speaks loudest.",
    "Fortune favors the bold, but wisdom favors the patient.",
    "Where there is smoke, there is fire burning bright.",
    "A picture is worth a thousand words and memories.",
    "The pen is mightier than the sword in the end.",
    "When in Rome, do as the Romans do every day.",
    "Better late than never, but never late is better.",
]
texts = base_texts  # 16 samples
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]  # Shape: (16, seq_len)

# ============== MODEL WRAPPER FOR PYHESSIAN ==============
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        return outputs.logits, outputs.loss

wrapped_model = ModelWrapper(model)

def criterion(outputs, labels):
    logits, loss = outputs
    return loss

# ============== HELPER FUNCTIONS ==============
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

# ============== COMPUTE HESSIAN ==============
hessian_comp = hessian(wrapped_model, criterion, data=(input_ids, input_ids), cuda=(device=="cuda"))

# Results storage
density_eigenvalues = None
density_weights = None
top_eigenvalues = None

# Compute SLQ density
if COMPUTE_SLQ:
    print(f"Computing SLQ density (iter={SLQ_ITER})...")
    density_eigenvalues, density_weights = hessian_comp.density(iter=SLQ_ITER)

# Compute exact Lanczos top-k
if COMPUTE_LANCZOS:
    print(f"Computing top {TOP_K} eigenvalues (maxIter={MAX_ITER})...")
    top_eigenvalues_raw = hessian_comp.eigenvalues(top_n=TOP_K, maxIter=MAX_ITER)
    top_eigenvalues = flatten(top_eigenvalues_raw)
    top_eigenvalues = [ev for ev in top_eigenvalues if ev > 0]  # Keep only positive
    print(f"Top {len(top_eigenvalues)} positive eigenvalues: {top_eigenvalues}")

# ============== PLOTTING ==============
# Determine number of subplots based on what's computed
num_plots = (1 if COMPUTE_SLQ else 0) + 1  # Always have log-log plot if anything computed
if not COMPUTE_SLQ and not COMPUTE_LANCZOS:
    print("Nothing to compute! Set COMPUTE_SLQ and/or COMPUTE_LANCZOS to True.")
    exit()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax1, ax2 = axes

# A. Density Plot (only if SLQ computed)
if COMPUTE_SLQ:
    grid, density = get_density(density_eigenvalues[0], density_weights[0])
    ax1.plot(grid, density, color='blue', lw=2)
    ax1.set_title("Hessian Eigenvalue Density (ESD)")
    ax1.set_xlabel(r"Eigenvalue $\lambda$")
    ax1.set_ylabel(r"Density $\rho(\lambda)$")
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, "SLQ not computed", ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title("Hessian Eigenvalue Density (ESD) - Skipped")

# B. Log-Log Plot
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if COMPUTE_SLQ:
    evs = np.array(density_eigenvalues[0])
    wts = np.array(density_weights[0])
    idx = np.argsort(evs)[::-1]
    evs_sorted = evs[idx]
    wts_sorted = wts[idx]
    mask = evs_sorted > 1e-6
    evs_final = evs_sorted[mask]
    ranks = np.cumsum(wts_sorted[mask]) * total_params  # Convert to index
    ax2.loglog(ranks, evs_final, marker='o', linestyle='None', alpha=0.6, markersize=4, label='SLQ Density')

if COMPUTE_LANCZOS and top_eigenvalues:
    top_evs = np.array(sorted(top_eigenvalues, reverse=True))
    top_indices = np.arange(1, len(top_evs) + 1)  # Simple integer indices
    ax2.loglog(top_indices, top_evs, marker='+', linestyle='None', color='red',
               markersize=4, label='Top-k Lanczos', zorder=5, alpha=0.6)

ax2.set_title("Log-Log Spectrum (Index vs. Value)")
ax2.set_xlabel("Eigenvalue Index")
ax2.set_ylabel(r"Eigenvalue $\lambda$")
if COMPUTE_SLQ or COMPUTE_LANCZOS:
    ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
# put date time in the title
plt.savefig("transformer_esd_analysis4.png")
plt.show()

# ============== SUMMARY ==============
print(f"\nAnalysis complete.")
if COMPUTE_SLQ:
    print(f"  Max Eigenvalue (SLQ): {max(evs_final):.4f}")
if COMPUTE_LANCZOS and top_eigenvalues:
    print(f"  Max Eigenvalue (Lanczos): {max(top_eigenvalues):.4f}")
