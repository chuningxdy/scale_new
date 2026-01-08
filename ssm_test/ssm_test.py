import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SSMTrainingConfig:
    """Configuration for SSM training"""
    model_size: int  # Target number of parameters (e.g., 10_000_000 for 10M)
    batch_size: int
    num_tokens: int  # Total number of tokens to train on
    seq_length: int = 4096  # Sequence length for training
    vocab_size: int = 12  # DNA tokens: A, C, G, T, N, and special tokens
    learning_rate: float = 1e-3
    warmup_steps: int = 1000
    eval_interval: int = 1000  # Evaluate every N steps
    seed: int = 42


# ============================================================================
# Evo2-style SSM Architecture Components
# ============================================================================

class Evo2Config(PretrainedConfig):
    """Configuration class for Evo2-style SSM"""
    model_type = "evo2_ssm"
    
    def __init__(
        self,
        vocab_size: int = 12,
        d_model: int = 512,
        n_layers: int = 12,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dropout: float = 0.0,
        bias: bool = False,
        conv_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.dropout = dropout
        self.bias = bias
        self.conv_bias = conv_bias


class MambaBlock(nn.Module):
    """
    Simplified Mamba/SSM block following Evo2 architecture
    """
    def __init__(self, config: Evo2Config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_state = config.d_state
        d_conv = config.d_conv
        d_inner = config.d_inner
        dt_rank = config.dt_rank
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=config.bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=config.conv_bias
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        
        # Initialize A (state transition matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Trim padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Activation
        x = nn.functional.silu(x)
        
        # SSM operation (simplified)
        x_ssm = self.ssm(x)
        
        # Gating
        x = x_ssm * nn.functional.silu(z)
        
        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x
    
    def ssm(self, x):
        """Simplified SSM operation"""
        # Project to get delta, B, C
        dt_b_c = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        
        dt, B, C = torch.split(
            dt_b_c,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1
        )
        
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        dt = nn.functional.softplus(dt)
        
        # Get A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Simplified SSM computation (selective scan)
        # In practice, this would use efficient CUDA kernels
        y = self.selective_scan(x, dt, A, B, C, self.D)
        
        return y
    
    def selective_scan(self, x, dt, A, B, C, D):
        """
        Simplified selective scan (not optimized)
        In production, use optimized CUDA kernels like in Mamba
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            # Discretize A and B
            dt_t = dt[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)
            A_discrete = torch.exp(A.unsqueeze(0) * dt_t)  # (batch, d_inner, d_state)
            B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            B_discrete = B_t * dt_t
            
            # State update
            h = h * A_discrete + x[:, t, :].unsqueeze(-1) * B_discrete
            
            # Output
            C_t = C[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            y_t = (h * C_t).sum(dim=-1) + x[:, t, :] * D
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        return y


class Evo2Block(nn.Module):
    """Complete Evo2 transformer block with SSM and normalization"""
    def __init__(self, config: Evo2Config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mamba = MambaBlock(config)
        
    def forward(self, x):
        return x + self.mamba(self.norm(x))


class Evo2Model(PreTrainedModel):
    """Complete Evo2 model"""
    config_class = Evo2Config
    
    def __init__(self, config: Evo2Config):
        super().__init__(config)
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            Evo2Block(config) for _ in range(config.n_layers)
        ])
        self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        """
        input_ids: (batch, seq_len)
        labels: (batch, seq_len) - for computing loss
        """
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return {"loss": loss, "logits": logits}
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Dataset
# ============================================================================

class GenomicsDataset(Dataset):
    """
    Synthetic genomics dataset for demonstration
    Replace with actual genomics data loading
    """
    def __init__(self, num_samples: int, seq_length: int, vocab_size: int = 12):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # For real use, load from files like FASTA
        # Here we generate synthetic data
        # Vocab: 0-3 (A,C,G,T), 4 (N), 5-11 (special tokens)
        self.data = torch.randint(0, 4, (num_samples, seq_length))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # For language modeling, input is sequence, label is shifted by 1
        return {
            "input_ids": sequence[:-1],
            "labels": sequence[1:]
        }


# ============================================================================
# Model Size Calculator
# ============================================================================

def calculate_model_dims(target_params: int, vocab_size: int = 12, 
                         n_layers: int = None) -> Dict:
    """
    Calculate model dimensions to approximately match target parameters
    
    Parameter count formula for Evo2-style model:
    - Embedding: vocab_size * d_model
    - Per layer: ~10 * d_model^2 (approximation for SSM block)
    - LM head: tied with embedding, so no extra params
    """
    
    # Try different layer counts if not specified
    if n_layers is None:
        layer_options = [6, 8, 12, 16, 24]
    else:
        layer_options = [n_layers]
    
    best_config = None
    best_diff = float('inf')
    
    for n_layers_try in layer_options:
        # Solve for d_model
        # target_params ≈ vocab_size * d_model + n_layers * 10 * d_model^2
        # Simplify: 10 * n_layers * d_model^2 + vocab_size * d_model - target_params = 0
        
        a = 10 * n_layers_try
        b = vocab_size
        c = -target_params
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            continue
            
        d_model = int((-b + math.sqrt(discriminant)) / (2*a))
        
        # Round to nearest multiple of 64 for efficiency
        d_model = max(64, (d_model // 64) * 64)
        
        # Create config and check actual param count
        config = Evo2Config(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers_try,
            d_state=min(64, d_model // 8),
            d_conv=4,
            expand=2,
        )
        
        # Create temporary model to count params
        temp_model = Evo2Model(config)
        actual_params = temp_model.count_parameters()
        
        diff = abs(actual_params - target_params)
        
        if diff < best_diff:
            best_diff = diff
            best_config = (config, actual_params)
        
        del temp_model
    
    if best_config is None:
        raise ValueError(f"Could not find suitable config for {target_params} parameters")
    
    config, actual_params = best_config
    print(f"Target params: {target_params:,}")
    print(f"Actual params: {actual_params:,}")
    print(f"Difference: {abs(actual_params - target_params):,} ({100*abs(actual_params - target_params)/target_params:.2f}%)")
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}, d_state={config.d_state}")
    
    return config


# ============================================================================
# Training Loop
# ============================================================================

def get_lr(step: int, warmup_steps: int, max_lr: float, total_steps: int):
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    elif step > total_steps:
        return 0.1 * max_lr
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 * max_lr + 0.5 * (max_lr - 0.1 * max_lr) * (1 + math.cos(math.pi * progress))


def train_ssm(
    model_size: int,
    batch_size: int,
    num_tokens: int,
    seq_length: int = 4096,
    vocab_size: int = 12,
    learning_rate: float = 1e-3,
    warmup_steps: int = 1000,
    eval_interval: int = 1000,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Main training function
    
    Args:
        model_size: Target number of parameters (e.g., 10_000_000 for 10M)
        batch_size: Batch size for training
        num_tokens: Total number of tokens to train on
        seq_length: Sequence length for each sample
        vocab_size: Vocabulary size
        learning_rate: Maximum learning rate
        warmup_steps: Number of warmup steps
        eval_interval: Evaluate every N steps
        seed: Random seed
        device: Device to train on
    
    Returns:
        loss_curve: List of (step, loss) tuples
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"Starting SSM Pre-training")
    print(f"{'='*80}")
    print(f"Target model size: {model_size:,} parameters")
    print(f"Batch size: {batch_size}")
    print(f"Total tokens: {num_tokens:,}")
    print(f"Sequence length: {seq_length}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Calculate model dimensions
    config = calculate_model_dims(model_size, vocab_size)
    
    # Create model
    print("\nInitializing model...")
    model = Evo2Model(config)
    model = model.to(device)
    
    actual_params = model.count_parameters()
    print(f"Model initialized with {actual_params:,} parameters")
    
    # Calculate training steps
    tokens_per_batch = batch_size * (seq_length - 1)  # -1 for shifted labels
    total_steps = math.ceil(num_tokens / tokens_per_batch)
    num_samples = total_steps * batch_size
    
    print(f"\nTraining for {total_steps} steps ({num_samples} samples)")
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = GenomicsDataset(num_samples, seq_length, vocab_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to > 0 for faster data loading
        pin_memory=True if device == "cuda" else False
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Training loop
    model.train()
    loss_curve = []
    step = 0
    total_tokens_seen = 0
    
    print("\nStarting training...\n")
    
    pbar = tqdm(total=total_steps, desc="Training")
    
    for batch in dataloader:
        if step >= total_steps:
            break
        
        # Move to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Learning rate schedule
        lr = get_lr(step, warmup_steps, learning_rate, total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update counters
        total_tokens_seen += tokens_per_batch
        step += 1
        
        # Logging
        if step % eval_interval == 0 or step == total_steps:
            loss_curve.append((step, loss.item()))
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.2e}',
                'tokens': f'{total_tokens_seen:,}'
            })
        
        pbar.update(1)
    
    pbar.close()
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Total steps: {step}")
    print(f"Total tokens: {total_tokens_seen:,}")
    print(f"Final loss: {loss_curve[-1][1]:.4f}")
    print(f"{'='*80}\n")
    
    return loss_curve, model


def plot_loss_curve(loss_curve, save_path: str = "loss_curve.png"):
    """Plot and save the loss curve"""
    steps, losses = zip(*loss_curve)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linewidth=2)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("SSM Pre-training Loss Curve", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Loss curve saved to {save_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Example usage
    
    # Configuration
    MODEL_SIZE = 10_000_000  # 10M parameters
    BATCH_SIZE = 8
    NUM_TOKENS = 1_000_000  # 1M tokens for quick demo (use 100M+ for real training)
    SEQ_LENGTH = 1024  # Shorter for demo (use 4096+ for real training)
    
    # Train model
    loss_curve, model = train_ssm(
        model_size=MODEL_SIZE,
        batch_size=BATCH_SIZE,
        num_tokens=NUM_TOKENS,
        seq_length=SEQ_LENGTH,
        learning_rate=1e-3,
        warmup_steps=100,
        eval_interval=100,
        seed=42
    )
    
    # Plot results
    plot_loss_curve(loss_curve)
    
    # Print final results
    print("\nLoss Curve (step, loss):")
    print("-" * 40)
    for step, loss in loss_curve:
        print(f"Step {step:6d}: {loss:.6f}")