# Hessian Training Pipeline

Automated pipeline for training TinyStories models with Hessian analysis, using Hydra for configuration management and hyperparameter sweeps.

## Installation

```bash
pip install hydra-core omegaconf
```

## Usage

### Single Run

```bash
# Use default config (bs=256, lr=1e-4)
python run_pipeline.py

# Override hyperparameters
python run_pipeline.py batch_size=128 learning_rate=5e-4

# Override multiple parameters
python run_pipeline.py batch_size=256 learning_rate=1e-4 total_steps=10000
```

### Hyperparameter Sweep

```bash
# Sweep over batch sizes
python run_pipeline.py --multirun batch_size=128,256,512

# Sweep over learning rates
python run_pipeline.py --multirun learning_rate=1e-4,3e-4,5e-4

# Grid sweep (all combinations)
python run_pipeline.py --multirun batch_size=128,256 learning_rate=1e-4,5e-4
```

## Output Structure

Results are organized by hyperparameters:

```
outputs/
└── run_pipeline/
    ├── bs128_lr0.0001/
    │   ├── config.yaml              # Full configuration
    │   ├── loss_history.json        # Train/eval loss per step
    │   ├── loss_curves.png          # Loss plot
    │   ├── hessian_step_0.json      # Hessian data at step 0
    │   ├── hessian_step_0.png       # Hessian plot at step 0
    │   ├── hessian_step_500.json
    │   ├── hessian_step_500.png
    │   ├── ...
    │   ├── eigenvalue_trajectories.json
    │   ├── eigenvalue_trajectories.png
    │   └── checkpoints/             # Model checkpoints
    ├── bs256_lr0.0001/
    │   └── ...
    └── bs256_lr0.0005/
        └── ...
```

## Configuration

Edit `config/config.yaml` to change defaults:

```yaml
# Training hyperparameters
batch_size: 256
learning_rate: 1e-4
total_steps: 5000

# Hessian settings
hessian_eval_interval: 500   # Compute Hessian every N steps
hessian_batch_size: 64       # Batch size for Hessian computation
top_k: 10                    # Top eigenvalues via exact Lanczos
slq_iter: 200                # SLQ iterations

# Analysis settings
eigenvalue_indices: [1, 2, 4, 8, 16, 32, 64, 128]
lanczos_indices: [1, 2, 3]   # Use Lanczos for these, SLQ for rest
eigenvalue_source: "slq_interp"  # "slq", "slq_interp", or "lanczos"
```

## Key Features

1. **Hessian Analysis**: Computes eigenvalue spectrum at regular intervals
2. **Power Law Fitting**: Fits λ = c·i^(-p) to the spectrum
3. **Mixed Sources**: Use Lanczos for top eigenvalues, SLQ interpolation for bulk
4. **Trajectory Tracking**: Visualizes how eigenvalues evolve during training
5. **Consistent Plots**: Fixed axis ranges for easy comparison across runs
