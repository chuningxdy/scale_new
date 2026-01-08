#!/usr/bin/env python3
"""Debug script to check StripedHyena model outputs"""

import os
os.environ["HF_HOME"] = "/mfs1/datasets/pile/huggingface"

import torch
import numpy as np

# Load the model and config that was just created
model_path = "outputs/nn_hf/temp/output/../"  # Adjust if needed

# Or recreate the model inline for testing
from transformers import AutoConfig, AutoModelForCausalLM

print("Loading config...")
config = AutoConfig.from_pretrained(
    "togethercomputer/StripedHyena-Hessian-7B",
    trust_remote_code=True
)

# Apply our modifications
config.vocab_size = 5
config.hidden_size = 320  # Adjusted for flash attention
config.num_filters = 320
config.num_layers = 8
config.num_attention_heads = 5
config.inner_mlp_size = 800
config.attn_layer_idxs = [1, 3, 5, 7]
config.hyena_layer_idxs = [0, 2, 4, 6]
config.make_vocab_size_divisible_by = 1
config.use_flash_attention_2 = False
config.use_flash_attn = False

print(f"Config vocab_size: {config.vocab_size}")
print(f"Config hidden_size: {config.hidden_size}")
print(f"Config num_layers: {config.num_layers}")

# Convert dtype strings to torch dtypes for model creation
# Use float32 for Hyena blocks to avoid numerical instability
config.hyena_block_dtype = torch.float32
config.attn_block_dtype = torch.bfloat16
config.mlp_dtype = torch.bfloat16

print("\nCreating model...")
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# Stable pole initialization for Hyena filters
# The poles must have |poles| < 1 so that log(poles) has negative real part,
# ensuring exp(log_poles * t) decays over time instead of exploding.
print("Applying stable pole initialization for Hyena filters...")
with torch.no_grad():
    for name, module in model.named_modules():
        if hasattr(module, 'poles') and hasattr(module, 'residues'):
            # poles shape: [num_systems, state_size, 1, 2] where last dim is [real, imag]
            poles_complex = torch.view_as_complex(module.poles.data.float())
            magnitude = poles_complex.abs()
            phase = poles_complex / (magnitude + 1e-8)
            # Clamp magnitude to [0.5, 0.99] for stable but expressive filters
            stable_magnitude = 0.5 + 0.49 * torch.sigmoid(magnitude)  # Maps to ~[0.62, 0.99]
            stable_poles = stable_magnitude * phase
            module.poles.data = torch.view_as_real(stable_poles).to(module.poles.dtype)
            print(f"  Fixed poles in {name}: magnitude range [{stable_magnitude.min():.4f}, {stable_magnitude.max():.4f}]")

model = model.cuda()
model.eval()

print(f"Model vocab_size attr: {model.vocab_size}")
print(f"Model config.vocab_size: {model.config.vocab_size}")

# Create test input
batch_size = 2
seq_len = 128
input_ids = torch.randint(0, 5, (batch_size, seq_len), dtype=torch.long).cuda()
labels = input_ids.clone()

print(f"\nInput shape: {input_ids.shape}")
print(f"Input range: [{input_ids.min()}, {input_ids.max()}]")
print(f"Labels shape: {labels.shape}")

# Check embedding layer first
print("\n--- Checking embedding layer ---")
with torch.no_grad():
    embeddings = model.backbone.embedding_layer.embed(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings has NaN: {torch.isnan(embeddings).any()}")
    print(f"Embeddings min/max: {embeddings.min():.4f} / {embeddings.max():.4f}")

# Check each block
print("\n--- Checking blocks one by one ---")
x = embeddings
with torch.no_grad():
    for i, block in enumerate(model.backbone.blocks):
        block_type = "Attention" if i in config.attn_layer_idxs else "Hyena"
        x_out, _ = block(x, inference_params=None, padding_mask=None)
        has_nan = torch.isnan(x_out).any()
        print(f"Block {i} ({block_type}): has_nan={has_nan}, min={x_out.min():.4f}, max={x_out.max():.4f}")
        if has_nan:
            print(f"  NaN first appeared in block {i} ({block_type})!")
            # Debug inside the Hyena block
            if block_type == "Hyena":
                print("\n--- Debugging Hyena block internals ---")
                # pre_norm
                pre_norm_out = block.pre_norm(x)
                print(f"pre_norm: has_nan={torch.isnan(pre_norm_out).any()}, min={pre_norm_out.min():.4f}, max={pre_norm_out.max():.4f}")

                # projections
                proj_out = block.projections(pre_norm_out)
                print(f"projections: has_nan={torch.isnan(proj_out).any()}, min={proj_out.min():.4f}, max={proj_out.max():.4f}")

                # filter
                filter_out, _ = block.filter(proj_out, inference_params=None, padding_mask=None)
                print(f"filter: has_nan={torch.isnan(filter_out).any()}, min={filter_out.min():.4f}, max={filter_out.max():.4f}")

                # Check filter internals
                print("\n--- Filter internals ---")
                filt = block.filter
                print(f"poles shape: {filt.poles.shape}, has_nan={torch.isnan(filt.poles).any()}")
                print(f"poles min/max: {filt.poles.min():.4f} / {filt.poles.max():.4f}")
                print(f"residues shape: {filt.residues.shape}, has_nan={torch.isnan(filt.residues).any()}")
                print(f"residues min/max: {filt.residues.min():.4f} / {filt.residues.max():.4f}")

                # Check D parameter
                print(f"D shape: {filt.D.shape}, has_nan={torch.isnan(filt.D).any()}")

                # Check short filter
                print(f"short_filter_weight shape: {filt.short_filter_weight.shape}")
                print(f"short_filter_weight has_nan: {torch.isnan(filt.short_filter_weight).any()}")

                # Debug the filter computation
                print("\n--- Filter computation debug ---")
                # The poles/residues are stored as [num_systems, state_size, 1, 2] where last dim is [real, imag]
                filter_dtype = torch.float32
                poles_complex = torch.view_as_complex(filt.poles.to(filter_dtype))
                residues_complex = torch.view_as_complex(filt.residues.to(filter_dtype))
                print(f"poles_complex shape: {poles_complex.shape}")
                print(f"poles_complex magnitude range: {poles_complex.abs().min():.6f} / {poles_complex.abs().max():.6f}")

                # Check for very small magnitudes that would cause log issues
                small_mag = (poles_complex.abs() < 1e-6).sum()
                print(f"Poles with magnitude < 1e-6: {small_mag}")

                # Log of poles
                log_poles = poles_complex.log()
                print(f"log_poles has_nan: {torch.isnan(log_poles.real).any() or torch.isnan(log_poles.imag).any()}")
                print(f"log_poles real range: {log_poles.real.min():.4f} / {log_poles.real.max():.4f}")

                # Time tensor
                L = proj_out.shape[1]  # sequence length
                t = torch.arange(L, device=filt.poles.device)[None, None]
                print(f"t shape: {t.shape}, L={L}")

                # log_poles * t
                log_poles_t = log_poles * t
                print(f"log_poles*t has_nan: {torch.isnan(log_poles_t.real).any() or torch.isnan(log_poles_t.imag).any()}")
                print(f"log_poles*t real max: {log_poles_t.real.max():.4f}")

                # exp(log_poles * t) - this is likely where overflow happens
                exp_term = (log_poles * t).exp()
                print(f"exp term has_nan: {torch.isnan(exp_term.real).any() or torch.isnan(exp_term.imag).any()}")
                print(f"exp term has_inf: {torch.isinf(exp_term.real).any() or torch.isinf(exp_term.imag).any()}")
            break
        x = x_out

# Forward pass
print("\n--- Full forward pass ---")
with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=labels)

print(f"\nLogits shape: {outputs.logits.shape}")
print(f"Logits dtype: {outputs.logits.dtype}")
print(f"Logits has NaN: {torch.isnan(outputs.logits).any()}")
print(f"Logits has Inf: {torch.isinf(outputs.logits).any()}")
print(f"Logits min: {outputs.logits.min()}")
print(f"Logits max: {outputs.logits.max()}")
print(f"Logits mean: {outputs.logits.mean()}")

print(f"\nLoss: {outputs.loss}")
print(f"Loss is NaN: {torch.isnan(outputs.loss) if outputs.loss is not None else 'N/A'}")

# Manual loss computation to debug
print("\n--- Manual loss computation ---")
logits = outputs.logits
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

print(f"shift_logits shape: {shift_logits.shape}")
print(f"shift_labels shape: {shift_labels.shape}")
print(f"Expected vocab_size for view: {config.vocab_size}")
print(f"Actual last dim of logits: {shift_logits.shape[-1]}")

if shift_logits.shape[-1] != config.vocab_size:
    print(f"WARNING: Logits vocab dim ({shift_logits.shape[-1]}) != config.vocab_size ({config.vocab_size})")

# Try the view operation
try:
    shift_logits_flat = shift_logits.view(-1, config.vocab_size)
    print(f"View succeeded: {shift_logits_flat.shape}")
except Exception as e:
    print(f"View failed: {e}")

# Compute loss manually
import torch.nn.functional as F
shift_logits_flat = shift_logits.view(-1, shift_logits.shape[-1])  # Use actual dim
shift_labels_flat = shift_labels.view(-1)
manual_loss = F.cross_entropy(shift_logits_flat, shift_labels_flat)
print(f"Manual loss (correct vocab dim): {manual_loss}")
