# StripedHyena Configuration Changes

This document describes the configuration changes applied when using the Together AI StripedHyena model for genomics training.

## Base Model

- **Source**: `togethercomputer/StripedHyena-Hessian-7B` from HuggingFace
- **Usage**: Architecture code only (weights randomly initialized)

## Configuration Changes

| Parameter | Original (7B) | Our Change | Reason |
|-----------|---------------|------------|--------|
| `vocab_size` | 32000 | 5 | Genomics vocab: A=0, C=1, G=2, T=3, N=4 |
| `hidden_size` | 4096 | Predicted | Scaled to target parameter count |
| `num_filters` | 4096 | = hidden_size | Must equal hidden_size (assertion in model) |
| `num_layers` | 32 | Predicted | Scaled to target parameter count |
| `num_attention_heads` | 32 | Adjusted | Must satisfy: hidden_size % head_dim == 0, head_dim ∈ {32,64,128,256} |
| `inner_mlp_size` | 14336 | ~2.5x hidden_size | Gated MLP expansion factor |
| `attn_layer_idxs` | [1,3,5,...] | Odd layers | Alternating Hyena/Attention pattern |
| `hyena_layer_idxs` | [0,2,4,...] | Even layers | Alternating Hyena/Attention pattern |
| `make_vocab_size_divisible_by` | 8 | 1 | Avoid vocab size mismatch in loss computation |
| `tie_word_embeddings` | True | False | StripedHyena has bug with tied embeddings |
| `proj_groups` | 4 | 1 | Standard MHA instead of GQA (see below) |
| `hyena_block_dtype` | bfloat16 | float32 | Numerical stability for Hyena filters |
| `attn_block_dtype` | bfloat16 | bfloat16 | (unchanged) |
| `mlp_dtype` | bfloat16 | bfloat16 | (unchanged) |
| `use_flash_attention_2` | True | True | (unchanged) |

## Grouped Query Attention (GQA) Disabled

StripedHyena-7B uses Grouped Query Attention with `proj_groups=4`:

```python
num_heads_kv = num_attention_heads // proj_groups  # 32 // 4 = 8 KV heads
```

**Why we set `proj_groups=1` (standard MHA):**

For small models in our scaling experiments, `num_attention_heads` can be less than 4 (e.g., 2 or 3 heads). With the default `proj_groups=4`, this would require `num_heads_kv < 1`, which is invalid.

Setting `proj_groups=1` means:
- Each query head has its own key-value head (standard Multi-Head Attention)
- No constraint on minimum number of heads
- Slightly more parameters in attention layers, but negligible for small models

For larger models where `num_heads >= 4`, GQA could be re-enabled by setting `proj_groups=4`.

## Post-Initialization Fix: Stable Pole Initialization

The original StripedHyena initializes Hyena filter poles with `torch.randn()`, which can produce poles with magnitude > 1. This causes numerical overflow in the filter computation:

```
h(t) = exp(log(poles) * t)
```

If |poles| > 1, then log(poles) has positive real part, causing exponential growth.

**Fix applied:**
```python
stable_magnitude = 0.5 + 0.49 * torch.sigmoid(magnitude)  # Maps to [0.62, 0.99]
```

This ensures all poles have magnitude < 1, guaranteeing stable (decaying) filters.

## MLP Expansion Factor (2.5x)

The `inner_mlp_size = 2.5 * hidden_size` choice:

| Model | MLP Expansion | Notes |
|-------|---------------|-------|
| GPT-2/3, Pythia | 4x | Standard FFN (2 matrices) |
| LLaMA | ~2.7x | SwiGLU (3 matrices, so effective params similar to 4x) |
| StripedHyena-7B | 3.5x | Gated MLP (14336 / 4096) |
| **SSM_configs table** | ~2.67x | Consistent ratio across all model sizes |
| **Our choice** | 2.5x | Conservative for smaller models |

### SSM_configs Reference

The `SSM_configs` DataFrame in `hf_utils_train_model_general.py` provides guidance on glu_size vs d_model scaling:

| d_model | glu_size | Ratio |
|---------|----------|-------|
| 128 | 336 | 2.625 |
| 320 | 848 | 2.65 |
| 512 | 1360 | 2.66 |
| 768 | 2048 | 2.67 |
| 1024 | 2736 | 2.67 |
| 1920 | 5120 | 2.67 |

The SSM_configs table consistently uses a ratio of **~2.67x** (approximately 8/3).

### Our Choice: 2.5x

We use 2.5x as a slightly conservative choice that:
1. Is close to the SSM_configs ratio (~2.67x)
2. Keeps parameter counts predictable with the simplified `C = 12 * L * D²` formula
3. Provides a small reduction in memory footprint for experiments

For production use or exact replication of SSM_configs scaling, change to 2.67x.
