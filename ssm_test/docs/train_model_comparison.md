# Comparison: `hf_utils_train_model.py` vs `hf_utils_train_model_general.py`

This document compares the original training script with the extended version that supports SSM/Hyena models and genomics data.

## 1. Supported Models

| Model | `train_model.py` | `_general.py` |
|-------|------------------|---------------|
| llama | Yes (disabled) | Yes (disabled) |
| pythia | Yes | Yes |
| ssm | No | Yes (HybridSSMModel) |
| hyena | No | Yes (StripedHyenaLite) |
| striped_hyena | No | Yes (Together AI from HuggingFace) |

## 2. Supported Datasets

| Dataset | `train_model.py` | `_general.py` |
|---------|------------------|---------------|
| wikitext-2-v1 | Yes | Yes |
| wikitext-103-v1 | Yes | Yes |
| lm1b | Yes | Yes |
| openwebtext2 | Yes | Yes |
| openwebtext2_stream | Yes | Yes |
| opengenome2 | No | Yes (HuggingFace streaming) |
| opengenome2_stream | No | Yes (HuggingFace streaming) |
| opengenome2_local | No | Yes (binary uint16 format) |

## 3. Config Prediction

| Aspect | `train_model.py` | `_general.py` |
|--------|------------------|---------------|
| Reference table | Pythia_configs only | Pythia_configs + SSM_configs |
| SSM_configs entries | N/A | 26 models (1M to 1.2B params) |
| Predicted params | hidden_size, n_layers, n_heads, intermediate_size | + d_model, glu_size, kv_size, learning_rate |
| Flash attention compat | No | Yes (ensures head_dim in {32, 64, 128, 256}) |

### SSM_configs Table

The `_general.py` file includes a curated table of 26 SSM model configurations:

| Model Size | d_model | glu_size | kv_size | n_heads | n_layer |
|------------|---------|----------|---------|---------|---------|
| 1M | 128 | 336 | 64 | 2 | 4 |
| 6M | 320 | 848 | 64 | 5 | 5 |
| 59M | 640 | 1696 | 64 | 10 | 12 |
| 303M | 1024 | 2736 | 64 | 16 | 24 |
| 680M | 1536 | 4096 | 128 | 12 | 24 |
| 1.2B | 1920 | 5120 | 128 | 15 | 25 |

Key ratios:
- `glu_size / d_model` ≈ 2.67 (consistent across all sizes)
- `kv_size` = 64 for models up to 473M, 128 for larger

## 4. Architecture Features

| Feature | `train_model.py` | `_general.py` |
|---------|------------------|---------------|
| hybrid_pattern | No | Yes |
| Stable pole init | No | Yes (for Hyena/StripedHyena) |
| vocab_size_override | No | Yes (for pre-tokenized data) |
| Detailed param accounting | Basic | Yes (per-layer breakdown) |

### Hybrid Patterns (`_general.py` only)

| Pattern | Description | Attention Layers |
|---------|-------------|------------------|
| `alternate` | Alternating SSM/Attention | Every other layer |
| `ssm_heavy` / `every_4` | SSM-heavy (Evo2-7B style) | Every 4th layer |
| `every_8` | Very SSM-heavy (Evo2-40B style) | Every 8th layer |
| `ssm_only` / `hyena_only` | Pure SSM | None |
| `attn_only` | Pure Attention | All layers |

### Stable Pole Initialization (`_general.py` only)

For Hyena/StripedHyena models, poles are initialized to ensure numerical stability:

```python
stable_magnitude = 0.5 + 0.49 * torch.sigmoid(magnitude)  # Maps to [0.62, 0.99]
```

This prevents `exp(log(poles) * t)` from overflowing when |poles| > 1.

## 5. Training Configuration

| Feature | `train_model.py` | `_general.py` |
|---------|------------------|---------------|
| Precision | fp16 only | fp16 or bf16 (for striped_hyena) |
| Dataloader workers | Fixed 12 | Adaptive (4/8/12 based on seq_length) |
| Sequence length | Fixed 128 | Adaptive (128 for text, 2048 for genomics) |
| Eval batch size | Same as train | Reduced for long sequences |

### Adaptive Dataloader Workers (`_general.py`)

| Sequence Length | Workers | Prefetch Factor |
|-----------------|---------|-----------------|
| >= 8192 | 4 | 2 |
| >= 2048 | 8 | 4 |
| < 2048 | 12 | 4 |

## 6. Data Handling

| Feature | `train_model.py` | `_general.py` |
|---------|------------------|---------------|
| Tokenizer | Always required | Optional (vocab_size_override) |
| Binary genome data | No | Yes (BinaryGenomeDataset class) |
| DNA tokenizer | No | Yes (dna_single_nucleotide) |
| Data collator | Standard HuggingFace | Standard + simple_collator for genomes |

### Genomics Data Format (`_general.py` only)

- **Path**: `/mfs1/datasets/pile/opengenome2_16gb/opengenome2_2048_uint16.bin`
- **Format**: Binary uint16, pre-tokenized
- **Vocabulary**: A=0, C=1, G=2, T=3, N=4 (vocab_size=5)
- **Train/Test Split**: Random i.i.d. with seed=42

## 7. Code Structure

| Aspect | `train_model.py` | `_general.py` |
|--------|------------------|---------------|
| Lines | ~995 | ~1750 |
| Config tables | 3 (Pythia, Llama1, Llama2) | 4 (+ SSM_configs) |
| Predictor functions | 1 | 2 (transformer + SSM) |

## Summary

`hf_utils_train_model_general.py` is a **superset** of `hf_utils_train_model.py` with additions for:

1. **SSM/Hyena models** - ssm, hyena, striped_hyena model families
2. **Genomics data** - opengenome2_local (binary), opengenome2 (HuggingFace)
3. **Adaptive training** - bf16 precision, dynamic dataloader workers, flexible seq_length
4. **Flash attention compatibility** - ensures head_dim constraints are met
5. **Stable pole initialization** - numerical stability for Hyena filter computations
6. **Hybrid patterns** - configurable SSM/Attention layer arrangements

### When to Use Each

| Use Case | Recommended Script |
|----------|-------------------|
| Pythia on text (OWT, WikiText) | `train_model.py` (simpler) |
| SSM/Hyena models | `_general.py` |
| Genomics data | `_general.py` |
| StripedHyena (Together AI) | `_general.py` |
| Scaling experiments with SSM | `_general.py` |
