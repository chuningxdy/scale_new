# Comparison: hf_utils_train_model_no_genome.py vs hf_utils_train_model.py

This document lists all differences between the two files.

---

## Summary

| Feature | `hf_utils_train_model_no_genome.py` | `hf_utils_train_model.py` |
|---------|-------------------------------------|---------------------------|
| Model families supported | `llama`, `pythia` | `llama`, `pythia`, `ssm`, `hyena`, `striped_hyena` |
| Datasets supported | `lm1b`, `wikitext-*`, `openwebtext2` | All above + `opengenome2`, `opengenome2_stream`, `opengenome2_local` |
| Sequence length | Fixed at 128 | 128 (text) or 2048 (genomics) |
| Lines of code | ~995 | ~1870 |

---

## 1. New SSM Configuration Table (Full Version Only)

**Location:** Lines 46-89 in `hf_utils_train_model.py`

The full version adds `SSM_configs` DataFrame containing StripedHyena/SSM model architecture configurations:

```python
SSM_configs = pd.DataFrame({
    'model': ['1', '6', '17', '29', '40', '59', '69', '84', '99', '114', '121', '135',
              '158', '175', '203', '232', '266', '303', '383', '473', '572', '680',
              '798', '926', '1063', '1209'],
    'd_model': [128, 320, 448, 512, 576, ...],
    'glu_size': [336, 848, 1200, 1360, 1536, ...],
    'kv_size': [64, 64, 64, ...],
    'n_heads': [2, 5, 7, 8, 8, ...],
    'n_layer': [4, 5, 7, 9, 10, ...],
    'learning_rate': [9.77e-4, 9.57e-4, ...],
    'total_parameters_count': [1e6, 6e6, 17e6, ...]
})
```

---

## 2. New Function: `func_ssm_config_from_params_count()` (Full Version Only)

**Location:** Lines 209-361 in `hf_utils_train_model.py`

New function to predict SSM/Hyena model configurations based on parameter count. Predicts:
- `d_model` (hidden dimension)
- `glu_size` (MLP inner dimension)
- `kv_size` (per-head dimension)
- `n_heads` (attention heads)
- `n_layer` (number of layers)

Includes flash-attention head dimension compatibility logic (supports head_dim of 32, 64, 128, 256).

---

## 3. Extended `build_model_with_predicted_config()` Function

### 3.1 New Parameters (Full Version)

```python
# No-genome version:
def build_model_with_predicted_config(target_param_count, tokenizer_path,
                                       model_family="llama", reference_configs=Pythia_configs,
                                       save_dir=None):

# Full version:
def build_model_with_predicted_config(target_param_count, tokenizer_path,
                                       model_family="llama", reference_configs=Pythia_configs,
                                       save_dir=None, hybrid_pattern="alternate",
                                       vocab_size_override=None):
```

### 3.2 New Model Family Support (Full Version)

**Lines 397-417:** Adds support for 3 new model families:

| Model Family | Config Class | Model Class |
|--------------|--------------|-------------|
| `ssm` | `HybridSSMConfig` | `HybridSSMModel` |
| `hyena` | `StripedHyenaLiteConfig` | `StripedHyenaLiteForCausalLM` |
| `striped_hyena` | Together AI AutoConfig | Together AI AutoModelForCausalLM |

### 3.3 Vocab Size Override (Full Version)

**Lines 427-444:** Allows overriding tokenizer vocab size (used for binary genome data with vocab=5):

```python
if vocab_size_override is not None:
    vocab_size = vocab_size_override
    print(f"\nUsing vocab_size override: {vocab_size}")
    tokenizer = None
```

### 3.4 Extended `simplified_param_count()` Function (Full Version)

**Lines 448-676:** Adds detailed parameter counting for:
- **Hyena models:** HyenaBlock (HyenaLikeMix + MLP), AttentionBlock, layer pattern calculations
- **StripedHyena models:** ParallelGatedConvBlock, MHA with GQA
- **SSM models:** SSMBlock (Mamba-style), attention layers

Includes `get_layer_counts()` helper function for hybrid patterns:
- `alternate`: Even layers = SSM/Hyena, odd layers = attention
- `ssm_heavy`/`evo2_7b`/`every_4`: Every 4th layer is attention
- `evo2_40b`/`every_8`: Every 8th layer is attention
- `ssm_only`/`hyena_only`: No attention layers
- `attn_only`: All attention layers

### 3.5 Model Config Creation (Full Version)

**Lines 711-856:** Adds config creation for:
- SSM models with `d_state`, `d_conv`, `hybrid_pattern`
- Hyena models with `hyena_kernel_size`, `hybrid_pattern`
- Together AI StripedHyena with custom layer indices, flash attention settings

### 3.6 StripedHyena Stable Pole Initialization (Full Version)

**Lines 891-906:** Applies stable pole initialization for Hyena filters to ensure `|poles| < 1`.

### 3.7 Safe Serialization (Full Version)

**Line 971:** Uses `safe_serialization=False` for weight-tied models:
```python
model.save_pretrained(save_dir, safe_serialization=False)
```

---

## 4. Extended `prepare_datasets()` Function

### 4.1 New Datasets Supported (Full Version)

**Lines 1064-1069:**
```python
elif dataset_name in ["opengenome2", "opengenome2_stream"]:
    data_label = "arcinstitute/opengenome2"  # ~5.5TB dataset
elif dataset_name == "opengenome2_local":
    data_label = "/mfs1/datasets/pile/opengenome2_16gb"  # Local ~16GB subset
```

### 4.2 New `BinaryGenomeDataset` Class (Full Version)

**Lines 1115-1152:** PyTorch Dataset for binary genome data:

```python
class BinaryGenomeDataset(TorchDataset):
    """PyTorch Dataset for binary genome data (uint16 format)
    Vocab: A=0, C=1, G=2, T=3, N=4 (5 tokens total)
    """
    def __init__(self, data_path, seq_length=2048):
        bin_file = os.path.join(data_path, "opengenome2_2048_uint16.bin")
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        ...
```

### 4.3 Genomics Streaming Support (Full Version)

**Lines 1163-1198:** Adds support for streaming `opengenome2_stream` from HuggingFace with subset selection (`pretrain_random_bacteria`).

### 4.4 Text Field Handling (Full Version)

**Lines 1215-1218:** Different text field for genomics vs text data:
```python
if dataset_name in ["opengenome2", "opengenome2_stream", "opengenome2_local"]:
    text_field = "sequence"
else:
    text_field = "text"
```

### 4.5 Genomics Cache Path (Full Version)

**Lines 1233-1238:**
```python
elif dataset_name == "opengenome2":
    token_cache_path = f"{cache_dir}/opengenome2_tokenized_128"
```

---

## 5. New `CustomTrainer._save()` Method (Full Version Only)

**Lines 1343-1356:** Override to handle weight-tied models:

```python
def _save(self, output_dir=None, state_dict=None):
    """Override _save to handle weight-tied models (embedding <-> lm_head)"""
    ...
    self.model.save_pretrained(output_dir, safe_serialization=False)
```

---

## 6. Extended `main()` Function

### 6.1 Genomics Tokenizer Paths (Full Version)

**Lines 1498-1504:**
```python
elif nn_dict["data"] in ["opengenome2", "opengenome2_stream"]:
    tokenizer_path = "datasets/tokenizers/dna_single_nucleotide"
elif nn_dict["data"] == "opengenome2_local":
    tokenizer_path = None  # No tokenizer needed - data is pre-tokenized
```

### 6.2 Binary Data Collator (Full Version)

**Lines 1508-1519:**
```python
if nn_dict["data"] == "opengenome2_local":
    tokenizer = None
    vocab_size_override = 5  # A, C, G, T, N

    def simple_collator(batch):
        """Simple collator for pre-tokenized binary data"""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    data_collator = simple_collator
```

### 6.3 Hybrid Pattern Configuration (Full Version)

**Line 1531:**
```python
hybrid_pattern = nn_dict.get("hybrid_pattern", "alternate")
```

### 6.4 Dynamic Sequence Length (Full Version)

**Lines 1554-1560:**
```python
if nn_dict["data"] in ["opengenome2", "opengenome2_stream", "opengenome2_local"]:
    seq_length = 2048  # Longer context for genomics data
else:
    seq_length = 128   # Default for text datasets
```

### 6.5 Sequence Length Factor in Gradient Accumulation (Full Version)

**Lines 1587-1599:**
```python
seq_length_factor = seq_length / 128.0
print(f"Sequence length factor for grad accumulation: {seq_length_factor:.1f}x")

if world_size == 1:
    num_grad_accu1 = h_dict['N']/1_000_000 * h_dict['B']/(128*256)/4 * seq_length_factor
    num_grad_accu2 = h_dict['B']/3072 * seq_length_factor
    ...
```

### 6.6 Dynamic Dataloader Settings (Full Version)

**Lines 1629-1642:**
```python
if seq_length >= 8192:
    dataloader_num_workers = 4
    dataloader_prefetch_factor = 2
elif seq_length >= 2048:
    dataloader_num_workers = 8
    dataloader_prefetch_factor = 4
else:
    dataloader_num_workers = 12
    dataloader_prefetch_factor = 4

eval_batch_size = max(1, BS // 2) if seq_length >= 2048 else BS
```

### 6.7 bf16 vs fp16 Selection (Full Version)

**Lines 1646, 1662-1663:**
```python
use_bf16 = nn_dict["model"] == "striped_hyena"
...
"bf16": use_bf16,
"fp16": not use_bf16,
```

---

## 7. Minor Differences

### 7.1 Pythia_configs DataFrame

**No-genome version (lines 19-27):** Includes `non_embed_parameters_count` column

**Full version (lines 1010-1017):** Does not include this column in the second definition

### 7.2 Run Command Comment

**No-genome version (lines 993-995):**
```python
# command to run:
# Single GPU: python hf_utils_train_model.py
# Multi-GPU:  torchrun --nproc_per_node=2 hf_utils_train_model.py
```

**Full version (lines 1867-1869):**
```python
# command to run:
# Single GPU: python hf_utils_train_model_general.py
# Multi-GPU:  torchrun --nproc_per_node=2 hf_utils_train_model_general.py
```

---

## 8. File Size Comparison

| Metric | No-genome | Full |
|--------|-----------|------|
| Lines | ~995 | ~1870 |
| Model families | 2 | 5 |
| Datasets | 4 | 7 |
| New functions | 0 | 1 (`func_ssm_config_from_params_count`) |
| New classes | 0 | 1 (`BinaryGenomeDataset`) |

---

## Conclusion

The `hf_utils_train_model.py` (full version) extends the no-genome version with:

1. **SSM/Hyena model support** - Three new model architectures with their own configuration predictors
2. **Genomics dataset support** - Including binary pre-tokenized data and HuggingFace streaming
3. **Longer sequence lengths** - 2048 tokens for genomics vs 128 for text
4. **Memory optimizations** - Dynamic dataloader settings and gradient accumulation based on sequence length
5. **Weight-tied model handling** - Custom save method with `safe_serialization=False`
