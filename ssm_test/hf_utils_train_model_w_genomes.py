import os
os.environ["HF_HOME"] = "/mfs1/datasets/pile/huggingface"
os.environ["HF_HUB_CACHE"]     = "/mfs1/datasets/pile/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/mfs1/datasets/pile/huggingface/datasets"

# CHANGE: Add distributed training environment setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])
print("HF_DATASETS_CACHE:", os.environ["HF_DATASETS_CACHE"])

import pandas as pd
import json
import sys
import yaml
from pathlib import Path

Pythia_configs = pd.DataFrame({
    'model': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b'],
    'hidden_size': [512, 768, 1024, 2048, 2048, 2560, 4096, 5120],
    'num_hidden_layers': [6, 12, 24, 16, 24, 32, 32, 36],
    'num_attention_heads': [8, 12, 16, 8, 16, 32, 32, 40],
    'intermediate_size': [2048, 3072, 4096, 8192, 8192, 10240, 16384, 20480],
    'non_embed_parameters_count': [18_915_328, 85_056_000, 302_311_424, 805_736_448, 1_208_602_624, 2_517_652_480, 6_444_163_072, 11_327_027_200],
    'total_parameters_count': [7e7, 85_056_000, 302_311_424, 805_736_448, 1_208_602_624, 2_517_652_480, 6_444_163_072, 11_327_027_200]
})

Llama1_configs = pd.DataFrame({
    'model': ['llama-7b', 'llama-13b', 'llama-33b', 'llama-65b'],
    'hidden_size': [4096, 5120, 6656, 8192],
    'num_hidden_layers': [32, 40, 60, 80],
    'num_attention_heads': [32, 40, 52, 64],
    'intermediate_size': [11008, 13824, 17920, 22016],
    'total_parameters_count': [7e9, 13e9, 33e9, 65e9]})

Llama2_configs = pd.DataFrame({
    'model': ['llama2-7b', 'llama2-13b', 'llama2-34b', 'llama2-70b'],
    'hidden_size': [4096, 5120, 6656, 8192],
    'num_hidden_layers': [32, 40, 48, 80],
    'num_attention_heads': [32, 40, 64, 64],
    'intermediate_size': [11008, 13824, 17920, 28672],
    'total_parameters_count': [7e9, 13e9, 34e9, 70e9]})


import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM
import torch
import os
import random

################################################################################
# NEW: Import AutoConfig and AutoModelForCausalLM for StripedHyena
################################################################################
from transformers import AutoConfig, AutoModelForCausalLM
################################################################################
# END NEW
################################################################################

def func_config_from_params_count(df, plot_filename=None):
    """
    Creates a function that predicts model config values based on parameter count,
    and optionally creates a visualization of the regression.

    Args:
        df: DataFrame with model configs including total_parameters_count
        plot_filename: If provided, save a visualization to this filename

    Returns:
        A function that takes parameter count and returns predicted config values
    """
    X = np.log(df['total_parameters_count'])
    X_billions = df['total_parameters_count'] / 1e9

    target_columns = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size']

    models = {}

    if plot_filename:
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 2)
        plt.suptitle(f'Log-Log Regression: Model Architecture vs. Parameter Count\nModel family: {df["model"].iloc[0].split("-")[0]}',
                    fontsize=16)

    for i, col in enumerate(target_columns):
        y = np.log(df[col])
        y_actual = df[col]

        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        models[col] = (intercept, slope)

        print(f"Log-log regression for {col}:")
        print(f"Intercept: {intercept:.4f}, Coefficient: {slope:.4f}, R²: {r_value**2:.4f}")

        if plot_filename:
            ax = plt.subplot(gs[i // 2, i % 2])

            ax.scatter(X_billions, y_actual, color='blue', s=100, alpha=0.7, label='Actual values')

            x_range = np.logspace(np.log10(min(X_billions)*0.7), np.log10(max(X_billions)*1.3), 100)
            y_pred = np.exp(intercept + slope * np.log(x_range * 1e9))

            ax.plot(x_range, y_pred, 'r-', linewidth=2, label=f'Regression (R²={r_value**2:.3f})')

            eqn_text = f"ln(y) = {intercept:.2f} + {slope:.2f} × ln(x)"
            ax.text(0.05, 0.95, eqn_text, transform=ax.transAxes,
                   fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.set_xlabel('Parameter Count (billions)', fontsize=12)
            ax.set_ylabel(col.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{col.replace("_", " ").title()} vs Model Size', fontsize=14)

            for j, model in enumerate(df['model']):
                ax.annotate(model, (X_billions[j], y_actual[j]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)

            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='best')

    if plot_filename:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")
        plt.close()

    def predict_config(params_count):
        """
        Predicts model configuration based on parameter count.

        Args:
            params_count: Total parameter count (e.g., 7e9 for 7B)

        Returns:
            Dictionary with predicted configuration values
        """
        log_params = np.log(params_count)
        result = {'non_embed_parameters_count': params_count}

        for col in ['hidden_size']:
            intercept, slope = models[col]
            predicted_value = np.exp(intercept + slope * log_params)
            result[col] = int(round(predicted_value/16) * 16)

        result["num_attention_heads"] = int(round(result['hidden_size'] / 16))

        for col in ['num_hidden_layers', 'intermediate_size']:
            intercept, slope = models[col]
            predicted_value = np.exp(intercept + slope * log_params)

            if col == 'num_hidden_layers':
                result[col] = int(round(predicted_value))
                if result[col] < 2:
                    result[col] = 2

            elif col == 'intermediate_size':
                result[col] = int(round(result['hidden_size'] * 4))

        return result

    return predict_config


################################################################################
# NEW: StripedHyena-specific configuration predictor
################################################################################
def func_config_from_params_count_striped_hyena(vocab_size=5):
    """
    Creates a function that predicts StripedHyena config values based on parameter count.
    Uses simplified 7-parameter model for SSM architectures.

    Args:
        vocab_size: Vocabulary size (default 5 for genomics A/C/G/T/N)

    Returns:
        A function that takes parameter count and returns predicted config values
    """

    def predict_config(params_count):
        """
        Predicts StripedHyena configuration based on parameter count.
        Uses simplified parameter counting: C = 12 * n_layer * d_model^2 + 2 * vocab_size * d_model
        """
        # Simplified model: C = 12 * L * D^2 + 2 * V * D
        # Given target C and vocab V, solve for D and L

        # Use ratio from reference models: L/D ratio stays roughly constant
        # For StripedHyena-7B: d_model=4096, n_layers=32 -> ratio ~= 0.0078
        # We'll use a slightly higher ratio for smaller models

        target_C = params_count
        V = vocab_size

        # Iteratively find d_model and n_layer
        best_d_model = 256
        best_n_layer = 4
        best_diff = float('inf')

        for d_model in range(128, 8192, 32):
            for n_layer in range(2, 64, 2):
                # Simplified param count for StripedHyena
                # 12 * L * D^2 captures attention + MLP + Hyena filter params
                estimated_C = 12 * n_layer * (d_model ** 2) + 2 * V * d_model
                diff = abs(estimated_C - target_C)

                if diff < best_diff:
                    best_diff = diff
                    best_d_model = d_model
                    best_n_layer = n_layer

        # Ensure d_model is divisible by flash attention head dims
        flash_attn_head_dims = [32, 64, 128, 256]
        valid_n_heads_options = []
        for head_dim in flash_attn_head_dims:
            if best_d_model % head_dim == 0:
                candidate_n_heads = best_d_model // head_dim
                if candidate_n_heads >= 1:
                    valid_n_heads_options.append((candidate_n_heads, head_dim))

        if valid_n_heads_options:
            # Prefer smaller head_dim (more heads) for expressivity
            best_n_heads, best_head_dim = max(valid_n_heads_options, key=lambda x: x[0])
        else:
            # Adjust d_model to be divisible by 64 (common head_dim)
            best_d_model = (best_d_model // 64) * 64
            if best_d_model < 128:
                best_d_model = 128
            best_n_heads = best_d_model // 64

        # Ensure even number of layers for alternating Hyena/Attention
        if best_n_layer % 2 != 0:
            best_n_layer += 1

        # GLU size is typically 2.5x hidden size for StripedHyena
        glu_size = int(best_d_model * 2.5)

        result = {
            'd_model': best_d_model,
            'n_layer': best_n_layer,
            'n_heads': best_n_heads,
            'glu_size': glu_size,
            'non_embed_parameters_count': params_count
        }

        return result

    return predict_config
################################################################################
# END NEW
################################################################################


def build_model_with_predicted_config(
        target_param_count,
        tokenizer_path,
        model_family = "llama",
        reference_configs = Pythia_configs,
        save_dir=None,
        ################################################################################
        # NEW: Add vocab_size parameter for models without tokenizer
        ################################################################################
        vocab_size=None):
        ################################################################################
        # END NEW
        ################################################################################
    """
    Build a LLaMA model with predicted configuration for a given parameter count.

    Args:
        param_count: Target parameter count (e.g., 1e9 for 1B)
        tokenizer_path: Path to the tokenizer
        save_dir: Directory to save the model (optional)

    Returns:
        Model, configuration, and actual parameter count
    """

    if model_family == "llama":
        raise ValueError("Llama model family is not supported yet. Use 'pythia' instead.")
        print(f"\nBuilding a LLaMA model with target size: {target_param_count/1e9:.1f}B parameters")
        config_creation_function = LlamaConfig
        model_creation_function = LlamaForCausalLM
    elif model_family == "pythia":
        print(f"\nBuilding a Pythia model with target size: {target_param_count/1e9:.1f}B parameters")
        config_creation_function = GPTNeoXConfig
        model_creation_function = GPTNeoXForCausalLM
    ################################################################################
    # NEW: StripedHyena model support
    ################################################################################
    elif model_family == "striped_hyena":
        print(f"\nBuilding a StripedHyena model with target size: {target_param_count/1e6:.1f}M parameters")
        # Will use AutoConfig and AutoModelForCausalLM with trust_remote_code
    ################################################################################
    # END NEW
    ################################################################################
    else:
        raise ValueError("Invalid model class. Choose 'llama', 'pythia', or 'striped_hyena'.")

    ################################################################################
    # NEW: Handle vocab_size for models without tokenizer (e.g., genomics)
    ################################################################################
    if vocab_size is not None:
        print(f"Using provided vocab_size: {vocab_size}")
    elif tokenizer_path is not None and os.path.exists(tokenizer_path):
        print(f"\nLoading custom tokenizer from {tokenizer_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            vocab_size = len(tokenizer.get_vocab())
            print(f"Tokenizer loaded successfully with vocab size: {vocab_size}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Falling back to default vocab size of 3000")
            vocab_size = 3000
    else:
        print("No tokenizer path provided, using default vocab size of 3000")
        vocab_size = 3000
    ################################################################################
    # END NEW
    ################################################################################

    ################################################################################
    # NEW: StripedHyena model building
    ################################################################################
    if model_family == "striped_hyena":
        predictor = func_config_from_params_count_striped_hyena(vocab_size=vocab_size)
        config_dict = predictor(target_param_count)

        print("\nPredicted StripedHyena configuration:")
        for key, value in config_dict.items():
            print(f" - {key}: {value}")

        # Load base config from pretrained StripedHyena
        print("\nLoading StripedHyena base config...")
        base_config = AutoConfig.from_pretrained(
            "togethercomputer/StripedHyena-Hessian-7B",
            trust_remote_code=True
        )

        # Apply our predicted config
        base_config.vocab_size = vocab_size
        base_config.hidden_size = config_dict['d_model']
        base_config.num_filters = config_dict['d_model']  # Must equal hidden_size
        base_config.num_layers = config_dict['n_layer']
        base_config.num_attention_heads = config_dict['n_heads']
        base_config.inner_mlp_size = config_dict['glu_size']

        # Set up alternating Hyena/Attention layers
        n_layers = config_dict['n_layer']
        attn_layer_idxs = [i for i in range(n_layers) if i % 2 == 1]  # Odd layers: attention
        hyena_layer_idxs = [i for i in range(n_layers) if i % 2 == 0]  # Even layers: Hyena
        base_config.attn_layer_idxs = attn_layer_idxs
        base_config.hyena_layer_idxs = hyena_layer_idxs

        # Important settings for numerical stability
        base_config.make_vocab_size_divisible_by = 1  # Avoid vocab mismatch
        base_config.tie_word_embeddings = False  # StripedHyena has bug with tied embeddings
        base_config.proj_groups = 1  # Standard MHA (no GQA)

        # Dtype settings - use float32 for Hyena to avoid numerical instability
        base_config.hyena_block_dtype = "float32"
        base_config.attn_block_dtype = "bfloat16"
        base_config.mlp_dtype = "bfloat16"

        # Flash attention settings
        base_config.use_flash_attention_2 = True
        base_config.use_flash_attn = True

        print(f"\nStripedHyena layer configuration:")
        print(f"  Attention layers: {attn_layer_idxs}")
        print(f"  Hyena layers: {hyena_layer_idxs}")

        # Convert dtype strings to torch dtypes for model creation
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        base_config.hyena_block_dtype = dtype_map[base_config.hyena_block_dtype]
        base_config.attn_block_dtype = dtype_map[base_config.attn_block_dtype]
        base_config.mlp_dtype = dtype_map[base_config.mlp_dtype]

        print("\nInitializing StripedHyena model...")
        model = AutoModelForCausalLM.from_config(base_config, trust_remote_code=True)

        # Apply stable pole initialization for Hyena filters
        # The poles must have |poles| < 1 so that log(poles) has negative real part,
        # ensuring exp(log_poles * t) decays over time instead of exploding.
        print("  Applying stable pole initialization for Hyena filters...")
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'poles') and hasattr(module, 'residues'):
                    poles_complex = torch.view_as_complex(module.poles.data.float())
                    magnitude = poles_complex.abs()
                    phase = poles_complex / (magnitude + 1e-8)
                    # Clamp magnitude to [0.62, 0.99] for stable but expressive filters
                    stable_magnitude = 0.5 + 0.49 * torch.sigmoid(magnitude)
                    stable_poles = stable_magnitude * phase
                    module.poles.data = torch.view_as_real(stable_poles).to(module.poles.dtype)
                    print(f"    Fixed poles in {name}: magnitude range [{stable_magnitude.min():.4f}, {stable_magnitude.max():.4f}]")

        # Convert dtype back to strings for config serialization (both base_config and model.config)
        base_config.hyena_block_dtype = "float32"
        base_config.attn_block_dtype = "bfloat16"
        base_config.mlp_dtype = "bfloat16"
        model.config.hyena_block_dtype = "float32"
        model.config.attn_block_dtype = "bfloat16"
        model.config.mlp_dtype = "bfloat16"

        config = base_config

        param_count_actual = sum(p.numel() for p in model.parameters())
        # Untied embeddings: 2x vocab_size * hidden_size
        embedding_params = config.vocab_size * config.hidden_size * 2
        non_embedding_params_actual = param_count_actual - embedding_params

        accuracy = (param_count_actual / target_param_count) * 100
        print(f"\nStripedHyena model created:")
        print(f"  Target params: {target_param_count:,}")
        print(f"  Actual params: {param_count_actual:,}")
        print(f"  Prediction accuracy: {accuracy:.2f}%")
        print(f"  Embedding params: {embedding_params:,}")
        print(f"  Non-embedding params: {non_embedding_params_actual:,}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"\nModel saved to {save_dir}")

        return model, config, non_embedding_params_actual, param_count_actual
    ################################################################################
    # END NEW
    ################################################################################

    predictor = func_config_from_params_count(reference_configs)


    print(f"\nLoading custom tokenizer from {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = len(tokenizer.get_vocab())
        print(f"Tokenizer loaded successfully with vocab size: {vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to default vocab size of 3000")
        vocab_size = 3000



    def simplified_param_count(config_dict, vocab_size):
        hidden_size_config = config_dict['hidden_size']
        num_layers_config = config_dict['num_hidden_layers']

        simplified_non_embedding_params_cnt = 12 * num_layers_config * hidden_size_config ** 2
        embedding_params_cnt = vocab_size * hidden_size_config * 2
        simplified_total_params_cnt = simplified_non_embedding_params_cnt + embedding_params_cnt

        return simplified_non_embedding_params_cnt, simplified_total_params_cnt




    candidate_param_counts = [target_param_count * (x / 100) for x in range(10, 151, 5)]
    candidate_param_counts = [int(x) for x in candidate_param_counts]

    candidate_param_counts_simplified_outputs = {}
    for param_count in candidate_param_counts:
        config_dict = predictor(param_count)
        non_embed_params_simplified, total_params_simplified = simplified_param_count(config_dict, vocab_size)
        candidate_param_counts_simplified_outputs[param_count] = (non_embed_params_simplified, total_params_simplified)

    best_param_count = None

    ratios_to_target = []
    best_ratio_diff = float('inf')
    for param_count, (non_embed_params_simplified, total_params_simplified) in candidate_param_counts_simplified_outputs.items():
        ratio = total_params_simplified / target_param_count
        ratios_to_target.append((param_count, ratio))
        ratio_diff = abs(ratio - 1)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_param_count = param_count

    print(f"\nSelected candidate parameter count for prediction: {best_param_count:,} ({best_param_count/1e9:.4f}B) with simplified total params closest to target")
    print(f" - Ratios are (param_count, ratio to target): {ratios_to_target} ...")

    if True:
        config_dict = predictor(best_param_count)
        print("\nPredicted configuration:")
        for key, value in config_dict.items():
            print(f" - {key}: {value}")

        config = config_creation_function(
            vocab_size=vocab_size,
            hidden_size=config_dict['hidden_size'],
            intermediate_size=config_dict['intermediate_size'],
            num_hidden_layers=config_dict['num_hidden_layers'],
            num_attention_heads=config_dict['num_attention_heads']
        )

        print("\nInitializing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        try:
            if 'model' in locals():
                del model
            model = model_creation_function(config)
        except Exception as e:
            raise ValueError("Model initialization failed. Please check the configuration.")

        param_count_actual = sum(p.numel() for p in model.parameters())

        embedding_params = config.vocab_size * config.hidden_size * 2
        non_embedding_params_actual = param_count_actual - embedding_params
        accuracy = (param_count_actual / target_param_count) * 100
        print(f"trying ratio: {best_param_count/target_param_count:.2f}")
        print(f"Prediction accuracy: {accuracy:.2f}% of target")

    if accuracy < 75 or accuracy > 130:
        raise ValueError(f"Final prediction accuracy {accuracy:.2f}% is outside the acceptable range.")

    attention_params_per_layer = 4 * (config.hidden_size ** 2)
    ffn_params_per_layer = 2 * config.hidden_size * config.intermediate_size

    total_attention_params = attention_params_per_layer * config.num_hidden_layers
    total_ffn_params = ffn_params_per_layer * config.num_hidden_layers

    print(f" - Attention layers: {total_attention_params:,} params")
    print(f" - Feed-forward layers: {total_ffn_params:,} params")
    print(f" - Other (norm, etc.): {param_count_actual - embedding_params - total_attention_params - total_ffn_params:,} params")
    print(f" - Non-embedding params: {param_count_actual - embedding_params:,} params")
    print(f" - Total params: {param_count_actual:,} params ({param_count_actual})")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        if os.path.exists(tokenizer_path):
            tokenizer.save_pretrained(save_dir)
        print(f"\nModel saved to {save_dir}")

    return model, config, non_embedding_params_actual, param_count_actual





import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import concatenate_datasets, load_dataset, load_from_disk, Dataset
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import json
import sys

import torch
import os
import random

from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

################################################################################
# NEW: Import for random train/test split
################################################################################
from torch.utils.data import Subset
################################################################################
# END NEW
################################################################################


Pythia_configs = pd.DataFrame({
    'model': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b'],
    'hidden_size': [512, 768, 1024, 2048, 2048, 2560, 4096, 5120],
    'num_hidden_layers': [6, 12, 24, 16, 24, 32, 32, 36],
    'num_attention_heads': [8, 12, 16, 8, 16, 32, 32, 40],
    'intermediate_size': [2048, 3072, 4096, 8192, 8192, 10240, 16384, 20480],
    'total_parameters_count': [18_915_328, 85_056_000, 302_311_424, 805_736_448, 1_208_602_624, 2_517_652_480, 6_444_163_072, 11_327_027_200]
})

def get_full_dataset(dataset, verbose=True):
    available_splits = list(dataset.keys())

    if verbose:
        print(f"Combining all available splits: {available_splits}")

    full_dataset = dataset[available_splits[0]]

    for split in available_splits[1:]:
        split_dataset = dataset[split]
        full_dataset = concatenate_datasets([full_dataset, split_dataset])

    return full_dataset


################################################################################
# NEW: Binary genome dataset class for opengenome2_local
################################################################################
class BinaryGenomeDataset(torch.utils.data.Dataset):
    """
    Dataset for pre-tokenized binary genome data stored as uint16.
    File format: binary file where each 2 bytes is a token ID.
    Vocab: A=0, C=1, G=2, T=3, N=4
    """
    def __init__(self, data_path, seq_length=2048):
        self.data_path = data_path
        self.seq_length = seq_length

        # Memory-map the file for efficient random access
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)

        # Number of complete sequences we can extract
        self.num_sequences = self.total_tokens // seq_length

        print(f"Loaded binary genome dataset from {data_path}")
        print(f"  Total tokens: {self.total_tokens:,}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Number of sequences: {self.num_sequences:,}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length

        # Get sequence and convert to torch tensor
        sequence = torch.from_numpy(self.data[start:end].astype(np.int64))

        return {
            'input_ids': sequence,
            'labels': sequence.clone()
        }
################################################################################
# END NEW
################################################################################


def prepare_datasets(tokenizer, dataset_name, seq_length, cache_dir, test_size=1000,
                         verbose=True):
        """
        Prepare training and test datasets.

        Args:
            tokenizer: Tokenizer to use
            dataset_name: Name of the dataset
            seq_length: Sequence length
            cache_dir: Directory to cache datasets
            test_size: Number of sequences for test set
            split: Split to use
            verbose: Whether to print progress

        Returns:
            tuple: (train_dataset, test_dataset)
        """
        if verbose:
            print(f"Loading dataset...")


        if dataset_name == "wikitext-2-v1":
            data_label = "wikitext"
        elif dataset_name == "wikitext-103-v1":
            data_label = "wikitext"
        elif dataset_name == "lm1b":
            data_label = "lm1b"
        elif dataset_name in ["openwebtext2", "openwebtext2_stream"]:
            data_label = "vietgpt/the_pile_openwebtext2"
        ################################################################################
        # NEW: opengenome2_local dataset support
        ################################################################################
        elif dataset_name == "opengenome2_local":
            data_label = "opengenome2_local"
        ################################################################################
        # END NEW
        ################################################################################
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        ################################################################################
        # NEW: Handle opengenome2_local binary dataset
        ################################################################################
        if dataset_name == "opengenome2_local":
            # Path to the pre-tokenized binary genome data
            data_dir = "/mfs1/datasets/pile/opengenome2_16gb"
            data_path = os.path.join(data_dir, "opengenome2_2048_uint16.bin")

            if not os.path.exists(data_path):
                raise ValueError(f"Binary genome data not found at {data_path}")

            full_dataset = BinaryGenomeDataset(data_path, seq_length=seq_length)
            total_size = len(full_dataset)

            # Random i.i.d. train/test split
            rng = np.random.RandomState(seed=42)
            all_indices = np.arange(total_size)
            rng.shuffle(all_indices)

            test_indices = all_indices[:test_size].tolist()
            train_indices = all_indices[test_size:].tolist()

            train_dataset = Subset(full_dataset, train_indices)
            test_dataset = Subset(full_dataset, test_indices)

            if verbose:
                print(f"Prepared opengenome2_local datasets (random i.i.d. split):")
                print(f"  - Training set: {len(train_dataset)} sequences")
                print(f"  - Test set: {len(test_dataset)} sequences")

            return train_dataset, test_dataset
        ################################################################################
        # END NEW
        ################################################################################

        if dataset_name == "lm1b":
            dataset = load_dataset(data_label, name=None, cache_dir=cache_dir, trust_remote_code=True)
            split_dataset = get_full_dataset(dataset, verbose=verbose)
            filtered_dataset = split_dataset
            stream = False

        elif dataset_name == "wikitext-2-v1" or dataset_name == "wikitext-103-v1":
            dataset = load_dataset(data_label, name=dataset_name, cache_dir=cache_dir)
            split_dataset = get_full_dataset(dataset, verbose=verbose)
            def is_non_empty(example):
                return bool(example["text"].strip())

            filtered_dataset = split_dataset.filter(
                is_non_empty,
                desc="Filtering out empty strings"
            )
            stream = False


        elif dataset_name == "openwebtext2":
            dataset = load_dataset(data_label, name=None,
                                   cache_dir='/mfs1/datasets/pile/openwebtext2')

            split_dataset = get_full_dataset(dataset, verbose=verbose)
            filtered_dataset = split_dataset
            stream = False

        elif dataset_name == "openwebtext2_stream":
            dataset = load_dataset(data_label, name=None,
                                   cache_dir='/mfs1/datasets/pile/openwebtext2',
                                   streaming=True)

            split_dataset = dataset
            split_dataset = dataset.shuffle(buffer_size=1000, seed=42)
            filtered_dataset = split_dataset
            stream = True



        if verbose:
            removed_count = len(split_dataset) - len(filtered_dataset)
            print(f"Filtered out {removed_count} empty examples ({removed_count/len(split_dataset)*100:.2f}%)")
            print(f"Original dataset size: {len(split_dataset)}, Filtered size: {len(filtered_dataset)}")

        split_dataset = filtered_dataset

        if dataset_name in ["wikitext-2-v1", "wikitext-103-v1", "openwebtext2", "openwebtext2_stream"]:
            def tokenize_with_stride(examples):

                tokenized_inputs = tokenizer(
                    examples["text"],
                    return_overflowing_tokens=True,
                    stride=1,
                    max_length=seq_length,
                    truncation=True,
                    return_special_tokens_mask=True
                )

                return tokenized_inputs

            if not stream:
                if dataset_name != "openwebtext2":
                    raise ValueError("Dataset name must be openwebtext2 for this code to work")
                token_cache_path = "/mfs1/datasets/pile/openwebtext2/owt2_128"
                if os.path.exists(token_cache_path):
                    print("Loading cached tokenized dataset from", token_cache_path)
                    tokenized_dataset = load_from_disk(token_cache_path)
                    tokenized_dataset.set_format("torch", columns=["input_ids"])
                else:
                    print("No cache found. Tokenizing dataset with stride to create multiple examples")
                    tokenized_dataset = split_dataset.map(
                        tokenize_with_stride,
                        batched=True,
                        num_proc=12,
                        remove_columns=split_dataset.column_names,
                        desc="Tokenizing with stride to create multiple examples"
                    )
                    print("Saving tokenized dataset to", token_cache_path)
                    tokenized_dataset.save_to_disk(token_cache_path)
                    print("Tokenized dataset saved successfully.")

            else:
                tokenized_dataset = split_dataset.map(
                    tokenize_with_stride,
                    batched=True,
                    remove_columns=["text"])

            if not stream:
                print("Dataset features:", tokenized_dataset.features)

            print("Number of examples:", len(tokenized_dataset))

        elif dataset_name in ["lm1b"]:

            def tokenize_with_stride(examples):

                tokenized_inputs = tokenizer(
                    examples["text"],
                    return_overflowing_tokens=True,
                    stride=1,
                    max_length=seq_length,
                    truncation=True,
                    return_special_tokens_mask=True
                )

                return tokenized_inputs

            grouped_dataset = split_dataset
            tokenized_dataset = grouped_dataset.map(
                tokenize_with_stride,
                batched=True,
                num_proc=12,
                remove_columns=["text"],
                desc="Tokenizing with stride to create multiple examples"
            )


        if not stream:
            test_size_frac = test_size / len(tokenized_dataset)
            split_dataset = tokenized_dataset.train_test_split(test_size=test_size_frac, seed=42)

            train_dataset = split_dataset["train"]
            test_dataset = split_dataset["test"]
            if len(tokenized_dataset) > test_size:
                    train_dataset = tokenized_dataset.select(range(test_size, len(tokenized_dataset)))
            else:
                    train_dataset = tokenized_dataset

            if verbose:
                print(f"Prepared datasets:")
                print(f"- Training set: {len(train_dataset)} sequences")
                print(f"- Test set: {len(test_dataset)} sequences")
        else:
            print("Keys of tokenized_dataset:", tokenized_dataset.keys())
            test_dataset = tokenized_dataset["train"].take(test_size)
            train_dataset = tokenized_dataset["train"].skip(test_size)
            if verbose:
                print(f"Prepared datasets: stream train, test")

        return train_dataset, test_dataset


################################################################################
# NEW: Data collator for pre-tokenized genome data (no tokenizer needed)
################################################################################
class GenomeDataCollator:
    """
    Data collator for pre-tokenized genome sequences.
    Simply stacks input_ids and labels into batches.
    """
    def __call__(self, features):
        input_ids = torch.stack([f['input_ids'] for f in features])
        labels = torch.stack([f['labels'] for f in features])

        return {
            'input_ids': input_ids,
            'labels': labels
        }
################################################################################
# END NEW
################################################################################


class CustomTrainer(Trainer):

        def __init__(self,
                    momentum: float = 0.9,
                    initial_learning_rate: float = 5e-5,
                    lr_scheduler_type: str = "linear",
                    step_decay_schedule_dict: dict = None,
                    optimizer_type: str = "sgd",
                    **kwargs):
            self.momentum = momentum
            self.initial_learning_rate = initial_learning_rate
            self.lr_scheduler_type = lr_scheduler_type
            self.step_decay_schedule_dict = step_decay_schedule_dict or {}
            self.optimizer_type = optimizer_type

            self.eval_losses = []
            self.eval_steps = []

            super().__init__(**kwargs)



        def create_optimizer_and_scheduler(self,
                                           num_training_steps: int):
                """
                Setup the optimizer and learning rate scheduler.
                """
                if self.optimizer_type == "sgd":
                    self.optimizer = SGD(
                        self.model.parameters(),
                        lr=self.initial_learning_rate,
                        momentum=self.momentum,
                        weight_decay=0
                    )
                elif self.optimizer_type == "adamw":
                    self.optimizer = AdamW(
                        self.model.parameters(),
                        lr=self.initial_learning_rate,
                        weight_decay=0
                    )

                if self.lr_scheduler_type == "constant":
                    self.lr_scheduler = get_constant_schedule(self.optimizer)
                elif self.lr_scheduler_type == "cosine":
                    self.lr_scheduler = get_cosine_schedule_with_warmup(
                        self.optimizer,
                        num_warmup_steps = int(num_training_steps * 0.01),
                        num_training_steps = num_training_steps)
                elif self.lr_scheduler_type == "step":
                    if self.step_decay_schedule_dict is None:
                        raise ValueError("step_decay_schedule_dict must be provided for step decay schedule")

                    milestones = [int(decay_at * num_training_steps) for decay_at in self.step_decay_schedule_dict['decay_at']]

                    lr_values = [1.0]
                    for decay_amt in self.step_decay_schedule_dict['decay_amt']:
                        lr_values.append(lr_values[-1] * decay_amt)

                    def lr_lambda(current_step: int):
                        for i, milestone in enumerate(milestones):
                            if current_step < milestone:
                                return lr_values[i]
                        return lr_values[-1]

                    self.lr_scheduler = LambdaLR(
                        self.optimizer,
                        lr_lambda,
                        last_epoch=-1
                    )

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            output = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )

            self.eval_losses.append(output["eval_loss"])
            self.eval_steps.append(self.state.global_step)

            self.save_eval_metrics()

            return output

        def save_eval_metrics(self):
            """Save evaluation metrics to a CSV file"""

            metrics_df = pd.DataFrame({
                'ckpt': self.eval_steps,
                'loss': self.eval_losses
            })

            metrics_df = metrics_df.drop_duplicates(subset=['ckpt'])
            metrics_df.to_csv(f"{self.args.output_dir}/loss_curve_df.csv", index=False)

            self.plot_eval_metrics()

        def plot_eval_metrics(self):
            """Create a plot of evaluation loss over steps"""


            plt.figure(figsize=(10, 6))
            plt.plot(self.eval_steps, self.eval_losses, marker='o')
            plt.title('Evaluation Loss Over Training')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(f"{self.args.output_dir}/eval_loss_curve.png")
            plt.close()

        def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
            """
            Perform an initial evaluation at step 0, then proceed with training.
            """
            print("Performing initial evaluation at step 0:")
            initial_metrics = self.evaluate()
            print(f"Initial eval loss: {initial_metrics['eval_loss']:.4f}")

            return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, **kwargs)

def main():

    if len(sys.argv) > 1:
            temp_path = sys.argv[1]
    else:
            temp_path = 'outputs/nn_hf/temp/'

    print(f"Using path: {temp_path}")

    datasets_cache_dir = "./datasets"

    hf_temp_folder = temp_path
    input_path = hf_temp_folder + 'input/'
    if not os.path.exists(input_path):
        raise ValueError(f"Input folder {input_path} does not exist.")

    with open(input_path + 'nn_dict.json', 'r') as f:
        nn_dict = json.load(f)
    with open(input_path + 'h_dict.json', 'r') as f:
        h_dict = json.load(f)

    with open(input_path + 'path.txt', 'r') as f:
        path = f.read()

    os.makedirs(datasets_cache_dir, exist_ok=True)
    os.makedirs(path, exist_ok=True)
    out_path = hf_temp_folder + 'output/'
    os.makedirs(out_path, exist_ok=True)


    target_size = h_dict["N"]

    ################################################################################
    # NEW: Handle tokenizer and vocab_size for different datasets
    ################################################################################
    if nn_dict["data"] == "opengenome2_local":
        # Genomics data: vocab is A=0, C=1, G=2, T=3, N=4
        tokenizer = None
        tokenizer_path = None
        vocab_size = 5
        print(f"Using opengenome2_local dataset with vocab_size={vocab_size}")
    elif nn_dict["data"] == "lm1b":
        tokenizer_path = "datasets/tokenizers/bpe_tokenizer_lm1b_vocab3000"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = None  # Will be determined from tokenizer
    elif nn_dict["data"] == "wikitext-2-v1":
        tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-2-v1_vocab3000"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = None
    elif nn_dict["data"] == "wikitext-103-v1":
        tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-103-v1_vocab3000"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = None
    elif nn_dict["data"] in ["openwebtext2", "openwebtext2_stream"]:
        tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-103-v1_vocab3000"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size = None
    else:
        raise ValueError(f"Unsupported dataset: {nn_dict['data']}")
    ################################################################################
    # END NEW
    ################################################################################

    ################################################################################
    # NEW: Choose data collator based on dataset type
    ################################################################################
    if nn_dict["data"] == "opengenome2_local":
        data_collator = GenomeDataCollator()
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )
    ################################################################################
    # END NEW
    ################################################################################

    ################################################################################
    # NEW: Pass vocab_size to build_model_with_predicted_config
    ################################################################################
    model, config, actual_nonembed_params, actual_params = build_model_with_predicted_config(
            target_param_count=target_size,
            tokenizer_path=tokenizer_path,
            save_dir=path,
            model_family=nn_dict["model"],
            reference_configs=Pythia_configs,
            vocab_size=vocab_size
        )
    ################################################################################
    # END NEW
    ################################################################################

    df = pd.DataFrame({'actual_N': [actual_params]})
    df.to_csv(os.path.join(out_path, 'actual_N_df.csv'), index=False)


    print(f"Model with {target_size/1e6:.1f}M parameters built successfully.")
    print(f"Model saved to {path}")
    print("requested parameter count: ", target_size)
    print("actual non-embedding parameter count: ", actual_nonembed_params)
    print("actual parameter count: ", actual_params)

    ################################################################################
    # NEW: Set sequence length based on dataset
    ################################################################################
    if nn_dict["data"] == "opengenome2_local":
        seq_length = 2048  # Default for genomics
    else:
        seq_length = 128  # Default for text
    ################################################################################
    # END NEW
    ################################################################################

    train_dataset, test_dataset = prepare_datasets(
            tokenizer=tokenizer,
            dataset_name=nn_dict["data"],
            seq_length=seq_length,
            cache_dir=datasets_cache_dir,
            test_size=1000,
            verbose=True
        )

    # CHANGE: Get world size for multi-GPU training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    print(f"\n{'='*60}")
    print(f"Multi-GPU Configuration:")
    print(f"  World size (total GPUs): {world_size}")
    print(f"  Local rank: {local_rank}")
    print(f"{'='*60}\n")

    # CHANGE: Updated gradient accumulation logic accounting for multi-GPU
    target_batch_size = h_dict["B"]
    print(f"Target global batch size: {target_batch_size}")

    if world_size == 1:
        num_grad_accu1 = h_dict['N']/1_000_000 * h_dict['B']/(128 *256)/4
        num_grad_accu2 = h_dict['B']/3072
        num_grad_accu = max(num_grad_accu1, num_grad_accu2)
    else:
        num_grad_accu1 = h_dict['N']/1_000_000 * h_dict['B']/(128 *256 * world_size)/4
        num_grad_accu2 = h_dict['B']/(3072 * world_size)
        num_grad_accu = max(num_grad_accu1, num_grad_accu2)

    if num_grad_accu > 1.0:
        grad_accumulation_steps = int(np.ceil(num_grad_accu))
        BS = int(np.round(target_batch_size / (grad_accumulation_steps * world_size)))
        effective_batch_size = BS * world_size * grad_accumulation_steps
        print(f"Using gradient accumulation with {grad_accumulation_steps} steps")
        print(f"  Per-device batch size: {BS}")
        print(f"  Num GPUs: {world_size}")
        print(f"  Effective global batch size: {effective_batch_size}")

        # CHANGE: Only rank 0 writes files
        if local_rank in [-1, 0]:
            with open(os.path.join(out_path, 'grad_accumulation_info.txt'), 'w') as f:
                f.write(f"Using per-device batch size {BS} with gradient accumulation steps {grad_accumulation_steps} for global batch size {target_batch_size}\n")
                f.write(f"Num GPUs: {world_size}, Effective batch size: {effective_batch_size}")
    else:
        BS = int(target_batch_size / world_size)
        grad_accumulation_steps = 1
        effective_batch_size = BS * world_size
        print(f"Using per-device batch size {BS} with no gradient accumulation")
        print(f"  Num GPUs: {world_size}")
        print(f"  Effective global batch size: {effective_batch_size}")

        # CHANGE: Only rank 0 writes files
        if local_rank in [-1, 0]:
            with open(os.path.join(out_path, 'grad_accumulation_info.txt'), 'w') as f:
                f.write(f"Using per-device batch size {BS} with no gradient accumulation for global batch size {target_batch_size}\n")
                f.write(f"Num GPUs: {world_size}, Effective batch size: {effective_batch_size}")

    ################################################################################
    # NEW: Adjust dataloader workers based on sequence length
    ################################################################################
    if seq_length >= 8192:
        dataloader_num_workers = 4
        dataloader_prefetch_factor = 2
    elif seq_length >= 2048:
        dataloader_num_workers = 8
        dataloader_prefetch_factor = 4
    else:
        dataloader_num_workers = 12
        dataloader_prefetch_factor = 4
    ################################################################################
    # END NEW
    ################################################################################

    ################################################################################
    # NEW: Use bf16 for StripedHyena, fp16 for others
    ################################################################################
    if nn_dict["model"] == "striped_hyena":
        use_fp16 = False
        use_bf16 = True
    else:
        use_fp16 = True
        use_bf16 = False
    ################################################################################
    # END NEW
    ################################################################################

    # CHANGE: Modified TrainingArguments - only set ddp params if actually in multi-GPU mode
    training_args_dict = {
        "output_dir": path,
        "per_device_train_batch_size": BS,
        "per_device_eval_batch_size": BS,
        "gradient_accumulation_steps": grad_accumulation_steps,
        "eval_strategy": "steps",
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": True,
        "dataloader_prefetch_factor": dataloader_prefetch_factor,
        "max_steps": h_dict["K"],
        "save_strategy": "steps",
        "save_steps": 10000,
        "save_total_limit": 5,
        "logging_steps": 1000,
        ################################################################################
        # NEW: Use bf16 for StripedHyena
        ################################################################################
        "fp16": use_fp16,
        "bf16": use_bf16,
        ################################################################################
        # END NEW
        ################################################################################
        "push_to_hub": False,
    }

    # CHANGE: Only add DDP settings if we're actually in a distributed environment
    if world_size > 1 or local_rank != -1:
        print("Adding DDP configuration for multi-GPU training")
        training_args_dict.update({
            "ddp_backend": "nccl",
            "ddp_find_unused_parameters": False,
            "local_rank": local_rank,
        })
    else:
        print("Single-GPU mode: not setting DDP parameters")

    args = TrainingArguments(**training_args_dict)


    trainer = CustomTrainer(
        momentum=h_dict["momentum"],
        initial_learning_rate=h_dict["lr"],
        lr_scheduler_type=h_dict["lr_schedule"],
        step_decay_schedule_dict=h_dict["step_decay_schedule"],
        optimizer_type=h_dict["optimizer"],
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # CHANGE: Checkpoint search with multi-GPU safety and error handling
    search_ground = "/mfs1/u/chuning/scale/outputs/nn_archive/runs"

    def _is_subset(sub, sup):
        """Recursive dict subset check; lists/scalars must match exactly."""
        assert isinstance(sub, dict) and isinstance(sup, dict)
        for key, value in sub.items():
            if not (key in sup and sup[key] == value):
                return False
        return True

    def find_matching_subdir(nn_dict, h_dict):
        """Find directories matching both nn_dict and h_dict - with error handling"""
        candidates = []
        root = Path(search_ground)

        if not root.exists():
            if local_rank in [-1, 0]:
                print(f"Warning: Search ground {search_ground} does not exist")
            return candidates

        for subdir in (p for p in root.iterdir() if p.is_dir()):
            try:
                # Check if the directory name contains "2025_12_" or "2026_01_"
                if "Run_2025_12_" not in subdir.name and "Run_2026_01_" not in subdir.name:
                    continue

                h_yaml_path = subdir / "hdict.yaml"
                m_yaml_path = subdir / "mdict.yaml"

                # Check if both YAML files exist
                if not (h_yaml_path.exists() and m_yaml_path.exists()):
                    continue

                h_yaml = yaml.safe_load(h_yaml_path.read_text(encoding="utf-8")) or {}
                m_yaml = yaml.safe_load(m_yaml_path.read_text(encoding="utf-8")) or {}

                if _is_subset(h_yaml, h_dict) and _is_subset(m_yaml, nn_dict):
                    candidates.append(subdir)
                    if local_rank in [-1, 0]:
                        print(f"  Found matching checkpoint directory: {subdir.name}")
            except Exception as e:
                if local_rank in [-1, 0]:
                    print(f"  Error processing {subdir.name}: {e}")
                continue

        return candidates

    def find_latest_checkpoint(paths_to_checkpoints):
        """Find most recent checkpoint by number"""
        checkpoint_candidates = []
        for path_to_checkpoints in paths_to_checkpoints:
            try:
                for subdir in (p for p in path_to_checkpoints.iterdir() if p.is_dir()):
                    if subdir.is_dir() and subdir.name.startswith("checkpoint-"):
                        checkpoint_candidates.append(subdir)
            except Exception as e:
                if local_rank in [-1, 0]:
                    print(f"  Error scanning checkpoints in {path_to_checkpoints}: {e}")
                continue

        if checkpoint_candidates:
            checkpoint_candidates.sort(key=lambda p: int(p.name.split("-")[-1]))
            return checkpoint_candidates[-1]
        return None

    # CHANGE: Only rank 0 searches for checkpoints
    latest_checkpoint_path = None
    if local_rank in [-1, 0]:
        print("\n" + "="*60)
        print("Searching for matching checkpoints...")
        print("="*60)
        matching_subdir = find_matching_subdir(nn_dict, h_dict)

        if matching_subdir:
            print(f"Found {len(matching_subdir)} matching directory(ies)")
            latest_checkpoint_path = find_latest_checkpoint(matching_subdir)
        else:
            print("No matching checkpoint directories found")

        print(f"\nLatest checkpoint path: {latest_checkpoint_path}")
        print("="*60 + "\n")

        # Only rank 0 writes checkpoint info
        with open("/mfs1/u/chuning/scale_new/last_checkpoint_path.txt", "w") as f:
            f.write(str(latest_checkpoint_path) if latest_checkpoint_path else "None")
            f.write("\n")
            f.write("Dec")

        with open(os.path.join(path, 'resume_from_checkpoint.txt'), 'w') as f:
            f.write(str(latest_checkpoint_path) if latest_checkpoint_path else "None")

    # CHANGE: Broadcast checkpoint path to all GPUs
    if world_size > 1:
        # Convert to list for broadcasting
        checkpoint_list = [str(latest_checkpoint_path) if latest_checkpoint_path else "None"]

        if dist.is_initialized():
            if local_rank in [-1, 0]:
                print(f"Rank {local_rank}: Broadcasting checkpoint path to all GPUs...")

            dist.broadcast_object_list(checkpoint_list, src=0)
            latest_checkpoint_path = checkpoint_list[0] if checkpoint_list[0] != "None" else None

            if local_rank != 0:
                print(f"Rank {local_rank}: Received checkpoint path: {latest_checkpoint_path}")

    # Convert to string or None for trainer
    checkpoint_str = str(latest_checkpoint_path) if latest_checkpoint_path else None

    if local_rank in [-1, 0]:
        print(f"\nAll ranks ready. Will resume from checkpoint: {checkpoint_str if checkpoint_str else 'None (training from scratch)'}\n")

    # CHANGE: Wrap training in try-finally to ensure cleanup
    try:
        # Train the model
        trainer.train(resume_from_checkpoint=checkpoint_str)

        trainer.save_model()

        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")

        # CHANGE: Only rank 0 copies files
        if local_rank in [-1, 0]:
            loss_curve_df = pd.read_csv(os.path.join(path, 'loss_curve_df.csv'))
            loss_curve_df.to_csv(os.path.join(out_path, 'loss_curve_df.csv'), index=False)

    finally:
        # CHANGE: Clean up distributed process group if it was initialized
        if world_size > 1 or local_rank != -1:
            if dist.is_initialized():
                print(f"\nRank {local_rank}: Cleaning up distributed process group...")
                dist.destroy_process_group()
                print(f"Rank {local_rank}: Distributed process group destroyed successfully.")




if __name__ == "__main__":
    # CHANGE: Add signal handlers for graceful shutdown
    import signal
    import sys

    def signal_handler(sig, frame):
        """Handle interrupt signals gracefully."""
        print("\n\nReceived interrupt signal. Cleaning up...")
        if dist.is_initialized():
            print("Destroying process group...")
            dist.destroy_process_group()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        main()
    except Exception as e:
        # CHANGE: Ensure cleanup even on exceptions
        print(f"\nException occurred: {e}")
        if dist.is_initialized():
            print("Cleaning up distributed process group...")
            dist.destroy_process_group()
        raise

 # command to run:
 # Single GPU: python hf_utils_train_model_w_genomes.py
 # Multi-GPU:  torchrun --nproc_per_node=2 hf_utils_train_model_w_genomes.py
