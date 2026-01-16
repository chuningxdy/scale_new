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

# NEW: SSM configurations based on the table in configurations.png
SSM_configs = pd.DataFrame({
    'model': [
        '1', '6', '17', '29', '40', '59', '69', '84', '99', '114', '121', '135' ,
        '158', '175', '203', '232', '266', '303', '383', '473', '572', '680',
        '798', '926', '1063', '1209'
    ],
    'd_model': [
        128, 320, 448, 512, 576, 640, 640, 704, 768, 768, 768, 768,
        832, 832, 896, 896, 960, 1024, 1152, 1280, 1408, 1536,
        1664, 1792, 1920, 1920
    ],
    'glu_size': [
        336, 848, 1200, 1360, 1536, 1696, 1712, 1872, 2048, 2048, 2048, 2048,
        2224, 2224, 2384, 2384, 2560, 2736, 3072, 3408, 3760, 4096,
        4432, 4784, 5120, 5120
    ],
    'kv_size': [
        64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
        64, 64, 64, 64, 64, 64, 64, 64, 128, 128,
        128, 128, 128, 128
    ],
    'n_heads': [
        2, 5, 7, 8, 8, 10, 10, 11, 12, 12, 12, 12,
        13, 13, 14, 14, 15, 16, 18, 20, 11, 12,
        13, 14, 15, 15
    ],
    'n_layer': [
        4, 5, 7, 9, 10, 12, 14, 14, 14, 16, 17, 19,
        19, 21, 21, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 25
    ],
    'learning_rate': [
        9.77e-4, 9.57e-4, 9.36e-4, 9.15e-4, 8.95e-4, 8.70e-4, 8.56e-4,
        8.37e-4, 8.18e-4, 8.00e-4, 7.75e-4, 7.50e-4,
        7.25e-4, 7.00e-4, 6.75e-4, 6.50e-4, 6.25e-4,
        6.00e-4, 5.66e-4, 5.33e-4, 5.00e-4, 4.75e-4,
        4.55e-4, 4.33e-4, 4.15e-4, 4.11e-4
    ],
    'total_parameters_count': [
        1e6, 6e6, 17e6, 29e6, 40e6, 59e6, 69e6, 84e6, 99e6, 114e6, 121e6, 135e6,
        158e6, 175e6, 203e6, 232e6, 266e6, 303e6, 383e6, 473e6, 572e6, 680e6,
        798e6, 926e6, 1063e6, 1209e6
    ]
})



import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM
import torch
import os
import random

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

# NEW: Function for SSM config prediction
def func_ssm_config_from_params_count(df, plot_filename=None):
    """
    Creates a function that predicts SSM model config values based on parameter count.
    Predicts ALL architecture parameters: d_model, glu_size, kv_size, n_heads, n_layer

    Args:
        df: DataFrame with SSM model configs including total_parameters_count
        plot_filename: If provided, save a visualization to this filename

    Returns:
        A function that takes parameter count and returns predicted config values for SSM
    """
    X = np.log(df['total_parameters_count'])
    X_millions = df['total_parameters_count'] / 1e6

    # Predict all architecture parameters from the table
    target_columns = ['d_model', 'glu_size', 'kv_size', 'n_heads', 'n_layer']

    models = {}

    if plot_filename:
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3)
        plt.suptitle(f'Log-Log Regression: SSM Architecture vs. Parameter Count\nModel family: StripedHyena',
                    fontsize=16)

    for i, col in enumerate(target_columns):
        y = np.log(df[col])
        y_actual = df[col]

        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        models[col] = (intercept, slope)

        print(f"Log-log regression for {col}:")
        print(f"Intercept: {intercept:.4f}, Coefficient: {slope:.4f}, R²: {r_value**2:.4f}")

        if plot_filename:
            ax = plt.subplot(gs[i // 3, i % 3])

            ax.scatter(X_millions, y_actual, color='blue', s=100, alpha=0.7, label='Actual values')

            x_range = np.logspace(np.log10(min(X_millions)*0.7), np.log10(max(X_millions)*1.3), 100)
            y_pred = np.exp(intercept + slope * np.log(x_range * 1e6))

            ax.plot(x_range, y_pred, 'r-', linewidth=2, label=f'Regression (R²={r_value**2:.3f})')

            eqn_text = f"ln(y) = {intercept:.2f} + {slope:.2f} × ln(x)"
            ax.text(0.05, 0.95, eqn_text, transform=ax.transAxes,
                   fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.set_xlabel('Parameter Count (millions)', fontsize=12)
            ax.set_ylabel(col.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{col.replace("_", " ").title()} vs Model Size', fontsize=14)

            for j, model in enumerate(df['model']):
                ax.annotate(model, (X_millions[j], y_actual[j]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)

            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='best')

    if plot_filename:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")
        plt.close()

    def predict_ssm_config(params_count):
        """
        Predicts SSM model configuration based on parameter count.

        Args:
            params_count: Total parameter count (e.g., 47e6 for 47M)

        Returns:
            Dictionary with predicted configuration values for SSM
        """
        log_params = np.log(params_count)
        result = {'non_embed_parameters_count': params_count}

        # Predict d_model - round to multiple of 16 for GPU efficiency
        intercept, slope = models['d_model']
        predicted_d_model = np.exp(intercept + slope * log_params)
        result['d_model'] = int(round(predicted_d_model / 16) * 16)
        if result['d_model'] < 16:
            result['d_model'] = 16

        # Predict glu_size (inner_mlp_size) - round to multiple of 16
        intercept, slope = models['glu_size']
        predicted_glu_size = np.exp(intercept + slope * log_params)
        result['glu_size'] = int(round(predicted_glu_size / 16) * 16)
        if result['glu_size'] < 16:
            result['glu_size'] = 16

        # Predict kv_size (per-head dimension) - typically 64 or 128
        intercept, slope = models['kv_size']
        predicted_kv_size = np.exp(intercept + slope * log_params)
        # round to nearest 16
        result['kv_size'] = int(round(predicted_kv_size / 16) * 16)
        if result['kv_size'] < 16:
            result['kv_size'] = 16

        # Predict n_heads (number of attention heads)
        intercept, slope = models['n_heads']
        predicted_n_heads = np.exp(intercept + slope * log_params)
        result['n_heads'] = int(round(predicted_n_heads))
        if result['n_heads'] < 1:
            result['n_heads'] = 1

        # Ensure d_model is divisible by n_heads AND head_dim is flash-attention compatible
        # Flash attention supports head_dim of 32, 64, 128, 256
        d_model = result['d_model']
        n_heads = result['n_heads']
        flash_attn_head_dims = [32, 64, 128, 256]

        # Find n_heads that gives a flash-attention compatible head_dim
        valid_n_heads_options = []
        for head_dim in flash_attn_head_dims:
            if d_model % head_dim == 0:
                candidate_n_heads = d_model // head_dim
                if candidate_n_heads >= 1:
                    valid_n_heads_options.append((candidate_n_heads, head_dim))

        if valid_n_heads_options:
            # Choose the n_heads closest to the predicted value
            best_n_heads, best_head_dim = min(valid_n_heads_options, key=lambda x: abs(x[0] - n_heads))
            result['n_heads'] = best_n_heads
        else:
            # Fallback: adjust d_model to be divisible by a valid head_dim
            # Round d_model to nearest multiple of 64 (most common head_dim)
            result['d_model'] = int(round(d_model / 64) * 64)
            if result['d_model'] < 64:
                result['d_model'] = 64
            # Recalculate n_heads
            d_model = result['d_model']
            for head_dim in flash_attn_head_dims:
                if d_model % head_dim == 0:
                    result['n_heads'] = d_model // head_dim
                    break

        # Predict n_layer (number of layers)
        intercept, slope = models['n_layer']
        predicted_n_layer = np.exp(intercept + slope * log_params)
        result['n_layer'] = int(round(predicted_n_layer))
        if result['n_layer'] < 2:
            result['n_layer'] = 2

        return result

    return predict_ssm_config

# test func_ssm_config_from_params_count
func_ssm_config_from_params_count(SSM_configs, plot_filename="ssm_config_regression.png")


def build_model_with_predicted_config(
        target_param_count,
        tokenizer_path,
        model_family = "llama",
        reference_configs = Pythia_configs,
        save_dir=None,
        hybrid_pattern="alternate",
        vocab_size_override=None):  # Use this to override tokenizer vocab (e.g., 5 for binary genome data)
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
    # Add support for SSM (Hybrid SSM/Attention) models - Mamba-style
    elif model_family == "ssm":
        print(f"\nBuilding a Hybrid SSM/Attention model with target size: {target_param_count/1e6:.1f}M parameters")
        # Use our custom hybrid model implementation
        from hybrid_ssm_model import HybridSSMModel, HybridSSMConfig
        config_creation_function = HybridSSMConfig
        model_creation_function = HybridSSMModel
    # Add support for Hyena (StripedHyena-Lite) models - Conv-based mixing
    elif model_family == "hyena":
        print(f"\nBuilding a StripedHyena-Lite model with target size: {target_param_count/1e6:.1f}M parameters")
        from striped_hyena import StripedHyenaLiteForCausalLM, StripedHyenaLiteConfig
        config_creation_function = StripedHyenaLiteConfig
        model_creation_function = StripedHyenaLiteForCausalLM
    # Add support for Together AI StripedHyena from HuggingFace
    elif model_family == "striped_hyena":
        print(f"\nBuilding a Together AI StripedHyena model with target size: {target_param_count/1e6:.1f}M parameters")
        from transformers import AutoConfig, AutoModelForCausalLM
        # Use the Together StripedHyena architecture with custom config
        config_creation_function = "striped_hyena_config"  # Special marker for custom handling
        model_creation_function = "striped_hyena_model"    # Special marker for custom handling
    else:
        raise ValueError("Invalid model class. Choose 'llama', 'pythia', 'ssm', 'hyena', or 'striped_hyena'.")

    # Use SSM predictor for SSM/Hyena/StripedHyena models (same architecture config table)
    if model_family in ["ssm", "hyena", "striped_hyena"]:
        predictor = func_ssm_config_from_params_count(SSM_configs)
    else:
        predictor = func_config_from_params_count(reference_configs)


    # Handle vocab size
    if vocab_size_override is not None:
        vocab_size = vocab_size_override
        print(f"\nUsing vocab_size override: {vocab_size}")
        tokenizer = None
    elif tokenizer_path is not None:
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
        tokenizer = None



    def simplified_param_count(config_dict, vocab_size, model_family, hybrid_pattern="alternate", hyena_kernel_size=127):
        # Helper function to get layer counts based on hybrid_pattern
        def get_layer_counts(n_layer, hybrid_pattern):
            if hybrid_pattern == "alternate":
                n_mix_layers = (n_layer + 1) // 2  # Even indices: 0,2,4,...
                n_attn_layers = n_layer // 2      # Odd indices: 1,3,5,...
            elif hybrid_pattern in ["ssm_heavy", "evo2_7b", "every_4"]:
                # Every 4th layer is attention, starting at 3
                n_attn_layers = len(list(range(3, n_layer, 4)))
                n_mix_layers = n_layer - n_attn_layers
            elif hybrid_pattern in ["evo2_40b", "every_8"]:
                # Every 8th layer is attention, starting at 7
                n_attn_layers = len(list(range(7, n_layer, 8)))
                n_mix_layers = n_layer - n_attn_layers
            elif hybrid_pattern in ["ssm_only", "hyena_only"]:
                n_mix_layers = n_layer
                n_attn_layers = 0
            elif hybrid_pattern == "attn_only":
                n_mix_layers = 0
                n_attn_layers = n_layer
            else:
                # Default to every_4 for hyena, alternate for ssm
                n_attn_layers = len(list(range(3, n_layer, 4)))
                n_mix_layers = n_layer - n_attn_layers
            return n_mix_layers, n_attn_layers

        # NEW: Support for Hyena model (striped_hyena.py)
        if model_family == "hyena":
            d_model = config_dict['d_model']
            n_layer = config_dict['n_layer']
            d_inner = config_dict['glu_size']  # d_inner in hyena = glu_size
            head_dim = config_dict['kv_size']  # head_dim in hyena = kv_size
            n_heads = config_dict['n_heads']
            n_kv_heads = config_dict.get('n_kv_heads', n_heads)  # GQA support

            n_hyena_layers, n_attn_layers = get_layer_counts(n_layer, hybrid_pattern)

            # HyenaBlock parameters (HyenaLikeMix + MLP + 2 norms):
            # - in_proj: d_model * 2 * d_inner
            # - dwconv: d_inner * kernel_size + d_inner (depthwise weights + bias)
            # - out_proj: d_inner * d_model
            # - gate_proj: d_model * d_inner
            # - up_proj: d_model * d_inner
            # - down_proj: d_inner * d_model
            # - 2x RMSNorm: 2 * d_model
            hyena_layer_params = (
                d_model * 2 * d_inner +             # in_proj
                d_inner * hyena_kernel_size +       # dwconv weights
                d_inner +                           # dwconv bias
                d_inner * d_model +                 # out_proj
                d_model * d_inner +                 # gate_proj (MLP)
                d_model * d_inner +                 # up_proj (MLP)
                d_inner * d_model +                 # down_proj (MLP)
                2 * d_model                         # 2x RMSNorm
            )

            # AttentionBlock parameters (SDPA + MLP + 2 norms):
            # - q_proj: d_model * n_heads * head_dim
            # - k_proj: d_model * n_kv_heads * head_dim
            # - v_proj: d_model * n_kv_heads * head_dim
            # - o_proj: n_heads * head_dim * d_model
            # - gate_proj: d_model * d_inner
            # - up_proj: d_model * d_inner
            # - down_proj: d_inner * d_model
            # - 2x RMSNorm: 2 * d_model
            attn_layer_params = (
                d_model * n_heads * head_dim +      # q_proj
                d_model * n_kv_heads * head_dim +   # k_proj
                d_model * n_kv_heads * head_dim +   # v_proj
                n_heads * head_dim * d_model +      # o_proj
                d_model * d_inner +                 # gate_proj (MLP)
                d_model * d_inner +                 # up_proj (MLP)
                d_inner * d_model +                 # down_proj (MLP)
                2 * d_model                         # 2x RMSNorm
            )

            # Total layer parameters
            total_layer_params = n_hyena_layers * hyena_layer_params + n_attn_layers * attn_layer_params

            # Final norm
            final_norm_params = d_model

            simplified_non_embedding_params_cnt = total_layer_params + final_norm_params

            # Embeddings (tied, so count once)
            embedding_params_cnt = vocab_size * d_model
            simplified_total_params_cnt = simplified_non_embedding_params_cnt + embedding_params_cnt

        # Together AI StripedHyena from HuggingFace
        elif model_family == "striped_hyena":
            d_model = config_dict['d_model']
            n_layer = config_dict['n_layer']
            d_inner = config_dict['glu_size']  # Maps to inner_mlp_size
            n_heads = config_dict['n_heads']

            # StripedHyena config defaults
            proj_groups = 4  # GQA groups (default from StripedHyena config)
            n_kv_heads = max(1, n_heads // proj_groups)
            short_filter_length = 3  # Default short filter length
            state_size = 2  # Default state size for poles/residues

            n_hyena_layers, n_attn_layers = get_layer_counts(n_layer, hybrid_pattern)

            # StripedHyena ParallelGatedConvBlock (Hyena layer) params:
            # - pre_norm (RMSNorm): d_model
            # - post_norm (RMSNorm): d_model
            # - projections: Linear(d_model, 3*d_model) with bias = 3*d_model*d_model + 3*d_model
            # - out_filter_dense: Linear(d_model, d_model) with bias = d_model*d_model + d_model
            # - mlp (ParallelGatedMLP): l1 + l2 + l3 = 3 * d_model * d_inner (no bias)
            # - filter.D: d_model
            # - filter.short_filter_weight: (3*d_model, 1, short_filter_length) = 3*d_model*short_filter_length
            # - filter.short_filter_bias: 3*d_model
            # - filter.poles: (d_model, state_size, 1, 2) = d_model * state_size * 2
            # - filter.residues: (d_model, state_size, 1, 2) = d_model * state_size * 2
            hyena_layer_params = (
                d_model +                                    # pre_norm
                d_model +                                    # post_norm
                3 * d_model * d_model + 3 * d_model +        # projections (weight + bias)
                d_model * d_model + d_model +                # out_filter_dense (weight + bias)
                3 * d_model * d_inner +                      # mlp (SwiGLU: l1, l2, l3)
                d_model +                                    # filter.D
                3 * d_model * short_filter_length +          # filter.short_filter_weight
                3 * d_model +                                # filter.short_filter_bias
                d_model * state_size * 2 +                   # filter.poles
                d_model * state_size * 2                     # filter.residues
            )

            # StripedHyena AttentionBlock params (using flash_attn MHA with GQA):
            # - pre_norm (RMSNorm): d_model
            # - post_norm (RMSNorm): d_model
            # - inner_mha_cls (MHA with GQA, proj_groups=4, no bias):
            #   - q_proj: d_model * d_model (n_heads * head_dim = d_model)
            #   - k_proj: d_model * (d_model / proj_groups)
            #   - v_proj: d_model * (d_model / proj_groups)
            #   - out_proj: d_model * d_model
            # - mlp (ParallelGatedMLP): 3 * d_model * d_inner
            attn_layer_params = (
                d_model +                                    # pre_norm
                d_model +                                    # post_norm
                d_model * d_model +                          # q_proj
                d_model * d_model // proj_groups +           # k_proj (GQA)
                d_model * d_model // proj_groups +           # v_proj (GQA)
                d_model * d_model +                          # out_proj
                3 * d_model * d_inner                        # mlp (SwiGLU)
            )

            total_layer_params = n_hyena_layers * hyena_layer_params + n_attn_layers * attn_layer_params

            # Final norm
            final_norm_params = d_model

            simplified_non_embedding_params_cnt = total_layer_params + final_norm_params

            # Embeddings (UNTIED due to bug in StripedHyena code - embed and unembed are separate)
            embedding_params_cnt = vocab_size * d_model * 2  # embed + unembed
            simplified_total_params_cnt = simplified_non_embedding_params_cnt + embedding_params_cnt

        # SSM model (hybrid_ssm_model.py) with Mamba-style selective scan
        elif model_family == "ssm":
            d_model = config_dict['d_model']
            n_layer = config_dict['n_layer']
            glu_size = config_dict['glu_size']
            kv_size = config_dict['kv_size']
            n_heads = config_dict['n_heads']
            d_state = 16  # Default SSM state dimension
            d_conv = 4    # Default convolution kernel size

            n_ssm_layers, n_attn_layers = get_layer_counts(n_layer, hybrid_pattern)

            # SSM Layer parameters (SSMBlock + norm):
            # - in_proj: d_model * 2 * glu_size
            # - conv1d: glu_size * d_conv (grouped, so just d_conv per channel)
            # - x_proj: glu_size * (dt_rank + 2*d_state), where dt_rank ≈ d_model/16
            # - dt_proj: dt_rank * glu_size
            # - A_log: glu_size * d_state
            # - D: glu_size
            # - out_proj: glu_size * d_model
            # - RMSNorm: d_model
            dt_rank = (d_model + 15) // 16
            ssm_layer_params = (
                d_model * 2 * glu_size +           # in_proj
                glu_size * d_conv +                 # conv1d
                glu_size * (dt_rank + 2*d_state) +  # x_proj
                dt_rank * glu_size +                # dt_proj
                glu_size * d_state +                # A_log
                glu_size +                          # D
                glu_size * d_model +                # out_proj
                d_model                             # RMSNorm
            )

            # Attention Layer parameters (Attention + MLP + 2 norms):
            # - q_proj: d_model * n_heads * kv_size
            # - k_proj: d_model * n_kv_heads * kv_size (assuming n_kv_heads = n_heads for now)
            # - v_proj: d_model * n_kv_heads * kv_size
            # - o_proj: n_heads * kv_size * d_model
            # - gate_proj: d_model * glu_size
            # - up_proj: d_model * glu_size
            # - down_proj: glu_size * d_model
            # - 2x RMSNorm: 2 * d_model
            attn_layer_params = (
                d_model * n_heads * kv_size +       # q_proj
                d_model * n_heads * kv_size +       # k_proj
                d_model * n_heads * kv_size +       # v_proj
                n_heads * kv_size * d_model +       # o_proj
                d_model * glu_size +                # gate_proj
                d_model * glu_size +                # up_proj
                glu_size * d_model +                # down_proj
                2 * d_model                         # 2x RMSNorm
            )

            # Total layer parameters
            total_layer_params = n_ssm_layers * ssm_layer_params + n_attn_layers * attn_layer_params

            # Final norm
            final_norm_params = d_model

            simplified_non_embedding_params_cnt = total_layer_params + final_norm_params

            # Embeddings (tied, so count once)
            embedding_params_cnt = vocab_size * d_model
            simplified_total_params_cnt = simplified_non_embedding_params_cnt + embedding_params_cnt
        else:
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
        non_embed_params_simplified, total_params_simplified = simplified_param_count(config_dict, vocab_size, model_family, hybrid_pattern)
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

        # Create SSM config using our custom implementation (Mamba-style)
        if model_family == "ssm":
            print("\nCreating Hybrid SSM/Attention config...")

            # Create config directly with our predicted values
            config = config_creation_function(
                vocab_size=vocab_size,
                d_model=config_dict['d_model'],
                n_layers=config_dict['n_layer'],
                n_heads=config_dict['n_heads'],
                glu_size=config_dict['glu_size'],
                kv_size=config_dict['kv_size'],
                d_state=16,  # SSM state dimension - typical value
                d_conv=4,    # Convolution kernel size for SSM
                dropout=0.0,
                bias=False,
                hybrid_pattern=hybrid_pattern  # Use the specified hybrid pattern
            )

            print(f"  Architecture Details:")
            print(f"    - d_model (hidden_size): {config_dict['d_model']}")
            print(f"    - glu_size (inner_mlp_size): {config_dict['glu_size']}")
            print(f"    - n_heads: {config_dict['n_heads']}")
            print(f"    - kv_size (per-head): {config_dict['kv_size']}")
            print(f"    - n_layers: {config_dict['n_layer']}")
            print(f"    - d_state (SSM): {config.d_state}")
            print(f"    - hybrid_pattern: {hybrid_pattern}")
            print(f"    - SSM layers: {len(config.ssm_layer_idxs)} at {config.ssm_layer_idxs}")
            print(f"    - Attention layers: {len(config.attn_layer_idxs)} at {config.attn_layer_idxs}")
            print(f"  Note: Using custom Hybrid SSM/Attention model (no flash_attn required)")

        # Create Hyena config using StripedHyena-Lite (Conv-based mixing)
        elif model_family == "hyena":
            print("\nCreating StripedHyena-Lite config...")

            # Get hyena-specific parameters from nn_dict or use defaults
            hyena_kernel_size = 127  # Default kernel size for ~2k context

            # Create config with predicted values
            config = config_creation_function(
                vocab_size=vocab_size,
                d_model=config_dict['d_model'],
                n_layers=config_dict['n_layer'],
                n_heads=config_dict['n_heads'],
                head_dim=config_dict['kv_size'],  # kv_size maps to head_dim
                d_inner=config_dict['glu_size'],  # glu_size maps to d_inner
                hyena_kernel_size=hyena_kernel_size,
                hybrid_pattern=hybrid_pattern,
                max_position_embeddings=2048,  # Default for genomics
                dropout=0.0,
                bias=False,
            )

            print(f"  Architecture Details:")
            print(f"    - d_model (hidden_size): {config_dict['d_model']}")
            print(f"    - d_inner (glu_size): {config_dict['glu_size']}")
            print(f"    - n_heads: {config_dict['n_heads']}")
            print(f"    - head_dim (kv_size): {config_dict['kv_size']}")
            print(f"    - n_layers: {config_dict['n_layer']}")
            print(f"    - hyena_kernel_size: {hyena_kernel_size}")
            print(f"    - hybrid_pattern: {hybrid_pattern}")
            print(f"    - Hyena layers: {len(config.ssm_layer_idxs)} at {config.ssm_layer_idxs}")
            print(f"    - Attention layers: {len(config.attn_layer_idxs)} at {config.attn_layer_idxs}")
            print(f"  Note: Using StripedHyena-Lite (Conv-based, no CUDA extensions required)")

        # Create Together AI StripedHyena config from HuggingFace
        elif model_family == "striped_hyena":
            print("\nCreating Together AI StripedHyena config...")
            from transformers import AutoConfig, AutoModelForCausalLM

            # Determine layer pattern based on hybrid_pattern
            def get_striped_hyena_layer_idxs(n_layer, hybrid_pattern):
                if hybrid_pattern == "alternate":
                    # Alternate: even=hyena, odd=attention
                    attn_layer_idxs = list(range(1, n_layer, 2))
                elif hybrid_pattern in ["ssm_heavy", "evo2_7b", "every_4"]:
                    # Every 4th layer is attention
                    attn_layer_idxs = list(range(3, n_layer, 4))
                elif hybrid_pattern in ["evo2_40b", "every_8"]:
                    # Every 8th layer is attention
                    attn_layer_idxs = list(range(7, n_layer, 8))
                elif hybrid_pattern in ["ssm_only", "hyena_only"]:
                    attn_layer_idxs = []
                elif hybrid_pattern == "attn_only":
                    attn_layer_idxs = list(range(n_layer))
                else:
                    # Default to every_4
                    attn_layer_idxs = list(range(3, n_layer, 4))
                return attn_layer_idxs

            attn_layer_idxs = get_striped_hyena_layer_idxs(config_dict['n_layer'], hybrid_pattern)
            # Hyena layers are all layers NOT in attn_layer_idxs
            hyena_layer_idxs = [i for i in range(config_dict['n_layer']) if i not in attn_layer_idxs]

            # Load the base config from Together's StripedHyena and modify it
            # Using trust_remote_code=True to load the custom architecture
            try:
                base_config = AutoConfig.from_pretrained(
                    "togethercomputer/StripedHyena-Hessian-7B",
                    trust_remote_code=True
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to load Together AI StripedHyena config from HuggingFace: {e}\n"
                    "Please ensure you have internet access and the transformers library is up to date.\n"
                    "You may need to run: pip install --upgrade transformers"
                )

            # Override with our predicted architecture values
            base_config.vocab_size = vocab_size
            base_config.hidden_size = config_dict['d_model']
            base_config.num_filters = config_dict['d_model']  # Must equal hidden_size for the assertion
            base_config.num_layers = config_dict['n_layer']
            base_config.num_attention_heads = config_dict['n_heads']
            base_config.inner_mlp_size = config_dict['glu_size']  # StripedHyena uses inner_mlp_size, not intermediate_size
            base_config.head_dim = config_dict['kv_size']
            base_config.attn_layer_idxs = attn_layer_idxs
            base_config.hyena_layer_idxs = hyena_layer_idxs
            base_config.tie_word_embeddings = True
            # Note: StripedHyena's tie_embeddings has a bug (references self.emb instead of self.embedding_layer)
            # Keep it False to use separate unembed weights (default behavior)
            # Set dtypes - use float32 for Hyena blocks to avoid numerical instability
            # The Hyena filter uses complex exponentials that can overflow in bfloat16
            base_config.hyena_block_dtype = "float32"
            base_config.attn_block_dtype = "bfloat16"
            base_config.mlp_dtype = "bfloat16"
            # Disable vocab_size adjustment to avoid mismatch in loss computation
            # (the model adjusts vocab_size to be divisible by this value, but loss uses config.vocab_size)
            base_config.make_vocab_size_divisible_by = 1
            # Enable flash attention for faster training
            base_config.use_flash_attention_2 = True
            base_config.use_flash_attn = True

            config = base_config

            print(f"  Architecture Details:")
            print(f"    - hidden_size (d_model): {config_dict['d_model']}")
            print(f"    - num_filters: {config_dict['d_model']}")
            print(f"    - inner_mlp_size (glu_size): {config_dict['glu_size']}")
            print(f"    - num_attention_heads: {config_dict['n_heads']}")
            print(f"    - head_dim (kv_size): {config_dict['kv_size']}")
            print(f"    - num_layers: {config_dict['n_layer']}")
            print(f"    - hybrid_pattern: {hybrid_pattern}")
            print(f"    - Attention layers at indices: {attn_layer_idxs}")
            print(f"  Note: Using Together AI StripedHyena from HuggingFace (trust_remote_code=True)")

        else:
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
            # Create model from config
            if model_family == "striped_hyena":
                # Special handling for Together AI StripedHyena
                # Convert string dtypes to torch dtypes for model construction
                dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
                dtype_fields = ['hyena_block_dtype', 'attn_block_dtype', 'mlp_dtype']
                original_dtypes = {}
                for field in dtype_fields:
                    if hasattr(config, field) and isinstance(getattr(config, field), str):
                        original_dtypes[field] = getattr(config, field)
                        setattr(config, field, dtype_map.get(getattr(config, field), torch.bfloat16))

                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=True
                )

                # Stable pole initialization for Hyena filters
                # The poles must have |poles| < 1 so that log(poles) has negative real part,
                # ensuring exp(log_poles * t) decays over time instead of exploding.
                # This is standard practice for SSMs (similar to S4, Mamba's log-space A matrix).
                print("  Applying stable pole initialization for Hyena filters...")
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

                # Convert back to strings for JSON serialization when saving
                for field, str_dtype in original_dtypes.items():
                    setattr(config, field, str_dtype)
            else:
                model = model_creation_function(config)
            print(f"  Model class: {model.__class__.__name__}")
        except Exception as e:
            raise ValueError(f"Model initialization failed: {e}. Please check the configuration.")

        param_count_actual = sum(p.numel() for p in model.parameters())

        # Calculate embedding parameters
        if model_family in ["ssm", "hyena"]:
            embedding_params = config.vocab_size * config.d_model  # Tied embeddings (count once)
        elif model_family == "striped_hyena":
            # Untied embeddings due to bug in StripedHyena code (embed and unembed are separate)
            embedding_params = config.vocab_size * config.hidden_size * 2
        else:
            embedding_params = config.vocab_size * config.hidden_size * 2  # Embedding + LM head

        non_embedding_params_actual = param_count_actual - embedding_params
        accuracy = (param_count_actual / target_param_count) * 100
        print(f"trying ratio: {best_param_count/target_param_count:.2f}")
        print(f"Prediction accuracy: {accuracy:.2f}% of target")

    if accuracy < 75 or accuracy > 130:
        raise ValueError(f"Final prediction accuracy {accuracy:.2f}% is outside the acceptable range.")

    # Skip detailed breakdown for SSM/Hyena/StripedHyena as structure is different
    if model_family not in ["ssm", "hyena", "striped_hyena"]:
        attention_params_per_layer = 4 * (config.hidden_size ** 2)
        ffn_params_per_layer = 2 * config.hidden_size * config.intermediate_size

        total_attention_params = attention_params_per_layer * config.num_hidden_layers
        total_ffn_params = ffn_params_per_layer * config.num_hidden_layers

        print(f" - Attention layers: {total_attention_params:,} params")
        print(f" - Feed-forward layers: {total_ffn_params:,} params")
        print(f" - Other (norm, etc.): {param_count_actual - embedding_params - total_attention_params - total_ffn_params:,} params")
    elif model_family == "hyena":
        # Print hyena-specific breakdown
        print(f" - Hyena layers: {len(config.ssm_layer_idxs)}")
        print(f" - Attention layers: {len(config.attn_layer_idxs)}")
    elif model_family == "striped_hyena":
        # Print striped_hyena-specific breakdown
        n_attn = len(config.attn_layer_idxs) if hasattr(config, 'attn_layer_idxs') else 0
        n_hyena = len(config.hyena_layer_idxs) if hasattr(config, 'hyena_layer_idxs') else 0
        print(f" - Hyena layers: {n_hyena} at {config.hyena_layer_idxs if hasattr(config, 'hyena_layer_idxs') else []}")
        print(f" - Attention layers: {n_attn}")

    print(f" - Non-embedding params: {param_count_actual - embedding_params:,} params")
    print(f" - Total params: {param_count_actual:,} params ({param_count_actual})")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Convert any torch.dtype fields to strings for JSON serialization
        if model_family == "striped_hyena":
            dtype_to_str = {torch.bfloat16: "bfloat16", torch.float32: "float32", torch.float16: "float16"}
            for attr in dir(model.config):
                val = getattr(model.config, attr, None)
                if isinstance(val, torch.dtype):
                    setattr(model.config, attr, dtype_to_str.get(val, str(val)))
        # safe_serialization=False needed for weight-tied models (embedding <-> lm_head)
        model.save_pretrained(save_dir, safe_serialization=False)
        if tokenizer_path and tokenizer and os.path.exists(tokenizer_path):
            tokenizer.save_pretrained(save_dir)
        print(f"\nModel saved to {save_dir}")

    return model, config, non_embedding_params_actual, param_count_actual

# test build_model_with_predicted_config
#model, config, non_embed_params, total_params = build_model_with_predicted_config(
 #   target_param_count=47_000_000,
  #  tokenizer_path="datasets/tokenizers/genome_tokenizer",
  #  model_family="ssm",
  #  save_dir=None
#)

#raise ValueError("Stop after test")

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
        # NEW: Add support for genomics dataset
        elif dataset_name in ["opengenome2", "opengenome2_stream"]:
            data_label = "arcinstitute/opengenome2"  # ~5.5TB dataset, 8.8T base pairs
        elif dataset_name == "opengenome2_local":
            data_label = "/mfs1/datasets/pile/opengenome2_16gb"  # Local ~16GB subset
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

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

        # Load local opengenome2 subset from binary file (~16GB downloaded)
        elif dataset_name == "opengenome2_local":
            print(f"Loading local opengenome2 binary dataset from {data_label}")
            import numpy as np
            from torch.utils.data import Dataset as TorchDataset, Subset

            class BinaryGenomeDataset(TorchDataset):
                """PyTorch Dataset for binary genome data (uint16 format)
                Vocab: A=0, C=1, G=2, T=3, N=4 (5 tokens total)
                """
                def __init__(self, data_path, seq_length=2048):
                    bin_file = os.path.join(data_path, "opengenome2_2048_uint16.bin")
                    self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
                    self.seq_length = seq_length
                    self.num_sequences = len(self.data) // seq_length
                    print(f"  Loaded {self.num_sequences:,} sequences of length {seq_length}")

                def __len__(self):
                    return self.num_sequences

                def __getitem__(self, idx):
                    start = idx * self.seq_length
                    end = start + self.seq_length
                    tokens = torch.tensor(self.data[start:end].astype(np.int64), dtype=torch.long)
                    return {"input_ids": tokens}

            # Create dataset and split with random (but fixed) test set selection
            full_dataset = BinaryGenomeDataset(data_label, seq_length=seq_length)
            total_size = len(full_dataset)

            # Randomly select test indices (fixed seed for reproducibility)
            rng = np.random.RandomState(seed=42)
            all_indices = np.arange(total_size)
            rng.shuffle(all_indices)
            test_indices = all_indices[:test_size].tolist()
            train_indices = all_indices[test_size:].tolist()

            train_dataset = Subset(full_dataset, train_indices)
            test_dataset = Subset(full_dataset, test_indices)

            print(f"  Train: {len(train_dataset):,}, Test: {len(test_dataset):,} (randomly selected, seed=42)")

            # Return early - data is already tokenized
            return train_dataset, test_dataset

        # Load full opengenome2 from HuggingFace (5.5TB - not recommended)
        elif dataset_name == "opengenome2":
            dataset = load_dataset(data_label, name=None,
                                   cache_dir=cache_dir,
                                   trust_remote_code=True)

            split_dataset = get_full_dataset(dataset, verbose=verbose)
            filtered_dataset = split_dataset
            stream = False

        elif dataset_name == "opengenome2_stream":
            # Use a specific subset to avoid rate limits
            # Available subsets: https://huggingface.co/datasets/arcinstitute/opengenome2
            # Options: "pretrain_random_archaea", "pretrain_random_bacteria",
            #          "pretrain_random_eukaryota", "pretrain_random_virus", etc.
            subset_name = "pretrain_random_bacteria"  # ~100GB subset, good for training
            print(f"Loading opengenome2 subset: {subset_name}")

            try:
                dataset = load_dataset(
                    data_label,
                    name=subset_name,  # Use specific subset instead of full dataset
                    cache_dir=cache_dir,
                    streaming=True,
                    trust_remote_code=True,
                    token=True
                )
            except Exception as e:
                print(f"Failed to load subset {subset_name}, trying without subset: {e}")
                dataset = load_dataset(
                    data_label,
                    name=None,
                    cache_dir=cache_dir,
                    streaming=True,
                    trust_remote_code=True,
                    token=True,
                    split="train"  # Just get train split
                )

            if hasattr(dataset, 'shuffle'):
                split_dataset = dataset.shuffle(buffer_size=1000, seed=42)
            else:
                split_dataset = dataset["train"].shuffle(buffer_size=1000, seed=42)
            filtered_dataset = split_dataset
            stream = True


        if verbose and not stream:
            removed_count = len(split_dataset) - len(filtered_dataset)
            print(f"Filtered out {removed_count} empty examples ({removed_count/len(split_dataset)*100:.2f}%)")
            print(f"Original dataset size: {len(split_dataset)}, Filtered size: {len(filtered_dataset)}")
        elif verbose and stream:
            print("Streaming dataset loaded (size unknown until iteration)")

        split_dataset = filtered_dataset

        # NEW: Include genomics datasets in tokenization
        if dataset_name in ["wikitext-2-v1", "wikitext-103-v1", "openwebtext2", "openwebtext2_stream",
                           "opengenome2", "opengenome2_stream", "opengenome2_local"]:

            # Determine the text field name (genomics uses "sequence", text datasets use "text")
            if dataset_name in ["opengenome2", "opengenome2_stream", "opengenome2_local"]:
                text_field = "sequence"
            else:
                text_field = "text"

            def tokenize_with_stride(examples):
                tokenized_inputs = tokenizer(
                    examples[text_field],
                    return_overflowing_tokens=True,
                    stride=1,
                    max_length=seq_length,
                    truncation=True,
                    return_special_tokens_mask=True
                )
                return tokenized_inputs

            if not stream:
                # NEW: Update cache path check for genomics
                if dataset_name == "openwebtext2":
                    token_cache_path = "/mfs1/datasets/pile/openwebtext2/owt2_128"
                elif dataset_name == "opengenome2":
                    token_cache_path = f"{cache_dir}/opengenome2_tokenized_128"
                else:
                    token_cache_path = None

                if token_cache_path and os.path.exists(token_cache_path):
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
                    if token_cache_path:
                        print("Saving tokenized dataset to", token_cache_path)
                        tokenized_dataset.save_to_disk(token_cache_path)
                        print("Tokenized dataset saved successfully.")

            else:
                tokenized_dataset = split_dataset.map(
                    tokenize_with_stride,
                    batched=True,
                    remove_columns=[text_field])  # Use correct field name for genomics

            if not stream:
                print("Dataset features:", tokenized_dataset.features)
                print("Number of examples:", len(tokenized_dataset))
            else:
                print("Streaming dataset tokenized (processing on-the-fly)")

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

        def _save(self, output_dir=None, state_dict=None):
            """Override _save to handle weight-tied models (embedding <-> lm_head)"""
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            # Use safe_serialization=False for weight-tied models
            self.model.save_pretrained(output_dir, safe_serialization=False)

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # Save training args
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))



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

    # NEW: Add tokenizer path for genomics data
    if nn_dict["data"] == "lm1b":
        tokenizer_path = "datasets/tokenizers/bpe_tokenizer_lm1b_vocab3000"
    elif nn_dict["data"] == "wikitext-2-v1":
        tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-2-v1_vocab3000"
    elif nn_dict["data"] == "wikitext-103-v1":
        tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-103-v1_vocab3000"
    elif nn_dict["data"] in ["openwebtext2", "openwebtext2_stream"]:
        tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-103-v1_vocab3000"
    elif nn_dict["data"] in ["opengenome2", "opengenome2_stream"]:
        # DNA single-nucleotide tokenizer (Evo2-style): A, C, G, T, N + special tokens (vocab=9)
        tokenizer_path = "datasets/tokenizers/dna_single_nucleotide"
    elif nn_dict["data"] == "opengenome2_local":
        # Binary format uses simple vocab: A=0, C=1, G=2, T=3, N=4 (vocab=5)
        tokenizer_path = None  # No tokenizer needed - data is pre-tokenized
    else:
        raise ValueError(f"Unsupported dataset: {nn_dict['data']}")

    # Handle tokenizer and data collator
    if nn_dict["data"] == "opengenome2_local":
        # No tokenizer needed for binary data - create a simple collator
        tokenizer = None
        vocab_size_override = 5  # A, C, G, T, N

        def simple_collator(batch):
            """Simple collator for pre-tokenized binary data"""
            input_ids = torch.stack([item["input_ids"] for item in batch])
            # Labels are same as input_ids, shifted by the model during loss computation
            return {"input_ids": input_ids, "labels": input_ids.clone()}

        data_collator = simple_collator
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab_size_override = None

        data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                return_tensors="pt"
            )

    # Get hybrid_pattern from nn_dict, default to "alternate" for StripedHyena-style
    hybrid_pattern = nn_dict.get("hybrid_pattern", "alternate")

    model, config, actual_nonembed_params, actual_params = build_model_with_predicted_config(
            target_param_count=target_size,
            tokenizer_path=tokenizer_path,
            save_dir=path,
            model_family=nn_dict["model"],
            reference_configs=Pythia_configs,
            hybrid_pattern=hybrid_pattern,
            vocab_size_override=vocab_size_override  # Use 5 for opengenome2_local
        )

    df = pd.DataFrame({'actual_N': [actual_params]})
    df.to_csv(os.path.join(out_path, 'actual_N_df.csv'), index=False)


    print(f"Model with {target_size/1e9:.1f}B parameters built successfully.")
    print(f"Model saved to ./llama_model_{int(target_size/1e9)}B")
    print("requested parameter count: ", target_size)
    print("actual non-embedding parameter count: ", actual_nonembed_params)
    print("actual parameter count: ", actual_params)


    # Set sequence length based on dataset type
    if nn_dict["data"] in ["opengenome2", "opengenome2_stream", "opengenome2_local"]:
        seq_length = 2048  # Longer context for genomics data
    else:
        seq_length = 128   # Default for text datasets (owt, wikitext, etc.)

    print(f"Using sequence length: {seq_length}")

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

    # CHANGE: Updated gradient accumulation logic accounting for multi-GPU and sequence length
    # For genomics data (seq_length=2048), memory usage is ~16x higher than text (seq_length=128)
    # Scale gradient accumulation accordingly to avoid OOM
    target_batch_size = h_dict["B"]
    print(f"Target global batch size: {target_batch_size}")

    # Sequence length factor: longer sequences need more gradient accumulation
    # Reference is seq_length=128 for text data
    seq_length_factor = seq_length / 128.0
    print(f"Sequence length factor for grad accumulation: {seq_length_factor:.1f}x (seq_length={seq_length})")

    if world_size == 1:
        num_grad_accu1 = h_dict['N']/1_000_000 * h_dict['B']/(128 *256)/4 * seq_length_factor
        num_grad_accu2 = h_dict['B']/3072 * seq_length_factor
        num_grad_accu = max(num_grad_accu1, num_grad_accu2)
    else:
        num_grad_accu1 = h_dict['N']/1_000_000 * h_dict['B']/(128 *256 * world_size)/4 * seq_length_factor
        num_grad_accu2 = h_dict['B']/(3072 * world_size) * seq_length_factor
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

    # Adjust dataloader settings based on sequence length
    # H100/A100 have plenty of memory, so we can use more workers even for long sequences
    if seq_length >= 8192:
        dataloader_num_workers = 4
        dataloader_prefetch_factor = 2
    elif seq_length >= 2048:
        dataloader_num_workers = 8
        dataloader_prefetch_factor = 4
    else:
        dataloader_num_workers = 12
        dataloader_prefetch_factor = 4

    # Reduce eval batch size for long sequences to avoid OOM
    eval_batch_size = max(1, BS // 2) if seq_length >= 2048 else BS

    # Modified TrainingArguments
    # Use bf16 for striped_hyena (flash kernels require consistent bf16), fp16 for others
    use_bf16 = nn_dict["model"] == "striped_hyena"
    training_args_dict = {
        "output_dir": path,
        "per_device_train_batch_size": BS,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": grad_accumulation_steps,
        "eval_strategy": "steps",
        "eval_steps": 1000,  # Evaluate every 1000 steps
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": True,
        "dataloader_prefetch_factor": dataloader_prefetch_factor,
        "max_steps": h_dict["K"],
        "save_strategy": "steps",
        "save_steps": 10000,
        "save_total_limit": 5,
        "logging_steps": 1000,
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "push_to_hub": False,
    }

    print(f"DataLoader: num_workers={dataloader_num_workers}, prefetch_factor={dataloader_prefetch_factor}")
    print(f"Batch sizes: train={BS}, eval={eval_batch_size}")

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

    # CHANGE: REMOVED the debugging raise statement
    # raise ValueError("stop here, checkpoint is: ", latest_checkpoint_path)  # THIS LINE IS DELETED

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
 # Single GPU: python hf_utils_train_model_general.py
 # Multi-GPU:  torchrun --nproc_per_node=2 hf_utils_train_model_general.py
