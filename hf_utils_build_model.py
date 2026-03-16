
import pandas as pd
import json
import sys

Pythia_configs = pd.DataFrame({
    'model': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b'],
    'hidden_size': [512, 768, 1024, 2048, 2048, 2560, 4096, 5120],
    'num_hidden_layers': [6, 12, 24, 16, 24, 32, 32, 36],
    'num_attention_heads': [8, 12, 16, 8, 16, 32, 32, 40],
    'intermediate_size': [2048, 3072, 4096, 8192, 8192, 10240, 16384, 20480],  # 4x hidden_size for all models
    'total_parameters_count': [18_915_328, 85_056_000, 302_311_424, 805_736_448, 1_208_602_624, 2_517_652_480, 6_444_163_072, 11_327_027_200]  # Total parameters count
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

# SSM configurations based on the table in configurations.png
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
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM
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
    # Extract the independent variable (total parameter count)
    X = np.log(df['total_parameters_count'])
    X_billions = df['total_parameters_count'] / 1e9  # For plotting
    
    # Define the columns we want to predict
    target_columns = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size']
    
    # Initialize a dictionary to store regression coefficients for each target
    models = {}
    
    # Prepare plotting if a filename is provided
    if plot_filename:
        # Setup the figure with a 2x2 grid
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(2, 2)
        plt.suptitle(f'Log-Log Regression: Model Architecture vs. Parameter Count\nModel family: {df["model"].iloc[0].split("-")[0]}', 
                    fontsize=16)
    
    # Fit a linear regression for each target variable
    for i, col in enumerate(target_columns):
        y = np.log(df[col])
        y_actual = df[col]  # For plotting
        
        # Perform linear regression using scipy.stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        models[col] = (intercept, slope)
        
        print(f"Log-log regression for {col}:")
        print(f"Intercept: {intercept:.4f}, Coefficient: {slope:.4f}, R²: {r_value**2:.4f}")
        # save the above text to a file
        with open(f'regression_{col}.txt', 'w') as f:
            f.write(f"Log-log regression for {col}:\n")
            f.write(f"Intercept: {intercept:.4f}, Coefficient: {slope:.4f}, R²: {r_value**2:.4f}\n")
            #raise ValueError("Stop after saving regression results.")
        # Create visualization if requested
        if plot_filename:
            ax = plt.subplot(gs[i // 2, i % 2])
            
            # Plot the actual data points
            ax.scatter(X_billions, y_actual, color='blue', s=100, alpha=0.7, label='Actual values')
            
            # Generate points for the regression line
            x_range = np.logspace(np.log10(min(X_billions)*0.7), np.log10(max(X_billions)*1.3), 100)
            y_pred = np.exp(intercept + slope * np.log(x_range * 1e9))
            
            # Plot the regression line
            ax.plot(x_range, y_pred, 'r-', linewidth=2, label=f'Regression (R²={r_value**2:.3f})')
            
            # Add equation to the plot
            eqn_text = f"ln(y) = {intercept:.2f} + {slope:.2f} × ln(x)"
            ax.text(0.05, 0.95, eqn_text, transform=ax.transAxes, 
                   fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Make regression line visible by using log scale
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Set labels and title
            ax.set_xlabel('Parameter Count (billions)', fontsize=12)
            ax.set_ylabel(col.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{col.replace("_", " ").title()} vs Model Size', fontsize=14)
            
            # Add model names as annotations
            for j, model in enumerate(df['model']):
                ax.annotate(model, (X_billions[j], y_actual[j]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Add gridlines and legend
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.legend(loc='best')
    
    # Save the figure if requested
    if plot_filename:
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the title
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")
        plt.close()
    
    # Create the prediction function
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
        
        # First predict attention heads and hidden size
        for col in ['hidden_size']:
            intercept, slope = models[col]
            predicted_value = np.exp(intercept + slope * log_params)
            result[col] = int(round(predicted_value/16) * 16)

        result["num_attention_heads"] = int(round(result['hidden_size'] / 16))

        for col in ['num_hidden_layers', 'intermediate_size']:
            intercept, slope = models[col]
            predicted_value = np.exp(intercept + slope * log_params)
            
            if col == 'num_hidden_layers':
                # Round to nearest integer
                result[col] = int(round(predicted_value))

                if result[col] < 2:
                    result[col] = 2
            
            elif col == 'intermediate_size':
                result[col] = int(round(result['hidden_size'] * 4 ))

        return result

    return predict_config

# Function for SSM config prediction
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

        # Derive n_heads from d_model and kv_size (following SSM_configs pattern)
        # Pattern: n_heads = d_model / kv_size
        d_model = result['d_model']
        kv_size = result['kv_size']

        # Calculate n_heads from the pattern
        n_heads = d_model // kv_size
        if n_heads < 1:
            n_heads = 1

        # Ensure GQA compatibility: n_heads % (n_heads // proj_groups) == 0
        # For proj_groups=4, valid n_heads are: 4, 5, 6, 7, 8, 10, 12, 15, 16, 20, 24, ...
        # Simplest approach: round to nearest multiple of 4 (always valid for GQA)
        proj_groups = 4

        def is_gqa_compatible(n, proj_groups):
            """Check if n_heads is compatible with GQA"""
            if n < proj_groups:
                return True  # When n < proj_groups, num_heads_kv = 0 which uses n directly
            num_heads_kv = n // proj_groups
            return num_heads_kv > 0 and n % num_heads_kv == 0

        if not is_gqa_compatible(n_heads, proj_groups):
            # Find nearest valid n_heads
            # Search in both directions for a valid value
            for delta in range(1, n_heads + 10):
                if n_heads - delta >= 1 and is_gqa_compatible(n_heads - delta, proj_groups):
                    n_heads = n_heads - delta
                    break
                if is_gqa_compatible(n_heads + delta, proj_groups):
                    n_heads = n_heads + delta
                    break

        result['n_heads'] = n_heads

        # Adjust d_model to match: d_model = n_heads * kv_size
        # This ensures the pattern is maintained
        result['d_model'] = n_heads * kv_size

        # Predict n_layer (number of layers)
        intercept, slope = models['n_layer']
        predicted_n_layer = np.exp(intercept + slope * log_params)
        result['n_layer'] = int(round(predicted_n_layer))
        if result['n_layer'] < 2:
            result['n_layer'] = 2

        return result

    return predict_ssm_config

def build_model_with_predicted_config(
        target_param_count,
        tokenizer_path,
        model_family = "llama",
        reference_configs = Pythia_configs,
        save_dir=None,
        hybrid_pattern="alternate",
        vocab_size_override=None):  # Use this to override tokenizer vocab (e.g., 5 for binary genome data)
    """
    Build a model with predicted configuration for a given parameter count.

    Args:
        target_param_count: Target parameter count (e.g., 1e9 for 1B)
        tokenizer_path: Path to the tokenizer
        model_family: Model family ('llama', 'pythia', 'striped_hyena')
        reference_configs: Reference configurations for regression
        save_dir: Directory to save the model (optional)
        hybrid_pattern: Layer pattern for hybrid models ('alternate', 'every_4', 'every_8', etc.)
        vocab_size_override: Override vocab size (e.g., 5 for binary genome data)

    Returns:
        Model, configuration, and actual parameter count
    """

    if model_family == "llama":
        #raise ValueError("Llama model family is not supported yet. Use 'pythia' instead.")
        print(f"\nBuilding a LLaMA model with target size: {target_param_count/1e9:.1f}B parameters")
        config_creation_function = LlamaConfig
        model_creation_function = LlamaForCausalLM
    elif model_family == "pythia":
        print(f"\nBuilding a Pythia model with target size: {target_param_count/1e9:.1f}B parameters")
        config_creation_function = GPTNeoXConfig
        model_creation_function = GPTNeoXForCausalLM
    # Add support for Together AI StripedHyena from HuggingFace
    elif model_family == "striped_hyena":
        print(f"\nBuilding a Together AI StripedHyena model with target size: {target_param_count/1e6:.1f}M parameters")
        from transformers import AutoConfig, AutoModelForCausalLM
        # Use the Together StripedHyena architecture with custom config
        config_creation_function = "striped_hyena_config"  # Special marker for custom handling
        model_creation_function = "striped_hyena_model"    # Special marker for custom handling
    else:
        raise ValueError("Invalid model class. Choose 'llama', 'pythia', 'striped_hyena'.")

    # Use SSM predictor for SSM/Hyena/StripedHyena models (same architecture config table)
    if model_family in ["striped_hyena"]:
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

        # Together AI StripedHyena from HuggingFace
        if model_family == "striped_hyena":
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

        elif model_family == "llama":
            # LLaMA with SwiGLU FFN - more accurate parameter estimation
            hidden_size_config = config_dict['hidden_size']
            num_layers_config = config_dict['num_hidden_layers']
            intermediate_size_config = config_dict['intermediate_size']
            # Per layer:
            # - Attention: 4 × hidden_size² (Q, K, V, O projections)
            # - FFN (SwiGLU): 3 × hidden_size × intermediate_size (gate, up, down)
            # - RMSNorm: 2 × hidden_size (pre-attention, pre-FFN)
            attention_params = 4 * hidden_size_config ** 2
            ffn_params = 3 * hidden_size_config * intermediate_size_config
            norm_params = 2 * hidden_size_config
            # Final RMSNorm: hidden_size
            simplified_non_embedding_params_cnt = num_layers_config * (attention_params + ffn_params + norm_params) + hidden_size_config
            embedding_params_cnt = vocab_size * hidden_size_config * 2
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

        # Create Together AI StripedHyena config from HuggingFace
        if model_family == "striped_hyena":
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
        if model_family == "striped_hyena":
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
    if model_family not in ["striped_hyena"]:
        attention_params_per_layer = 4 * (config.hidden_size ** 2)
        ffn_params_per_layer = 2 * config.hidden_size * config.intermediate_size

        total_attention_params = attention_params_per_layer * config.num_hidden_layers
        total_ffn_params = ffn_params_per_layer * config.num_hidden_layers

        print(f" - Attention layers: {total_attention_params:,} params")
        print(f" - Feed-forward layers: {total_ffn_params:,} params")
        print(f" - Other (norm, etc.): {param_count_actual - embedding_params - total_attention_params - total_ffn_params:,} params")
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



# Example usage:
if __name__ == "__main__":
    
        if len(sys.argv) > 1:
            temp_path = str(sys.argv[1])
        else:
            temp_path = 'outputs/nn_hf/temp/'  # default value
        
        print(f"Using path: {temp_path}")
        
        # Define cache directory here for easy changing
        datasets_cache_dir = "./datasets"
        
        hf_temp_folder = temp_path
        input_path = hf_temp_folder + 'input/'
        if not os.path.exists(input_path):
            raise ValueError(f"Input folder {input_path} does not exist.")
        
        # read nn_dict and h_dict
        with open(input_path + 'nn_dict.json', 'r') as f:
            nn_dict = json.load(f)
            # example: {"data": "wikitext-103-v1", "loss": "condcrossent", "model": "pythia"}
        with open(input_path + 'h_dict.json', 'r') as f:
            h_dict = json.load(f)
            # example: {"N": 1000000, "B": 24, "K": 31901, "lr": 1.2156143558093906, "end_lr": 0.1, "momentum": 0.0, "lr_schedule": "step", "optimizer": "sgd", "step_decay_schedule": {"decay_at": [0.5], "decay_amt": [1]}}
        
        # read path
        with open(input_path + 'path.txt', 'r') as f:
            path = f.read()
            # example: "outputs/llama_model/0_test_out"
        
        # Make sure cache directory exists
        os.makedirs(datasets_cache_dir, exist_ok=True)
        # create the path directory if it does not exist
        os.makedirs(path, exist_ok=True)
        # create a subdirectory under path with folder name out
        #out_path = os.path.join(path, 'output')
        out_path = hf_temp_folder + 'output/'
        os.makedirs(out_path, exist_ok=True)
        

        target_size = h_dict["N"]

        # Determine tokenizer path and vocab_size_override based on dataset
        if nn_dict["data"] == "lm1b":
            tokenizer_path = "datasets/tokenizers/bpe_tokenizer_lm1b_vocab3000"
            vocab_size_override = None
        elif nn_dict["data"] == "wikitext-2-v1":
            tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-2-v1_vocab3000"
            vocab_size_override = None
        elif nn_dict["data"] == "wikitext-103-v1":
            tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-103-v1_vocab3000"
            vocab_size_override = None
        elif nn_dict["data"] in ["openwebtext2", "openwebtext2_stream"]:
            tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-103-v1_vocab3000"
            vocab_size_override = None
        elif nn_dict["data"] in ["opengenome2", "opengenome2_stream"]:
            # DNA single-nucleotide tokenizer (Evo2-style): A, C, G, T, N + special tokens (vocab=9)
            tokenizer_path = "datasets/tokenizers/dna_single_nucleotide"
            vocab_size_override = None
        elif nn_dict["data"] == "opengenome2_local":
            # Binary format uses simple vocab: A=0, C=1, G=2, T=3, N=4 (vocab=5)
            tokenizer_path = None  # No tokenizer needed - data is pre-tokenized
            vocab_size_override = 5
        else:
            raise ValueError(f"Unknown data type {nn_dict['data']}. Please specify a valid tokenizer path.")

        # Only load tokenizer if path is specified
        if tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = None

        # Get hybrid_pattern from nn_dict, default to "alternate" for StripedHyena-style
        hybrid_pattern = nn_dict.get("hybrid_pattern", "alternate")

        model, config, actual_nonembed_params, actual_params = build_model_with_predicted_config(
                target_param_count=target_size,
                tokenizer_path=tokenizer_path,
                save_dir=path,
                model_family=nn_dict["model"],
                reference_configs=Pythia_configs,
                hybrid_pattern=hybrid_pattern,
                vocab_size_override=vocab_size_override
            )
        
        def extract_model_info(model):
            """
            Extracts key architecture and size info from a HuggingFace/PyTorch model.

            Returns a dict with:
            - num_hidden_layers      : int or None
            - hidden_size            : int or None
            - num_attention_heads    : int or None
            - trainable_params       : int
            """
            # Get config values (if present)
            cfg = getattr(model, "config", None)
            # Handle striped_hyena which uses num_layers instead of num_hidden_layers
            n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
            hidden_size = getattr(cfg, "hidden_size", None)
            n_heads = getattr(cfg, "num_attention_heads", None)

            # Parameter counts
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            return {
                "num_hidden_layers":     n_layers,
                "hidden_size":           hidden_size,
                "num_attention_heads":   n_heads,
                "trainable_params":      trainable,
            }
        
        # save a dataframe with one row and one column, 
        # the name of the column is actual_N and the value is actual_nonembed_params
        df = pd.DataFrame({'actual_N': [actual_params]})
        # save the dataframe to a csv file in the out_path directory
        df.to_csv(os.path.join(out_path, 'actual_N_df.csv'), index=False)

        # return a dataframe with the model info
        model_info = extract_model_info(model)
        model_info_df = pd.DataFrame([model_info])
        # save the model info to a csv file in the out_path directory
        model_info_df.to_csv(os.path.join(out_path, 'model_info_df.csv'), index=False)

        
        print(f"Model with {target_size/1e9:.1f}B parameters built successfully.")
        print(f"Model saved to ./llama_model_{int(target_size/1e9)}B")
        print("requested parameter count: ", target_size)
        print("actual non-embedding parameter count: ", actual_nonembed_params)
        print("actual parameter count: ", actual_params)
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None