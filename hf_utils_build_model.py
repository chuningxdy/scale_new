
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

def build_model_with_predicted_config(
        target_param_count, 
        tokenizer_path, 
        model_family = "llama",
        reference_configs = Pythia_configs,
        save_dir=None):
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
    else: 
        raise ValueError("Invalid model class. Choose 'llama' or 'pythia'.")
    
    # Get the prediction function
    predictor = func_config_from_params_count(reference_configs) 

    
    # Load custom tokenizer
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

        # 12* num_layers * hidden_size^2
        simplified_non_embedding_params_cnt = 12 * num_layers_config * hidden_size_config ** 2
        embedding_params_cnt = vocab_size * hidden_size_config * 2
        simplified_total_params_cnt = simplified_non_embedding_params_cnt + embedding_params_cnt

        return simplified_non_embedding_params_cnt, simplified_total_params_cnt
    

        

    # try ratios between 0.5 and 1.5 of param_count
   # random.seed(6)
   # random_numbers = random.sample(range(10, 150),30)
    candidate_param_counts = [target_param_count * (x / 100) for x in range(10, 151, 5)]
    # integer
    candidate_param_counts = [int(x) for x in candidate_param_counts]

    candidate_param_counts_simplified_outputs = {}
    for param_count in candidate_param_counts:
        # try simplied param count and check if the result is close to target_param_count
        config_dict = predictor(param_count)
        non_embed_params_simplified, total_params_simplified = simplified_param_count(config_dict, vocab_size)
        candidate_param_counts_simplified_outputs[param_count] = (non_embed_params_simplified, total_params_simplified)

    # in the dictionary, find the param_count that gives the closest total_params_simplified to target_param_count 
    # measure the differnce as a abs(ratio - 1):
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
    #raise ValueError("Stopping here to check best_param_count:", best_param_count,
    #                    candidate_param_counts_simplified_outputs[best_param_count],
    #                    best_ratio_diff)
    print(f"\nSelected candidate parameter count for prediction: {best_param_count:,} ({best_param_count/1e9:.4f}B) with simplified total params closest to target")
    print(f" - Ratios are (param_count, ratio to target): {ratios_to_target} ...")
    #for param_count in candidate_param_counts:
    if True:
        # Get predicted configuration
        config_dict = predictor(best_param_count)
        print("\nPredicted configuration:")
        for key, value in config_dict.items():
            #if key != 'total_parameters_count':
                print(f" - {key}: {value}")
        # Create LLaMA config
        config = config_creation_function(
            vocab_size=vocab_size,
            hidden_size=config_dict['hidden_size'],
            intermediate_size=config_dict['intermediate_size'],
            num_hidden_layers=config_dict['num_hidden_layers'],
            num_attention_heads=config_dict['num_attention_heads']
        )
    
        # Create the model
        print("\nInitializing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # If memory is a concern, use low precision or CPU
        #dtype = torch.float32 # if torch.cuda.is_available() else torch.float32
        
        # Create model
        try:
            # start a new model, if model exists, remove it
            if 'model' in locals():
                del model
            model = model_creation_function(config)
        except Exception as e:
            raise ValueError("Model initialization failed. Please check the configuration.")
        
        # Count parameters
        param_count_actual = sum(p.numel() for p in model.parameters())
       # print(f"\nActual parameter count: {param_count_actual:,} ({param_count_actual/1e9:.4f}B)")
        

        
        # Detailed parameter breakdown
      #  print("\nParameter breakdown:")
        embedding_params = config.vocab_size * config.hidden_size * 2
       # print(f" - Input embeddings: {embedding_params:,} params")
        non_embedding_params_actual = param_count_actual - embedding_params
            # Calculate prediction accuracy
        accuracy = (param_count_actual / target_param_count) * 100
        print(f"trying ratio: {best_param_count/target_param_count:.2f}")
        print(f"Prediction accuracy: {accuracy:.2f}% of target")

        # if accuracy is not between 70 and 150, try again
       # if accuracy < 75 or accuracy > 130:
       #     print(f"Accuracy {accuracy:.2f}% is outside the acceptable range. Trying again...")
       #     continue
       # else:
       #     print(f"Accuracy {accuracy:.2f}% is within the acceptable range. Proceeding...")
        #    break
    
    # if accuracy is not between 70 and 150 after all attempts, raise an error
    if accuracy < 75 or accuracy > 130:
        raise ValueError(f"Final prediction accuracy {accuracy:.2f}% is outside the acceptable range.")

    attention_params_per_layer = 4 * (config.hidden_size ** 2)  # Q, K, V, and output projections
    ffn_params_per_layer = 2 * config.hidden_size * config.intermediate_size  # Up and down projections
    
    total_attention_params = attention_params_per_layer * config.num_hidden_layers
    total_ffn_params = ffn_params_per_layer * config.num_hidden_layers
    
    print(f" - Attention layers: {total_attention_params:,} params")
    print(f" - Feed-forward layers: {total_ffn_params:,} params")
    print(f" - Other (norm, etc.): {param_count_actual - embedding_params - total_attention_params - total_ffn_params:,} params")
    print(f" - Non-embedding params: {param_count_actual - embedding_params:,} params")
    print(f" - Total params: {param_count_actual:,} params ({param_count_actual})")
    # Save the model if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        if os.path.exists(tokenizer_path):
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
        

        target_size = h_dict["N"] # 1e6 #, 3e9, 5e9]

        if nn_dict["data"] == "lm1b":
            tokenizer_path = "datasets/tokenizers/bpe_tokenizer_lm1b_vocab3000"
        elif nn_dict["data"] == "wikitext-2-v1":
            tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-2-v1_vocab3000"
        elif nn_dict["data"] == "wikitext-103-v1":
            tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-103-v1_vocab3000"
        elif nn_dict["data"] in ["openwebtext2", "openwebtext2_stream"]:
            tokenizer_path = "datasets/tokenizers/bpe_tokenizer_wikitext-103-v1_vocab3000"
        else:
            raise ValueError(f"Unknown data type {nn_dict['data']}. Please specify a valid tokenizer path.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


        model, config, actual_nonembed_params, actual_params = build_model_with_predicted_config(
                target_param_count=target_size,
                tokenizer_path=tokenizer_path,
                save_dir= path,
                model_family= nn_dict["model"], 
                reference_configs= Pythia_configs
            )
        
        def extract_model_info(model):
            """
            Extracts key architecture and size info from a HuggingFace/PyTorch model.

            Returns a dict with:
            - num_hidden_layers      : int or None
            - sequence_length        : int or None
            - hidden_size            : int or None
            - num_attention_heads    : int or None
            - trainable_params       : int
            - non_trainable_params   : int
            - buffer_params          : int
            """
            # 1) Get config values (if present)
            cfg = getattr(model, "config", None)
            n_layers = getattr(cfg, "num_hidden_layers", None)
            hidden_size = getattr(cfg, "hidden_size", None)
            n_heads = getattr(cfg, "num_attention_heads", None)

            # 3) Parameter counts
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
          #  non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)

            # 4) Registered buffers (e.g. BatchNorm stats)
          #  buffer_count = sum(b.numel() for b in getattr(model, "buffers", lambda: [])())

            return {
                "num_hidden_layers":     n_layers,
                "hidden_size":           hidden_size,
                "num_attention_heads":   n_heads,
                "trainable_params":      trainable,
               # "non_trainable_params":  non_trainable,
               # "buffer_params":         buffer_count,
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