
import os
os.environ["HF_HOME"] = "/mfs1/datasets/pile/huggingface"
# or, if you prefer the finer-grained knobs:
os.environ["HF_HUB_CACHE"]     = "/mfs1/datasets/pile/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/mfs1/datasets/pile/huggingface/datasets"
# print the environment variables
print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])
print("HF_DATASETS_CACHE:", os.environ["HF_DATASETS_CACHE"])
#raise ValueError("Stopping here to check environment variables")

import pandas as pd
import json
import sys
import time

Pythia_configs = pd.DataFrame({
    'model': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b'],
    'hidden_size': [512, 768, 1024, 2048, 2048, 2560, 4096, 5120],
    'num_hidden_layers': [6, 12, 24, 16, 24, 32, 32, 36],
    'num_attention_heads': [8, 12, 16, 8, 16, 32, 32, 40],
    'intermediate_size': [2048, 3072, 4096, 8192, 8192, 10240, 16384, 20480],  # 4x hidden_size for all models
    'non_embed_parameters_count': [18_915_328, 85_056_000, 302_311_424, 805_736_448, 1_208_602_624, 2_517_652_480, 6_444_163_072, 11_327_027_200],  # Total parameters count
    'total_parameters_count': [7e7, 85_056_000, 302_311_424, 805_736_448, 1_208_602_624, 2_517_652_480, 6_444_163_072, 11_327_027_200]  # Total parameters count
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

#from hf_utils_build_model import build_model_with_predicted_config

import pandas as pd


                    
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup



Pythia_configs = pd.DataFrame({
    'model': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b'],
    'hidden_size': [512, 768, 1024, 2048, 2048, 2560, 4096, 5120],
    'num_hidden_layers': [6, 12, 24, 16, 24, 32, 32, 36],
    'num_attention_heads': [8, 12, 16, 8, 16, 32, 32, 40],
    'intermediate_size': [2048, 3072, 4096, 8192, 8192, 10240, 16384, 20480],  # 4x hidden_size for all models
    'total_parameters_count': [18_915_328, 85_056_000, 302_311_424, 805_736_448, 1_208_602_624, 2_517_652_480, 6_444_163_072, 11_327_027_200]  # Total parameters count
})

# If you want to use the entire dataset (all splits combined)
def get_full_dataset(dataset, verbose=True):
    # Get all available splits
    available_splits = list(dataset.keys())
    
    if verbose:
        print(f"Combining all available splits: {available_splits}")
    
    # Start with the first split
    full_dataset = dataset[available_splits[0]]
    
    # Add all other splits
    for split in available_splits[1:]:
        split_dataset = dataset[split]
        full_dataset = concatenate_datasets([full_dataset, split_dataset])
        
    #if verbose:
   #     print(f"Created combined dataset with {len(full_dataset)} examples")
        
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
        

        #print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])
        #print("HF_DATASETS_CACHE:", os.environ["HF_DATASETS_CACHE"])
        #raise ValueError("Stopping here to check environment variables")
        # Load dataset
        if dataset_name == "wikitext-2-v1":
            data_label = "wikitext"
        elif dataset_name == "wikitext-103-v1":
            data_label = "wikitext"
        elif dataset_name == "lm1b":
            data_label = "lm1b"
        elif dataset_name in ["openwebtext2", "openwebtext2_stream"]:
            data_label = "vietgpt/the_pile_openwebtext2"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

       #dataset = load_dataset(data_label, dataset_name, cache_dir=cache_dir)

        # To this:
        if dataset_name == "lm1b":
            dataset = load_dataset(data_label, name=None, cache_dir=cache_dir, trust_remote_code=True)
            split_dataset = get_full_dataset(dataset, verbose=verbose)
            filtered_dataset = split_dataset
            stream = False

        elif dataset_name == "wikitext-2-v1" or dataset_name == "wikitext-103-v1":
            dataset = load_dataset(data_label, name=dataset_name, cache_dir=cache_dir)
            split_dataset = get_full_dataset(dataset, verbose=verbose)
                    # Filter out empty strings before tokenization
            # there are many empty strings in wikitext datasets
            def is_non_empty(example):
                return bool(example["text"].strip())
        
            # Apply the filter
            filtered_dataset = split_dataset.filter(
                is_non_empty,
                desc="Filtering out empty strings"
            )
            stream = False


        elif dataset_name == "openwebtext2":
            dataset = load_dataset(data_label, name=None,
                                   cache_dir='/mfs1/datasets/pile/openwebtext2')
                                   #streaming=True)
            
            # get the train split
            split_dataset = get_full_dataset(dataset, verbose=verbose)
            filtered_dataset = split_dataset
            stream = False #True

        elif dataset_name == "openwebtext2_stream":
            dataset = load_dataset(data_label, name=None,
                                   cache_dir='/mfs1/datasets/pile/openwebtext2',
                                   streaming=True)
            
            # get the train split
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
            # Define preprocessing function            
            def tokenize_with_stride(examples):

                # Tokenize with stride to get overlapping chunks
                tokenized_inputs = tokenizer(
                    examples["text"],
                    return_overflowing_tokens=True,  # This creates multiple sequences per example
                    stride=1,  
                    max_length=seq_length,
                    truncation=True,
                    return_special_tokens_mask=True
                )
            
            # Return all sequences as separate examples
                return tokenized_inputs

            # Step 3: Tokenize the grouped examples (will be cached separately)
            if not stream:
                # raise an error if not openwebtext2
                if dataset_name != "openwebtext2":
                    raise ValueError("Dataset name must be openwebtext2 for this code to work")
                # cache to tokenized_dataset.save_to_disk("/mfs1/datasets/tokenized/owt2_128")
                # if openwebtext2, check if cache exists, if exists, load it
                # otherwise, map and tokenize
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
                        num_proc=12, #12,#4,
                        #remove_columns=["text"],
                        remove_columns=split_dataset.column_names,
                        desc="Tokenizing with stride to create multiple examples"
                    )
                    # cache
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
            def display_example(example_idx):
                example = tokenized_dataset[example_idx]
                
                # Get the input IDs for this example
                input_ids = example['input_ids']
                
                # Decode the tokens back to text
                decoded_text = tokenizer.decode(input_ids)
                
                print(f"Example {example_idx}:")
                print(f"Length: {len(input_ids)} tokens")
                print(f"Text: {decoded_text}")

            # print first 5 examples
            #print("First 5 examples:")
            #for i in range(3):
            #    display_example(i)
            #    print("\n")

        #raise ValueError("Stopping here to check tokenization")
            
        # Special processing for LM1B: group examples before tokenization
        elif dataset_name in ["lm1b"]:
        
            # Step 1: Define a function that groups examples
            # this is because the LM1B dataset is a collection of sentences
            # where each sentence is a separate example, and is typically very short
            # so inorder to fill in the context length, we need to group multiple sentences together
            def group_examples(examples, group_size=50, _cache_version=1):
                """Group multiple examples into one larger example.
                The _cache_version parameter helps invalidate cache when needed.
                """
                all_texts = examples["text"]
                result_texts = []
                
                # Process in batches (each batch will become one new example)
                for i in range(0, len(all_texts), group_size):
                    group = all_texts[i:i+group_size]
                    grouped_text = "\n".join(group)
                    result_texts.append(grouped_text)
                    
                return {"text": result_texts}
            
            # Step 2: Apply the grouping function via map (will be cached)
           # grouped_dataset = split_dataset.map(
          #      group_examples,
          #      batched=True,
          #      batch_size=1000,  # Process 1000 examples at a time (will create 20 grouped examples)
          #      remove_columns=split_dataset.column_names,  # Remove original columns
          #      desc="Grouping sentences into larger examples"
          #  )
            
            def tokenize_with_stride(examples):

                # Tokenize with stride to get overlapping chunks
                tokenized_inputs = tokenizer(
                    examples["text"],
                    return_overflowing_tokens=True,  # This creates multiple sequences per example
                    stride=1,  
                    max_length=seq_length,
                    truncation=True,
                    return_special_tokens_mask=True
                )
            
            # Return all sequences as separate examples
                return tokenized_inputs

            grouped_dataset = split_dataset
            # Step 3: Tokenize the grouped examples (will be cached separately)
            tokenized_dataset = grouped_dataset.map(
                tokenize_with_stride,
                batched=True,
                num_proc=12,#4,
                remove_columns=["text"],
                desc="Tokenizing with stride to create multiple examples"
            )
            
          #  def tokenize_examples(examples):
           #     return tokenizer(
           #         examples["text"], 
          #          truncation=True,
          #          max_length=seq_length,
          #          return_special_tokens_mask=True
          #      )
            
          #  tokenized_dataset = grouped_dataset.map(
          #      tokenize_with_stride,
          #      batched=True,
          #      num_proc=4,
          #      remove_columns=["text"],
           #     desc="Tokenizing grouped examples"
           # )


        # check if the tokenized dataset contains any empty examples
        # if so, print out the text of the example
       # for i, example in enumerate(tokenized_dataset):
       #     if 'input_ids' in example and len(example['input_ids']) <= 0:
       #         print(f"Example {i}: {example}")
       #         print(f"Example {i}: {example['text']}")    
      #          raise ValueError("Found invalid example:", i, ":", example, "\n", example['text'])

        # Use the specified test_size
        # Randomly select indices for the test set
        if not stream:
            test_size_frac = test_size / len(tokenized_dataset)
            split_dataset = tokenized_dataset.train_test_split(test_size=test_size_frac, seed=42)  

            train_dataset = split_dataset["train"]
            test_dataset = split_dataset["test"]
            #test_dataset = tokenized_dataset.select(range(min(test_size, len(tokenized_dataset))))
            if len(tokenized_dataset) > test_size:
                    train_dataset = tokenized_dataset.select(range(test_size, len(tokenized_dataset)))
            else:
                    train_dataset = tokenized_dataset
            
            if verbose:
                print(f"Prepared datasets:")
                print(f"- Training set: {len(train_dataset)} sequences")
                print(f"- Test set: {len(test_dataset)} sequences")
           # raise ValueError("Stopping here to check keys of tokenized_dataset")
        else:
            # print the keys of the tokenized_dataset
            print("Keys of tokenized_dataset:", tokenized_dataset.keys())
            #raise ValueError("Stopping here to check keys of tokenized_dataset")
            test_dataset = tokenized_dataset["train"].take(test_size)
            train_dataset = tokenized_dataset["train"].skip(test_size)
            if verbose:
                print(f"Prepared datasets: stream train, test")
                #print(f"- Training set: {len(train_dataset)} sequences")
                #print(f"- Test set: {len(test_dataset)} sequences")

        return train_dataset, test_dataset





class CustomTrainer(Trainer):
        
        def __init__(self, 
                    momentum: float = 0.9,
                    initial_learning_rate: float = 5e-5,
                    lr_scheduler_type: str = "linear",
                    step_decay_schedule_dict: dict = None,
                    optimizer_type: str = "sgd",
                    **kwargs):
            # Store custom parameters as instance variables
            self.momentum = momentum
            self.initial_learning_rate = initial_learning_rate
            self.lr_scheduler_type = lr_scheduler_type
            self.step_decay_schedule_dict = step_decay_schedule_dict or {}
            self.optimizer_type = optimizer_type

                    # Initialize a list to store eval losses
            self.eval_losses = []
            self.eval_steps = []
            
            # Call parent class constructor with the standard Trainer arguments
            super().__init__(**kwargs)
            

    
        def create_optimizer_and_scheduler(self, 
                                           num_training_steps: int):
                """
                Setup the optimizer and learning rate scheduler.
                """
                # if self.optimizer is None:
                # Create SGD with momentum
                if self.optimizer_type == "sgd":
                    self.optimizer = SGD(
                        self.model.parameters(),
                        lr=self.initial_learning_rate,              # Initial learning rate
                        momentum=self.momentum,    # Momentum factor
                        weight_decay=0  # Weight decay (L2 penalty)
                    )
                elif self.optimizer_type == "adamw":
                    # Create AdamW optimizer
                    self.optimizer = AdamW(
                        self.model.parameters(),
                        lr=self.initial_learning_rate,  # Initial learning rate
                        weight_decay= 0 #self.weight_decay  # Weight decay (L2 penalty)
                    )
                
                # check if lr_scheduler_type is in ["constant","step"]
                if self.lr_scheduler_type == "constant":
                    #raise ValueError("lr_scheduler_type 'constant' is not supported. Use 'step' instead.")
                    # Use a constant learning rate
                    #self.lr_scheduler = None
                    self.lr_scheduler = get_constant_schedule(self.optimizer)
                elif self.lr_scheduler_type == "cosine":
                    self.lr_scheduler = get_cosine_schedule_with_warmup(
                        self.optimizer, 
                        num_warmup_steps = int(num_training_steps * 0.01), 
                        num_training_steps = num_training_steps)
                elif self.lr_scheduler_type == "step":
                     # check if step_decay_schedule_dict is provided,
                    # if not, raise an error
                    if self.step_decay_schedule_dict is None:
                        raise ValueError("step_decay_schedule_dict must be provided for step decay schedule")
                    # use step decay schedule dict to create a step decay schedule
                    # the dict contains two items
                    # decay_at and decay_amt
                    # decay_at is the % of step at which to decay the learning rate
                    # decay_amt is the amount to decay the learning rate by at that step
                    # for example, if decay_at = [0.5, 0.8] and decay_amt = [0.1,0.01]
                    # the learning rate will be decayed to 10% at 50% of the steps
                    # and further to 10% * 1% at 80% of the steps
                    
                    # Calculate actual step numbers for milestones
                    milestones = [int(decay_at * num_training_steps) for decay_at in self.step_decay_schedule_dict['decay_at']]
                
                    # Create lr values for each segment
                    lr_values = [1.0]  
                    for decay_amt in self.step_decay_schedule_dict['decay_amt']:
                        lr_values.append(lr_values[-1] * decay_amt)  # Apply decay to previous rate

                    def lr_lambda(current_step: int):
                        # Find the segment for the current step
                        for i, milestone in enumerate(milestones):
                            if current_step < milestone:
                                return lr_values[i]
                        return lr_values[-1]
                    
                
                    # Create the scheduler
                    self.lr_scheduler = LambdaLR(
                        self.optimizer,
                        lr_lambda ,
                        last_epoch=-1
                    )

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            # Call the parent's evaluate method
            output = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
            
            # Store the loss and current step
            self.eval_losses.append(output["eval_loss"])
            self.eval_steps.append(self.state.global_step)
            
            # Save to CSV after each evaluation
            self.save_eval_metrics()
            
            return output
        
        def save_eval_metrics(self):
            """Save evaluation metrics to a CSV file"""
            
            # Create a DataFrame
            metrics_df = pd.DataFrame({
                'ckpt': self.eval_steps,
                'loss': self.eval_losses
            })
            
            # Save to CSV
            # remove duplicates from the dataframe
            metrics_df = metrics_df.drop_duplicates(subset=['ckpt'])
            metrics_df.to_csv(f"{self.args.output_dir}/loss_curve_df.csv", index=False)
            
            # Optionally, plot the metrics
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
            # Perform evaluation at step 0
            print("Performing initial evaluation at step 0:")
            initial_metrics = self.evaluate()
            print(f"Initial eval loss: {initial_metrics['eval_loss']:.4f}")
            
            # Continue with regular training
            return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, **kwargs)

#if __name__ == "__main__":

def main():


    start_time = time.time()
    if len(sys.argv) > 1:
            temp_path = sys.argv[1]
    else:
            temp_path = 'outputs/nn_hf/may_20/'  # default value
        
    print(f"Using path: {temp_path}") # ---- ***
        
    # Define cache directory here for easy changing
    datasets_cache_dir = "./datasets" # ---- ***
        
    hf_temp_folder = temp_path
    input_path = hf_temp_folder + 'input/' 
    if not os.path.exists(input_path):
        raise ValueError(f"Input folder {input_path} does not exist.")
    
    # read nn_dict and h_dict
    with open(input_path + 'nn_dict.json', 'r') as f:
        nn_dict = json.load(f) # --- ***
        # example: {"data": "wikitext-103-v1", "loss": "condcrossent", "model": "pythia"}
    with open(input_path + 'h_dict.json', 'r') as f:
        h_dict = json.load(f) # --- ***

        # example: {"N": 1000000, "B": 24, "K": 31901, "lr": 1.2156143558093906, "end_lr": 0.1, "momentum": 0.0, "lr_schedule": "step", "optimizer": "sgd", "step_decay_schedule": {"decay_at": [0.5], "decay_amt": [1]}}
    
    # read path
    with open(input_path + 'path.txt', 'r') as f:
        path = f.read() # --- ***
        # example: "outputs/llama_model/0_test_out"
        # this is where to save files from training
    
    # Make sure cache directory exists
    os.makedirs(datasets_cache_dir, exist_ok=True)
    # create the path directory if it does not exist
    os.makedirs(path, exist_ok=True)
    # create a subdirectory under path with folder name out
    #out_path = os.path.join(path, 'output')
    out_path = hf_temp_folder + 'output/'
    os.makedirs(out_path, exist_ok=True)
    # this is where to replace the final output files: actual_N, and loss_curve_df.csv
    

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
        raise ValueError(f"Unsupported dataset: {nn_dict['data']}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            # Create data collator with explicit tensor type
    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt"
        )

    model, config, actual_nonembed_params, actual_params = build_model_with_predicted_config(
            target_param_count=target_size,
            tokenizer_path=tokenizer_path,
            save_dir= path,
            model_family= nn_dict["model"], 
            reference_configs= Pythia_configs
        )
    
    # save a dataframe with one row and one column, 
    # the name of the column is actual_N and the value is actual_nonembed_params
    df = pd.DataFrame({'actual_N': [actual_params]})
    #raise ValueError("Actual params:", actual_params, "actual_nonembed_params:", actual_nonembed_params)
    # save the dataframe to a csv file in the out_path directory
    df.to_csv(os.path.join(out_path, 'actual_N_df.csv'), index=False)

    
    print(f"Model with {target_size/1e9:.1f}B parameters built successfully.")
    print(f"Model saved to ./llama_model_{int(target_size/1e9)}B")
    print("requested parameter count: ", target_size)
    print("actual non-embedding parameter count: ", actual_nonembed_params)
    print("actual parameter count: ", actual_params)


    train_dataset, test_dataset = prepare_datasets(
            tokenizer=tokenizer,
            dataset_name= nn_dict["data"] , #"wikitext-103-v1",  # "lm1b",  # "wikitext-2-v1",
            seq_length= 128,
            cache_dir= datasets_cache_dir,
            test_size=1000,
            verbose=True
        )
    
        # if B > 3072, use gradient accumulation

    num_grad_accu1 = h_dict['N']/1_000_000 * h_dict['B']/(128 *256)
    num_grad_accu2 = h_dict['B']/3072
    num_grad_accu = max(num_grad_accu1, num_grad_accu2)
    if num_grad_accu > 1.0:
        large_BS = h_dict["B"]
        grad_accumulation_steps = int(np.ceil(num_grad_accu))
        BS = int(np.round(large_BS / grad_accumulation_steps))
        print(f"Using gradient accumulation with {grad_accumulation_steps} steps for batch size {h_dict['B']}")
    elif h_dict['N'] >= 25000000 and h_dict["B"] > 512:
        large_BS = h_dict["B"]
        ratio_to_512 = large_BS / 512
        # round up the ratio
        ratio_to_512_rounded = int(np.ceil(ratio_to_512))
        BS = large_BS / ratio_to_512_rounded
        BS = int(BS)  # Ensure BS is an integer
        grad_accumulation_steps = ratio_to_512_rounded   #h_dict["B"] // BS
        print(f"Using gradient accumulation with {grad_accumulation_steps} steps for batch size {h_dict['B']}")
        # save a text file
        with open(os.path.join(out_path, 'grad_accumulation_info.txt'), 'w') as f:
            f.write(f"Using gradient accumulation with {grad_accumulation_steps} steps for batch size {h_dict['B']}, each step with batch size {BS}")
        #raise ValueError("Stopping here to check gradient accumulation steps and batch size")        
    
    elif h_dict['N'] >= 128000000 and h_dict["B"] > 256:
        large_BS = h_dict["B"]
        ratio_to_512 = large_BS / 256
        # round up the ratio
        ratio_to_512_rounded = int(np.ceil(ratio_to_512))
        BS = large_BS / ratio_to_512_rounded
        BS = int(BS)  # Ensure BS is an integer
        grad_accumulation_steps = ratio_to_512_rounded   #h_dict["B"] // BS
        print(f"Using gradient accumulation with {grad_accumulation_steps} steps for batch size {h_dict['B']}")
        # save a text file
        with open(os.path.join(out_path, 'grad_accumulation_info.txt'), 'w') as f:
            f.write(f"Using gradient accumulation with {grad_accumulation_steps} steps for batch size {h_dict['B']}, each step with batch size {BS}")
        #raise ValueError("Stopping here to check gradient accumulation steps and batch size")        
    elif h_dict['N'] >= 500000000 and h_dict["B"] > 128:
        large_BS = h_dict["B"]
        ratio_to_512 = large_BS / 128
        # round up the ratio
        ratio_to_512_rounded = int(np.ceil(ratio_to_512))
        BS = large_BS / ratio_to_512_rounded
        BS = int(BS)  # Ensure BS is an integer
        grad_accumulation_steps = ratio_to_512_rounded   #h_dict["B"] // BS
        print(f"Using gradient accumulation with {grad_accumulation_steps} steps for batch size {h_dict['B']}")
        # save a text file
        with open(os.path.join(out_path, 'grad_accumulation_info.txt'), 'w') as f:
            f.write(f"Using gradient accumulation with {grad_accumulation_steps} steps for batch size {h_dict['B']}, each step with batch size {BS}")
        #raise ValueError("Stopping here to check gradient accumulation steps and batch size")        

    elif h_dict["B"] > 3072:
        
        large_BS = h_dict["B"]
        ratio_to_3072 = large_BS / 3072
        # round up the ratio
        ratio_to_3072_rounded = int(np.ceil(ratio_to_3072))
        BS = large_BS / ratio_to_3072_rounded
        BS = int(BS)  # Ensure BS is an integer
        grad_accumulation_steps = ratio_to_3072_rounded   #h_dict["B"] // BS
        print(f"Using gradient accumulation with {grad_accumulation_steps} steps for batch size {h_dict['B']}")
        # save a text file
        with open(os.path.join(out_path, 'grad_accumulation_info.txt'), 'w') as f:
            f.write(f"Using gradient accumulation with {grad_accumulation_steps} steps for batch size {h_dict['B']}, each step with batch size {BS}")
        #raise ValueError("Stopping here to check gradient accumulation steps and batch size")
    else:
        BS = h_dict["B"]
        grad_accumulation_steps = 1
        print(f"Using batch size {BS} with no gradient accumulation")

    # Training arguments
    args = TrainingArguments(
        output_dir= path,
        per_device_train_batch_size = BS,
        per_device_eval_batch_size = BS,
        gradient_accumulation_steps=grad_accumulation_steps,
        evaluation_strategy="steps",

        dataloader_num_workers=12,      # spawn 4 workers
        dataloader_pin_memory=True,    # page-lock host memory for faster H2D copies
        dataloader_prefetch_factor=2,  # each worker pre-loads 2 batches
        
        # evaluate at step 0

        max_steps=h_dict["K"] ,        # Set your desired number of steps
        save_strategy="steps",
        save_steps= 10000,
        save_total_limit=5,
        logging_steps= 1000,
        # Don't set learning_rate or lr_scheduler_type as we'll override them
        fp16=True, # TEMPORARY - for fast testing
        push_to_hub=False,
    )


    
    # check size of train_dataset
    #if not stream:
    #    print(f"Train dataset size: {len(train_dataset)}")
        
    #raise ValueError("Stopping here to check dataset size")
    # Initialize your custom trainer
    trainer = CustomTrainer(
        momentum= h_dict["momentum"],
        initial_learning_rate= h_dict["lr"],
        lr_scheduler_type= h_dict["lr_schedule"],
        step_decay_schedule_dict= h_dict["step_decay_schedule"],
        optimizer_type= h_dict["optimizer"],
        #{
       #     'decay_at': [0.5, 0.8],  # Decay at 50% and 80% of the steps
        #    'decay_amt': [0.1, 0.01]  # Decay to 10% and then to 1%
       # },
        # Pass the training arguments
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    training_start_time = time.time()
    
    # Train the model
    trainer.train()

    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    print(f"Training duration: {training_duration:.2f} seconds")
    
    # Save the model
    trainer.save_model()

    # evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # make a copy of loss_curve_df.csv (in path) to out_path
    loss_curve_df = pd.read_csv(os.path.join(path, 'loss_curve_df.csv'))
    loss_curve_df.to_csv(os.path.join(out_path, 'loss_curve_df.csv'), index=False)

    total_duration = time.time() - start_time
    print(f"Total script duration: {total_duration:.2f} seconds")

    # in the outpath, create a csv file with training duration and total duration
    time_df = pd.DataFrame({
        'training_duration_seconds': [training_duration],
        'total_duration_seconds': [total_duration]
    })
    time_df.to_csv(os.path.join(out_path, 'time_durations.csv'), index=False)




if __name__ == "__main__":
    #build_model_with_predicted_config(target_param_count=2.5e8,
    #                                    tokenizer_path="datasets/tokenizers/bpe_tokenizer_lm1b_vocab3000",
    #                                    save_dir=None,
    #                                    model_family="pythia",
    #                                    reference_configs=Pythia_configs)
    
    
    
    main()