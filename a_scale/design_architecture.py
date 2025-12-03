
import numpy as np
import pandas as pd
import os
import json
import scipy.stats as stats


# ------- helpers -----------

class ModelArchitecture:
    """Module for calculating and designing model architectures."""
    """Using requested parameter count"""
    
    @staticmethod
    def calculate_params_simplified(hidden_size, num_layers):
        """
        Calculate transformer parameters using the simplified formula: layers * hidden² * 12
        This formula matches Pythia's parameter counting very accurately.
        """
        return num_layers * (hidden_size ** 2) * 12
    
    @staticmethod
    def get_pythia_data():
        """Get Pythia model architecture data."""
        return pd.DataFrame({
            'params_millions': [18.9, 70, 160, 410, 1000, 1400, 2800, 6900, 12000],
            'd_model': [512, 512, 768, 1024, 2048, 2048, 2560, 4096, 5120],
            'n_layers': [6, 6, 12, 24, 16, 24, 32, 32, 36]
        })
    
    @classmethod
    def fit_power_law_scaling(cls, data):
        """
        Fit power law relationships to empirical model data.
        
        Args:
            data: DataFrame with columns 'params_millions', 'd_model', 'n_layers'
            
        Returns:
            Dict with power law coefficients and exponents
        """
        # Recalculate parameters using the simplified formula
        data = data.copy()
        for i, row in data.iterrows():
            # Calculate parameters with the simplified formula
            params = cls.calculate_params_simplified(row['d_model'], row['n_layers'])
            data.at[i, 'params_millions'] = params / 1_000_000
        
        # Extract data
        params_millions = data['params_millions'].values
        hidden_size = data['d_model'].values
        layers = data['n_layers'].values
        
        # Convert to log space for power law fitting
        log_params = np.log(params_millions)
        log_hidden = np.log(hidden_size)
        log_layers = np.log(layers)
        
        # Fit linear regression for hidden size
        slope_hidden, intercept_hidden, r_value_hidden, _, _ = stats.linregress(log_params, log_hidden)
        
        # Fit linear regression for layers
        slope_layers, intercept_layers, r_value_layers, _, _ = stats.linregress(log_params, log_layers)
        
        # Calculate power law coefficients
        hidden_coef = np.exp(intercept_hidden)
        hidden_exp = slope_hidden
        layers_coef = np.exp(intercept_layers)
        layers_exp = slope_layers
        
        return {
            'hidden_coef': hidden_coef,
            'hidden_exp': hidden_exp,
            'hidden_r2': r_value_hidden**2,
            'layers_coef': layers_coef,
            'layers_exp': layers_exp,
            'layers_r2': r_value_layers**2
        }
    
    @classmethod
    def get_pythia_scaling_parameters(cls):
        """Compute power law scaling parameters for Pythia models."""
        # Get Pythia data
        pythia_df = cls.get_pythia_data()
        
        # Fit power law relationships
        return cls.fit_power_law_scaling(pythia_df)
    
    @classmethod
    def design_model_architecture(cls, 
                                  target_total_params, 
                                  vocab_size):
        """
        Design model architecture based on Pythia's empirical scaling laws.
        Uses the simplified parameter formula: layers * hidden² * 12
        
        Args:
            target_non_embedding_params: Target parameter count
            vocab_size: Vocabulary size
            
        Returns:
            Dict with model architecture configuration
        """
        # Get Pythia scaling parameters by computing them dynamically
        scaling = cls.get_pythia_scaling_parameters()


        # we know that hidden = scaling['hidden_coef'] * (target_params_millions ** scaling['hidden_exp'])
        # and layers = scaling['layers_coef'] * (target_params_millions ** scaling['layers_exp'])
        # we also know that total_nonembedding_params = layers * hidden² * 12
        # and embedding_params = vocab_size * hidden
        # thus total_params = total_nonembedding_params + embedding_params = layers * hidden² * 12 + vocab_size * hidden + vocab_size * hidden
        # we want to express total_nonembedding_params in terms of total_params: 
        
        def calculate_total_params(non_embedding_params, vocab_size, scaling_params):
            """Calculate total params for a given non-embedding params target."""
            # Convert to millions for the power law formulas
            params_millions = non_embedding_params / 1_000_000
            
            # Predict hidden size using power law
            hidden_size_raw = scaling_params['hidden_coef'] * (params_millions ** scaling_params['hidden_exp'])
            hidden_size = int(round(hidden_size_raw / 16) * 16)  # Round to multiple of 16
            hidden_size = max(16, hidden_size)  # Ensure at least 16
            
            # Predict number of layers using power law
            num_layers_raw = scaling_params['layers_coef'] * (params_millions ** scaling_params['layers_exp'])
            num_layers = int(round(num_layers_raw)) #max(2, int(round(num_layers_raw)))  # Minimum 2 layers
            # if num_layers <2 , raise
            if num_layers < 2:
                raise ValueError('Number of layers must be at least 2; need more params')
            # Calculate embedding params based on hidden size
            embedding_params = vocab_size * hidden_size
            
            # Calculate non-embedding params based on the simplified formula
            actual_non_embedding = num_layers * (hidden_size ** 2) * 12
            
            return {
                'total_params': actual_non_embedding + embedding_params,
                'non_embedding_params': actual_non_embedding,
                'embedding_params': embedding_params,
                'hidden_size': hidden_size,
                'num_layers': num_layers
            }
        
        def solve_for_non_embedding_params(total_params, vocab_size, scaling_params):
            """
            Find the target non-embedding params that will yield a model close to
            the desired total parameter count, using uniform sampling.
            """
            # Updated search range: 40% to 99.9% of total params
            min_ratio = 0.50
            max_ratio = 1.50
            
            # Number of samples to evaluate (increased to 1000)
            num_samples = 1000
            
            # Initialize best solution
            best_solution = None
            best_error = float('inf')
            
            # Uniform sampling across the range
            for i in range(num_samples):
                # Sample uniformly from the range
                ratio = min_ratio + (max_ratio - min_ratio) * (i / (num_samples - 1))
                non_embedding = ratio * total_params
                
                result = calculate_total_params(non_embedding, vocab_size, scaling_params)
                error = abs(result['total_params'] - total_params)
                
                if error < best_error:
                    best_error = error
                    best_solution = result
                
                # Early stopping if we're within tolerance
                if error / total_params < 0.05:  # 5% tolerance
                    break
            
            return best_solution
    
        # Find the optimal architecture configuration
        solution = solve_for_non_embedding_params(target_total_params, vocab_size, scaling)
        
        # Extract architecture parameters
        hidden_size = solution['hidden_size']
        num_layers = solution['num_layers']
        target_non_embedding_params = solution['non_embedding_params']
    

       # # Convert target params to millions for scaling formula
       # target_params_millions = target_non_embedding_params / 1_000_000
        
        # Apply power law formulas
       # hidden_size_raw = scaling['hidden_coef'] * (target_params_millions ** scaling['hidden_exp'])
       # num_layers_raw = scaling['layers_coef'] * (target_params_millions ** scaling['layers_exp'])
        
        # Round to practical values
       # hidden_size = int(round(hidden_size_raw / 16) * 16)  # Round to multiple of 16
       # num_layers = max(2, int(round(num_layers_raw)))      # Minimum 2 layers
        
        # Ensure hidden size is at least 16
       # hidden_size = max(16, hidden_size)
        
        # Calculate number of attention heads (1 head per 16 dimensions)
        num_heads = max(1, hidden_size // 16)
        
        # Standard 4x for intermediate size
        intermediate_size = 4 * hidden_size
        
        # Calculate actual parameters with simplified formula
        actual_params = cls.calculate_params_simplified(hidden_size, num_layers)
        
        return {
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
            'num_attention_heads': num_heads,
            'intermediate_size': intermediate_size,
            'vocab_size': vocab_size,
            'max_position_embeddings': 2048,
            'rotary_pct': 0.25,
            'tie_word_embeddings': True,  # Tie embeddings to reduce parameter count
            'simplified_params': actual_params  # Include estimated parameters
        }



def remove_embedding_params(hdict, vocab_size):

    # vocab_size = 0 means no need to remove embedding params
    total_N = hdict["N"]
    hdict_copy = hdict.copy()
    arch_dict = ModelArchitecture.design_model_architecture(total_N, vocab_size)
    hdict_copy["N"] = arch_dict["simplified_params"]
    return hdict_copy

def convert_to_effective_params(hdict, effective_model_size_factor, pow = False):
    
    
    # vocab_size = 0 means no need to remove embedding params
    total_N = hdict["N"]
    hdict_copy = hdict.copy()
    combined = True
    if combined:
        effective_model_size_mult = effective_model_size_factor[0]
        effective_model_size_pow = effective_model_size_factor[1]
        hdict_copy["N"] = int((effective_model_size_mult * total_N) ** effective_model_size_pow)
        #raise ValueError('effective model size factor:', effective_model_size_factor,
        #      'effective_model_size_mult:', effective_model_size_mult,
        #        'effective_model_size_pow:', effective_model_size_pow,
         #       'raw_N:', total_N,
         #       'new_N:', hdict_copy["N"])
    elif pow:
        hdict_copy["N"] = int(total_N ** effective_model_size_factor)
    else:
        hdict_copy["N"] = int(total_N * effective_model_size_factor)
    return hdict_copy

if __name__ == "__main__":
    # Design model architecture for a target parameter count
    target_params = 100_000
    vocab_size = 3000
    architecture = ModelArchitecture.design_model_architecture(target_params, vocab_size)
    total_params = architecture['vocab_size'] * architecture['hidden_size'] + architecture['simplified_params']
    architecture['total_params'] = total_params
    architecture['embedding_params'] = architecture['vocab_size'] * architecture['hidden_size']
    architecture['non_embedding_params'] = architecture['simplified_params']
    print(json.dumps(architecture,
                        indent=4,
                        sort_keys=True))