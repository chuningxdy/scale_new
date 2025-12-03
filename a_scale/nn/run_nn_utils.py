import jax
import jax.numpy as jnp
import equinox as eqx
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict

def parameter_count(model):
    """
    Count the total number of parameters in a model.
    
    Args:
        model: An Equinox model
        
    Returns:
        int: Total number of parameters in the model
    """
    return sum(p.size for p in jax.tree_util.tree_flatten(eqx.filter(model, eqx.is_array))[0])

def conv_output_dimensions(ih, iw, ks, pad, stride, c):
    """
    Calculate the output dimensions of a convolutional layer.
    
    Args:
        ih: Input height
        iw: Input width
        ks: Kernel size
        pad: Padding
        stride: Stride
        c: Number of output channels
        
    Returns:
        tuple: Tuple containing output channels, height, and width
    """
    h = ih + 2 * pad
    w = iw + 2 * pad
    h = (h - ks + stride) // stride
    w = (w - ks + stride) // stride
    return c, h, w


def is_checkpoint_iteration(iteration, checkpoint_iterations, max_iterations):
    """
    Determines if the current iteration is a checkpoint iteration.
    
    Args:
        iteration: Current training iteration
        checkpoint_iterations: List/set of iterations to checkpoint at
        max_iterations: Total number of training iterations
        
    Returns:
        bool: True if this iteration should trigger a checkpoint
    """
    return (iteration in checkpoint_iterations) or (iteration - max_iterations in checkpoint_iterations)

@eqx.filter_jit
def evaluate(model, data, loss):
    """
    Evaluate the loss function of a model over a batch of data.
    
    Args:
        model: The model to evaluate
        data: A batch of data
        loss: Loss function to compute
        
    Returns:
        float: Mean loss value over the batch
    """
    losses = jax.vmap(loss, in_axes=(None, 0))(model, data)
    return jnp.mean(losses)

def evaluate_evals(model, loader, evals, batches):
    """
    Evaluate multiple evaluation metrics over multiple batches of data.
    
    Args:
        model: The model to evaluate
        loader: Data loader yielding batches
        evals: Dictionary of evaluation functions
        batches: Number of batches to evaluate
        
    Returns:
        dict: Dictionary containing average values for each evaluation metric
    """
    eval_values = defaultdict(lambda: 0)
    batches = hydra.utils.instantiate(batches)
    for _ in range(batches):
        batch = next(loader)
        for name, eval in evals.items():
            eval_values[name] += evaluate(model, batch, eval)/batches
    return eval_values

def init_state(cfg, filepath = None):
    """
    Initialize the training state from config and optionally load from a checkpoint.
    
    Args:
        cfg: Hydra configuration object
        filepath: Optional path to load checkpoint from
        
    Returns:
        dict: Configuration dictionary containing model, optimizer, data loader, and other training state
    """
    config = {}
    config["evals"] = {name: hydra.utils.instantiate(config) for name, config in cfg.checkpoints.evals.items()} 
    config["loss"] = hydra.utils.instantiate(cfg.loss)
    config["key"] = jax.random.key(cfg.seed)
    config["key"], subkey = jax.random.split(config["key"])
    config["model"] = hydra.utils.instantiate(cfg.model)(subkey)
    config["optimizer"] = hydra.utils.instantiate(cfg.h.optimizer)
    config["opt_state"] = config["optimizer"].init(eqx.filter(config["model"], eqx.is_inexact_array))

    if filepath is not None:
        config.update(load_state(filepath, key=config["key"], opt_state=config["opt_state"], model=config["model"]))

    config["loader"] = iter(hydra.utils.instantiate(cfg.data.loader, key=config["key"]))
    return config

def load_state(filepath, **config):
    """
    Load a checkpoint state from a file.
    
    Args:
        filepath: Path to the checkpoint file
        **config: Additional configuration parameters
        
    Returns:
        dict: Loaded state dictionary
    """
    if "key" in config:
        config["key"] = jax.random.key_data(config["key"])

    with open(filepath, "rb") as f:
        config = eqx.tree_deserialise_leaves(f, config)
    
    if "key" in config:
        config["key"] = jax.random.wrap_key_data(config["key"])

    return config

def save_state(filepath, **config):
    """
    Save the current state to a checkpoint file.
    
    Args:
        filepath: Path to save the checkpoint file
        **config: State dictionary to save
    """
    if "key" in config:
        config["key"] = jax.random.key_data(config["key"])

    with open(filepath, "wb") as f:
        eqx.tree_serialise_leaves(f, config)



def generate_step_decay_schedule(num_iterations, decay_schedule, init_lr, end_lr):
    """
    Args:
        num_iterations (int): Total number of iterations.
        decay_schedule (dict): A dictionary where keys are percentages of iterations (0-1)
                               and values are decay factors.

    Returns:
        dict: A dictionary mapping iteration boundaries to scaling factors.
    """
    #raise ValueError(decay_schedule)

    # if the first element of decay_at is <=1.0, it is interpreted as fractions of the total number of iterations
    if decay_schedule["decay_at"][0] <= 1.0:
        percents = decay_schedule["decay_at"]
        decays = decay_schedule["decay_amt"]
        sch = zip(percents, decays)
        # sort the schedule by percentage
        sorted_sch = sorted(sch, key=lambda x: x[0])

        boundaries_and_scales = {}
        for percent, decay in sorted_sch: #sorted(decay_schedule.items()):
            #percent = v[0]
        # decay = v[1]
            boundary = int(num_iterations * percent)
            boundaries_and_scales[boundary] = float(decay)
    elif decay_schedule["decay_at"][0] > 1.0:
        # if the first element of decay_at is > 1.0, it is interpreted as steps
        boundaries_and_scales = {int(boundary): float(scale) for boundary, scale in zip(decay_schedule["decay_at"], decay_schedule["decay_amt"])}
    return boundaries_and_scales

def num_batches_for_eval(batch_size, eval_size):
    # Calculate the number of batches needed to 
    # reach the desired evaluation size
    num_batches = eval_size // batch_size + 1
    return int(num_batches)



def main():
    # Test cases for conv_output_dimensions
    test_cases = [
        # (input_height, input_width, kernel_size, padding, stride, channels)
        (32, 32, 3, 1, 1, 64),  # Common CNN layer params
        (28, 28, 5, 0, 1, 32),  # MNIST-like dimensions
        (224, 224, 7, 3, 2, 16),  # ResNet-like first layer
        (153, 68, 4, 2, 5, 64),  # ResNet-like first layer
    ]
    
    for ih, iw, ks, pad, stride, c in test_cases:
        out_c, out_h, out_w = conv_output_dimensions(ih, iw, ks, pad, stride, c)
        print(f"Input: {ih}x{iw}x_, Kernel: {ks}, Pad: {pad}, Stride: {stride}, Out Channels: {c}")
        print(f"Output dimensions: {out_c}x{out_h}x{out_w}\n")

def num_batches_for_eval(batch_size, eval_size):
    # Calculate the number of batches needed to 
    # reach the desired evaluation size
    num_batches = eval_size // batch_size + 1
    return int(num_batches)

if __name__ == "__main__":
    main()
