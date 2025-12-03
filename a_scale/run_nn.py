# fit NN: h + neural_net
# look up/update NN archive
# structure


# given an h_dict, and a nn_dict, return the loss curve 
# cache relevant information to path

# first, write a function that run transformers from hugging face.
# arguments: 
# model_name: pythia or llama
# model_specs: a dictionary with the modifications to the config,
#             e.g. {"num_labels": 2, "hidden_size": 512}
# dataset: the dataset to use - check if the dataset is saved locally, if not
#         download it from hugging face
# path: the path for cache

import equinox as eqx
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import os
from collections import defaultdict
import yaml
from datetime import datetime

import a_scale.archiving_utils as a_utils
import a_scale.nn.run_nn_utils as utils
import pandas as pd
import jax 
from jax import numpy as jnp
import gc

OmegaConf.register_new_resolver("eval", eval)


import jax
import jax.numpy as jnp
import functools
import contextlib

import json
import subprocess
import time





@contextlib.contextmanager
def use_f32_precision():
    """Context manager to temporarily force 32-bit precision in JAX."""
    original = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        yield
    finally:
        jax.config.update("jax_enable_x64", original)

def f32_function(func):
    """Decorator to ensure a function runs with 32-bit precision."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with use_f32_precision():
            return func(*args, **kwargs)
    return wrapper

@eqx.filter_jit
def train_step(model, opt_state, loss, batch, optim):
    """
    Perform a single training step:
    1. Calculate loss and gradients
    2. Update model parameters using optimizer
    3. Return updated model and loss value
    """
    loss_value, grads = eqx.filter_value_and_grad(utils.evaluate)(model, batch, loss)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

@f32_function
def train_nn_simple(nn_cfg, path):

    train_cfg = OmegaConf.load("conf/train/train.yaml")
    # merge the two configurations
    cfg = OmegaConf.merge(nn_cfg, train_cfg)
    # save the merged configuration in the output directory
    with open(os.path.join(path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    # Set up logging to track training progress and configuration details
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("TRAIN")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Hydra automatically creates unique output directories for each run
    # This section gets the directory path and job information
    #hydra_config = hydra.core.hydra_config.HydraConfig.get()
    output_dir = path
    # split path into the folder name and the path to the folder
    folder_name = os.path.basename(output_dir)
    dir_name = os.path.dirname(output_dir)
    job_name = dir_name
    job_id = folder_name
    #job_id = hydra_config.job.id if hydra_config.mode == "MULTIRUN" else None

    
    # if configuration has run_dir, use that as the output_dir
   # if "run_dir" in cfg:
   #     output_dir = cfg["run_dir"]
   # if "run_id" in cfg:
   #     job_id = (hydra_config.job.id + "_"+ str(cfg["run_id"])) if hydra_config.mode == "MULTIRUN" else None
    

    # Initialize Weights & Biases (wandb) for experiment tracking
    # This will create a new run and log all configurations
    wandb.init(project=cfg.wandb.project,
               config=OmegaConf.to_container(cfg, resolve=True),
               group=job_name,
               job_type="train",
               dir=output_dir,
               name=cfg.wandb.nameprefix + (f":{job_id}" if job_id is not None else ""))
    logger.info(f"wandb.run.name: {wandb.run.name}")

    # Initialize all components needed for training:
    # - loss: The loss function to optimize
    # - model: The neural network to be trained
    # - optimizer: Updates model parameters based on gradients
    # - opt_state: The optimizer state
    # - loader: Provides batches of training data
    # - evals: Functions to evaluate model performance
  
    # update the train_cfg with nn_cfg
    configuration = utils.init_state(cfg)
    loss = configuration["loss"]
    model = configuration["model"]
    optim = configuration["optimizer"]
    opt_state = configuration["opt_state"]
    loader = configuration["loader"]
    evals = configuration["evals"]

    # Count the number of parameters in the model
    # and report them in the training report
    true_param_count = utils.parameter_count(model)
    report = defaultdict(list)
    report["num_parameters"] = true_param_count
    report["num_iterations"] = cfg.h.K
    report["learning_rate"] = cfg.h.lr
    report["batch_size"] = cfg.h.B

    # Set up directories for saving checkpoints and evaluation results
    # A checkpoint contains the model state at a particular point in training
    evals_filepath = os.path.join(output_dir, "evals.yaml")
    checkpoint_dir = os.path.join(output_dir, "ckpts")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Main training loop - iterates through specified number of training steps
    logger.info(f"model: {model}")
    logger.info(f"model param count: {true_param_count}")
    logger.info("Starting training...")
    for i in range(cfg.h.K+1):

        # Checkpoint block: Periodically save model state and evaluate performance
        if utils.is_checkpoint_iteration(i, cfg.checkpoints.iterations, cfg.h.K):

            logger.info(f"CHECKPOINT i={i}")
            # Save model state if configured
            if cfg.checkpoints.savestate:
                ckpt_filepath = os.path.join(checkpoint_dir, f"ckpt-{i}.eqx")
                utils.save_state(ckpt_filepath, key=loader.key, opt_state=opt_state, model=model)
                logger.info(f"\tmodel saved to {ckpt_filepath}")

            # Run evaluation functions and log results
            num_batches_for_evals_log = hydra.utils.instantiate(cfg.checkpoints.num_batches)
            logger.info(f"evaluating model on {num_batches_for_evals_log} batches of batch size: {cfg.h.B}")
            eval_values = utils.evaluate_evals(model, loader, evals, cfg.checkpoints.num_batches)

            for name in evals.keys():
                logger.info(f"\t{name}: {eval_values[name]}")
                wandb.log({f"evals/{name}": eval_values[name]}, step=i)
                report[name].append({i : float(eval_values[name])})

            # Save accumulated evaluation results to YAML file for later analysis
            with open(evals_filepath, 'w') as f:
                yaml.dump(dict(report), f)
            logger.info(f"\tevals saved to {evals_filepath}")

        if i == cfg.h.K:
            break
        
        # Training step: Get next batch of data and update model parameters
        batch = next(loader)
        model, opt_state, loss_value = train_step(model, opt_state, loss, batch, optim)

        # Log training loss to wandb at specified intervals for monitoring
        if i % cfg.logevery == 0:
            wandb.log({"train/batch-loss": loss_value}, step=i)

    # Clean up wandb connection when training is complete
    logger.info("Training completed.")
    wandb.finish()
    
    with open(path + "evals.yaml") as f:
        evals = OmegaConf.load(f)
        loss_dicts = evals["cross_entropy"]
        loss_dict = {k: v for d in loss_dicts for k, v in d.items()}
        # the keys become the ckpt column, the values become the loss column
        loss_df_i = pd.DataFrame(loss_dict.items(), columns = ["ckpt", "loss"])
        # convert the iteration to int, so that it can be used as a join key
        loss_df_i["ckpt"] = loss_df_i["ckpt"].astype(int)

        num_params = int(evals["num_parameters"])
        # convert to a data frame
        actual_N_df = pd.DataFrame({"actual_N": [num_params]})
    
    
    return {"loss_curve_df":loss_df_i, "actual_N_df": actual_N_df}


def train_hf_jax(nn_dict, h_dict, path):

    hf_temp_folder = 'outputs/nn_hf/'
    # add date and time to the folder name
    # use all the time to avoid overwriting incl. seconds and milliseconds


    # Get current time with milliseconds
    now = datetime.now()

    # Format with milliseconds
    timestamp = now.strftime("%Y%m%d-%H%M%S") + f".{now.microsecond // 1000:03d}"
    hf_temp_folder = hf_temp_folder + timestamp + '/'
    # create a subfolder to save the inputs
    input_path = hf_temp_folder + 'input/'
    # if the folder exist, rename it with a new name
    # using date and time
    date_time_str = time.strftime("%Y%m%d-%H%M%S")
    new_folder_name = hf_temp_folder + 'input_old' + date_time_str + '/'
    if os.path.exists(input_path):
        os.rename(input_path, new_folder_name)
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # save nn_dict and h_dict
    with open(input_path + 'nn_dict.json', 'w') as f:
        json.dump(nn_dict, f)
    with open(input_path + 'h_dict.json', 'w') as f:
        json.dump(h_dict, f)
    # save path
    with open(input_path + 'path.txt', 'w') as f:
        f.write(path)

    # check if output folder exists, if so rename it with a new name
    # using date and time
    output_path = hf_temp_folder + 'output/'
    new_folder_name = hf_temp_folder + 'output_old' + date_time_str + '/'
    if os.path.exists(output_path):
        os.rename(output_path, new_folder_name)

    path_to_conda_env = "/mfs1/u/chuning/torch_hf"

    RESUME_FROM_CHECKPOINT = False

    if nn_dict["model"] in ["pythia", "llama"]:
        if nn_dict["data"] == "openwebtext2":
            if h_dict["lr_schedule"] == "step" and isinstance(h_dict["step_decay_schedule"], dict):
            # check if B_decay_amt in step_decay_schedule
                if "B_decay_amt" in h_dict["step_decay_schedule"]:
                    path_to_script = "hf_utils_train_model_owt_BS_sch.py"
                else:
                    if RESUME_FROM_CHECKPOINT:
                        path_to_script = "hf_utils_train_model_resume_from_checkpoint.py"
                    else:
                        path_to_script = "hf_utils_train_model.py"
            else:
                if RESUME_FROM_CHECKPOINT:
                    path_to_script = "hf_utils_train_model_resume_from_checkpoint.py"
                else:
                    path_to_script = "hf_utils_train_model.py"

        elif nn_dict["data"] == "lm1b":
            path_to_script = "hf_utils_train_model_lm1b.py"
        
        else:
            raise ValueError("Data is ", nn_dict["data"], ", not supported")
        #path_to_script = "hf_utils_train_model.py"
    else:
        raise ValueError("Model is ", nn_dict["model"], ", not supported")
    
    
    #Dump memory profile to a file
    #jax.profiler.save_device_memory_profile("jax_memory_profile.txt")
    jax.clear_backends()  # Clears JAX's internal backend state
    gc.collect()  # Runs garbage collection to free memory
    #jax.profiler.save_device_memory_profile("jax_memory_profile_after_gc.txt")

    
    # run the script hf_utils_pythia.py under a different conda env using command line
    # the name of the conda env is "hf": conda run -prefix /path/to/conda/env python hf_utils_pythia.py
    

    # Add it to the command
    command = f"conda run --prefix {path_to_conda_env} python {path_to_script} {hf_temp_folder}"
    print("run command: ", path_to_script)
    print(h_dict["step_decay_schedule"])
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Script executed successfully")
        print("Output:", result.stdout)
    else:
        print("Error executing script")
        print("Error:", result.stderr)
        raise ValueError("Error executing script")
    

    

    # load the outputs from the output folder into a dictionary,
    # where the keys are the names of the files in the output folder
    # and the values are data frames loaded from the files (csv)
    output_dict = {}
    for file in os.listdir(output_path):
        # get file name w/o extension
        file_name = os.path.splitext(file)[0]
        output_dict[file_name] = pd.read_csv(output_path + file)
    
    return output_dict


def build_hf_jax(nn_dict, h_dict, path, return_model_details = False):
    
    hf_temp_folder = 'outputs/nn_hf/'
    # add date and time to the folder name
    # Get current time with milliseconds
    now = datetime.now()

    # Format with milliseconds
    timestamp = now.strftime("%Y%m%d-%H%M%S") + f".{now.microsecond // 1000:03d}"
    hf_temp_folder = hf_temp_folder + timestamp + '/'

    # create a subfolder to save the inputs
    input_path = hf_temp_folder + 'input/'
    # if the folder exist, rename it with a new name
    # using date and time
    date_time_str = time.strftime("%Y%m%d-%H%M%S")
    new_folder_name = hf_temp_folder + 'input_old' + date_time_str + '/'
    if os.path.exists(input_path):
        os.rename(input_path, new_folder_name)
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # save nn_dict and h_dict
    with open(input_path + 'nn_dict.json', 'w') as f:
        json.dump(nn_dict, f)
    with open(input_path + 'h_dict.json', 'w') as f:
        json.dump(h_dict, f)
    # save path
    with open(input_path + 'path.txt', 'w') as f:
        f.write(path)

    # check if output folder exists, if so rename it with a new name
    # using date and time
    output_path = hf_temp_folder + 'output/'
    new_folder_name = hf_temp_folder + 'output_old' + date_time_str + '/'
    if os.path.exists(output_path):
        os.rename(output_path, new_folder_name)

    path_to_conda_env = "/mfs1/u/chuning/torch_hf"

    if nn_dict["model"] in ["pythia", "llama"]:
        if nn_dict["data"] == "lm1b":
            path_to_script = "hf_utils_build_model_lm1b.py"
        elif nn_dict["data"] == "openwebtext2":
            path_to_script = "hf_utils_build_model.py"
        else:
            raise ValueError("Data is ", nn_dict["data"], ", not supported")
    else:
        raise ValueError("Model is ", nn_dict["model"], ", not supported")
    

    #Dump memory profile to a file
    #jax.profiler.save_device_memory_profile("jax_memory_profile.txt")
    jax.clear_backends()  # Clears JAX's internal backend state
    gc.collect()  # Runs garbage collection to free memory
    #jax.profiler.save_device_memory_profile("jax_memory_profile_after_gc.txt")

    
    # run the script hf_utils_pythia.py under a different conda env using command line
    # the name of the conda env is "hf": conda run -prefix /path/to/conda/env python hf_utils_pythia.py
    command = f"conda run --prefix {path_to_conda_env} python {path_to_script} {hf_temp_folder}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Script executed successfully")
        print("Output:", result.stdout)
    else:
        print("Error executing script")
        print("Error:", result.stderr)
        raise ValueError("Error executing script")
    

    

    # load the outputs from the output folder into a dictionary,
    # where the keys are the names of the files in the output folder
    # and the values are data frames loaded from the files (csv)
    output_dict = {}
    for file in os.listdir(output_path):
        # get file name w/o extension
        file_name = os.path.splitext(file)[0]
        output_dict[file_name] = pd.read_csv(output_path + file)
    
    actual_N_df = output_dict["actual_N_df"]
    actual_N_value = actual_N_df["actual_N"].values[0]
    actual_N_value = int(actual_N_value)

    if return_model_details:
        # there should be a model_info_df.csv file in the output folder
        model_info_df = output_dict["model_info_df"]
        model_info = model_info_df.to_dict(orient='records')[0]  # Convert to a dictionary
        
        return actual_N_value, model_info
    

    return actual_N_value

def train_hf_jax_old(nn_dict, h_dict, path):

    hf_temp_folder = 'outputs/hf_temp/'
    # create a subfolder to save the inputs
    input_path = hf_temp_folder + 'input/'
    # if the folder exist, rename it with a new name
    # using date and time
    date_time_str = time.strftime("%Y%m%d-%H%M%S")
    new_folder_name = hf_temp_folder + 'input_old' + date_time_str + '/'
    if os.path.exists(input_path):
        os.rename(input_path, new_folder_name)
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # save nn_dict and h_dict
    with open(input_path + 'nn_dict.json', 'w') as f:
        json.dump(nn_dict, f)
    with open(input_path + 'h_dict.json', 'w') as f:
        json.dump(h_dict, f)
    # save path
    with open(input_path + 'path.txt', 'w') as f:
        f.write(path)

    # check if output folder exists, if so rename it with a new name
    # using date and time
    output_path = hf_temp_folder + 'output/'
    new_folder_name = hf_temp_folder + 'output_old' + date_time_str + '/'
    if os.path.exists(output_path):
        os.rename(output_path, new_folder_name)

    path_to_conda_env = "/mfs1/u/chuning/torch_hf"

    if nn_dict["model"] == "pythia":
        if nn_dict["data"] == "lm1b":
            path_to_script = "hf_utils_pythia_lm1b.py"
        elif nn_dict["data"] == "openwebtext2":
            path_to_script = "hf_utils_pythia.py"
        else:
            raise ValueError("Data is", nn_dict["data"], "not supported")
    else:
        raise ValueError("Model is", nn_dict["model"], "not supported")
    

    #Dump memory profile to a file
    #jax.profiler.save_device_memory_profile("jax_memory_profile.txt")
    jax.clear_backends()  # Clears JAX's internal backend state
    gc.collect()  # Runs garbage collection to free memory
    #jax.profiler.save_device_memory_profile("jax_memory_profile_after_gc.txt")

    
    # run the script hf_utils_pythia.py under a different conda env using command line
    # the name of the conda env is "hf": conda run -prefix /path/to/conda/env python hf_utils_pythia.py
    command = f"conda run --prefix {path_to_conda_env} python {path_to_script}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Script executed successfully")
        print("Output:", result.stdout)
    else:
        print("Error executing script")
        print("Error:", result.stderr)
        raise ValueError("Error executing script")
    

    

    # load the outputs from the output folder into a dictionary,
    # where the keys are the names of the files in the output folder
    # and the values are data frames loaded from the files (csv)
    output_dict = {}
    for file in os.listdir(output_path):
        # get file name w/o extension
        file_name = os.path.splitext(file)[0]
        output_dict[file_name] = pd.read_csv(output_path + file)
    
    return output_dict


def train_nn(mdict, hdict, path):

    if "model" in mdict and mdict["model"] in ["pythia", "llama"]:
        
        # call the function
        output_dict = train_hf_jax(nn_dict = mdict, h_dict = hdict, 
                                   path = path)
        out_dict = output_dict

    else:
        raise ValueError("Model is", mdict["model"], "not supported")
        
    #elif "model" in mdict and mdict["model"] in ["simplecnn", "mlp"]:
    #    nn_cfg = a_utils.mhdict_to_cfg(mdict, hdict)
    ##    with open(path + "cfg.yaml", "w") as f:
    #        OmegaConf.save(nn_cfg, f)
    #    out_dict = train_nn_simple(nn_cfg, path)
    return out_dict


def build_nn(mdict, hdict, path = "outputs/llama_model/temp"):

    if "model" in mdict and mdict["model"] in ["pythia", "llama"]:
        
        # call the function
        actual_N_value = build_hf_jax(nn_dict = mdict, h_dict = hdict, 
                                   path = path)
        actual_N_value = int(actual_N_value)#out_dict = output_dict

    else:
        raise ValueError("Model is", mdict["model"], "not supported")
        
    #elif "model" in mdict and mdict["model"] in ["simplecnn", "mlp"]:
    #    nn_cfg = a_utils.mhdict_to_cfg(mdict, hdict)
    ##    with open(path + "cfg.yaml", "w") as f:
    #        OmegaConf.save(nn_cfg, f)
    #    out_dict = train_nn_simple(nn_cfg, path)
    return actual_N_value

if __name__ == "__main__":
    # Get the path to the output directory
    path = "/mfs1/u/chuning/scale/outputs/tst/0/"
    hdict = {"N": 2000000, "B": 12, "K": 100, "lr": 0.01, "end_lr": 0.001, "momentum": 0.9, 
            "lr_schedule": "step", "optimizer": "sgd"}
    mdict = {"model": "simplecnn", "data": "cifar5m", "loss": "condcrossent"}
    
    # Load the neural network configuration
    nn_cfg = a_utils.mhdict_to_cfg(mdict, hdict)
    out_dict = train_nn(nn_cfg, path)
    loss_curve_df = out_dict["loss_curve_df"]
    print(loss_curve_df.tail())
    actual_N_df = out_dict["actual_N_df"]
    print(actual_N_df)
    