"""
Test script for hf_utils_train_model.py
Uses configs from: outputs/nn_archive/runs/Run_2025_10_07_21_14_21_305447/
  hdict.yaml -> h_dict (hyperparameters)
  mdict.yaml -> nn_dict (model/data config)
"""

import os
import json
import shutil
import tempfile

# --- Configs from Run_2025_10_07_21_14_21_305447 ---

# From mdict.yaml
nn_dict = {
    "data": "openwebtext2",
    "loss": "condcrossent",
    "model": "pythia",
}

# From hdict.yaml
h_dict = {
    "B": 96,
    "C": 920000,
    "K": 1250,
    "N": 10000000,
    "end_lr": 0.1,
    "lr": 0.001,
    "lr_schedule": "cosine",
    "momentum": 0.0,
    "optimizer": "adamw",
    "step_decay_schedule": None,
}

# --- Set up temp folder structure expected by main() ---
temp_dir = os.path.join(os.path.dirname(__file__), "outputs", "test_run_temp")
input_dir = os.path.join(temp_dir, "input")
output_model_dir = os.path.join(os.path.dirname(__file__), "outputs", "test_run_model")

# Clean up previous test run if exists
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

os.makedirs(input_dir, exist_ok=True)

with open(os.path.join(input_dir, "nn_dict.json"), "w") as f:
    json.dump(nn_dict, f, indent=2)

with open(os.path.join(input_dir, "h_dict.json"), "w") as f:
    json.dump(h_dict, f, indent=2)

with open(os.path.join(input_dir, "path.txt"), "w") as f:
    f.write(output_model_dir)

print(f"Test inputs written to: {input_dir}")
print(f"Model will be saved to: {output_model_dir}")
print(f"Launching hf_utils_train_model.main() with temp_path={temp_dir}/")

# --- Run ---
import sys
sys.argv = [sys.argv[0], temp_dir + "/"]

from hf_utils_train_model_track_weightnorm import main
main()
