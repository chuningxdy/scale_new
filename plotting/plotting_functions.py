import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
if not (jnp.array(1.).dtype is jnp.dtype("float64")):
    raise RuntimeError("jax.config.jax_enable_x64 must be set for numerical stability. "
                      "Please set this environment variable before using NQS.")
from collections import namedtuple
from functools import partial
import os
import time
import datetime
import json
from jax import lax

import numpy as np
from pyDOE import lhs  # for latin hypercube sampling

import pandas as pd

from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from nqs_sgd import EM_nqs_from_cfg_six_standard as compute_nqs_no_lra
from nqs_sgd import EM_nqs_from_cfg_six_optimized as compute_nqs


# start a logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# ============ NQS Loss Cache ============
import pickle
import hashlib

NQS_CACHE_FILE = "/mfs1/u/chuning/scale_new/plotting/nqs_loss_cache.pkl"
_nqs_cache = None  # Global cache dictionary

def _load_nqs_cache():
    """Load the NQS loss cache from disk."""
    global _nqs_cache
    if _nqs_cache is not None:
        return _nqs_cache
    if os.path.exists(NQS_CACHE_FILE):
        try:
            with open(NQS_CACHE_FILE, 'rb') as f:
                _nqs_cache = pickle.load(f)
            logger.info(f"Loaded NQS cache with {len(_nqs_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load NQS cache: {e}. Starting fresh.")
            _nqs_cache = {}
    else:
        _nqs_cache = {}
    return _nqs_cache

def _save_nqs_cache():
    """Save the NQS loss cache to disk."""
    global _nqs_cache
    if _nqs_cache is not None:
        try:
            with open(NQS_CACHE_FILE, 'wb') as f:
                pickle.dump(_nqs_cache, f)
            logger.info(f"Saved NQS cache with {len(_nqs_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save NQS cache: {e}")

def _make_nqs_cache_key(N, B, K, init_lr, lr_sch_type, sch, model_dict, use_LRA):
    """Create a unique cache key from NQS loss parameters."""
    # Convert all params to a canonical string representation
    key_parts = [
        f"N={N}",
        f"B={B}",
        f"K={K}",
        f"init_lr={init_lr}",
        f"lr_sch_type={lr_sch_type}",
        f"sch={json.dumps(sch, sort_keys=True)}",
        f"model={json.dumps(model_dict, sort_keys=True)}",
        f"use_LRA={use_LRA}"
    ]
    key_str = "|".join(key_parts)
    # Use hash for shorter key
    return hashlib.md5(key_str.encode()).hexdigest()

# ========================================

def compute_chin_loss(N, B, K, model_dict, seq_len):
    """Compute the chin loss given N, B, K and model_dict

    Chinchilla loss formula: L(N, D) = E + A/N^α + B_coef/D^β
    where D = seq_len * B * K (total tokens)
    """
    # extract parameters from model_dict
    A_coef = model_dict['A']
    B_coef = model_dict['B']  # Renamed to avoid collision with batch size B
    E = model_dict['E']
    N_power = model_dict['N_power']
    D_power = model_dict['D_power']
    D = seq_len * B * K  # B here is batch size (function parameter)
    loss = E + A_coef / (N ** N_power) + B_coef / (D ** D_power)

    return loss

def compute_nqs_loss(N, B, K, init_lr, lr_sch_type, sch, model_dict, use_LRA=True):
    """Compute the nqs loss given N, B, K, sch and model_dict.

    Results are cached to avoid recomputation.
    """
    # Check cache first
    cache = _load_nqs_cache()
    cache_key = _make_nqs_cache_key(N, B, K, init_lr, lr_sch_type, sch, model_dict, use_LRA)

    if cache_key in cache:
        return cache[cache_key]

    # Not in cache, compute it
    # extract parameters from model_dict
    # p, q, P, Q, e_irr, R, r
    p = model_dict['p']
    q = model_dict['q']
    P = model_dict['P']
    Q = model_dict['Q']
    e_irr = model_dict['e_irr']
    R = model_dict['R']
    r = model_dict['r']

    # turn into an OmegaConf dict
    # start with empty cfg
    nqs_cfg = OmegaConf.create()

    h = OmegaConf.create()
    h['lr'] = init_lr
    h['lr_schedule'] = lr_sch_type
    if use_LRA:
        h['lr_schedule'] = "optimized"
    h['step_decay_schedule'] = sch
    h['momentum'] = 0.0
    h['N'] = N
    h['B'] = B
    h['K'] = K

    nqs_cfg['h'] = h

    nqs_cfg['P'] = P
    nqs_cfg['Q'] = Q
    nqs_cfg['p'] = p
    nqs_cfg['q'] = q
    nqs_cfg['e_irr'] = e_irr
    nqs_cfg['R'] = R
    nqs_cfg['r'] = r

    nqs = jnp.array([p, q, P, Q, e_irr, R, r])
    if use_LRA:
        out_dict = compute_nqs(nqs_cfg)
    else:
        out_dict = compute_nqs_no_lra(nqs_cfg)

    result = out_dict['nqs_df']['nqs_risk'].values[0]

    # Save to cache
    cache[cache_key] = result

    return result



def load_NN_data(isoflops_path, NBK_path, isotokens_path):

    keep_cols = ["actual_N", "N", "B", "K", "loss", "NN_loss"]
    join_cols = ["N", "B", "K"]
    isoflops_df = pd.read_csv(isoflops_path)
    
    # retain columns N, B, K, loss, nqs_loss, NN_loss (if these columns exist)
    isoflops_df = isoflops_df[[col for col in keep_cols if col in isoflops_df.columns]]
    isoflops_df["type"] = "isoflops"
    print("isoflops_df head:")
    print(isoflops_df.head(2))

    NBK_df = pd.read_csv(NBK_path)
    
    NBK_df = NBK_df[[col for col in keep_cols if col in NBK_df.columns]]
    NBK_df["type"] = "NBK"
    print("NBK_df head:")
    print(NBK_df.head(2))


    isotokens_df = pd.read_csv(isotokens_path)
    
    isotokens_df = isotokens_df[[col for col in keep_cols if col in isotokens_df.columns]]
    isotokens_df["type"] = "isotokens"
    print("isotokens_df head:")
    print(isotokens_df.head(2))

    # concatenate the three dataframes
    combined_df = pd.concat([isoflops_df, NBK_df, isotokens_df], ignore_index=True)
    
    # remove rows with any NaN values
    
    # log the nas removed
    initial_len = len(combined_df)
    combined_df = combined_df.dropna()
    final_len = len(combined_df)
    logger.info(f"Removed {initial_len - final_len} rows with NaN values.")

    # save to csv
    combined_df.to_csv("/mfs1/u/chuning/scale_new/plotting/combined_data.csv", index=False)

    return None

def scientific_round(x, sig_figs):
    if x == 0:
        return 0
    else:
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)

def compute_C_from_NBK_seq_len(N, B, K, seq_len):
    C = N/1e5 * B/1e5 * K/1e5 * seq_len * 6 # in Petaflops
    # round to two significant figures, scientific notation
    C = scientific_round(C, 2)
    return C

def figure1(NN_dat, seq_len, color_palette="magma"):
    # figure 1 contains two panels.
    # left is isoflops; right is NBK surface

    # first, compute C & D
    NN_dat["C"] = NN_dat.apply(lambda row: compute_C_from_NBK_seq_len(row["actual_N"], row["B"], row["K"], seq_len), axis=1)
    NN_dat["D"] = seq_len * NN_dat["B"] * NN_dat["K"]

    isoflop_dat = NN_dat[NN_dat["type"] == "isoflops"]
    isoflop_dat = isoflop_dat.copy()

    # add a column nqs_loss_value
    isoflop_dat["nqs_loss"] = isoflop_dat.apply(lambda row: compute_nqs_loss(row["actual_N"], row["B"], row["K"], 1.999, "constant", {"decay_at":[], "decay_amt":[], "B_decay_amt":[]}, nqs_model_dict), axis=1)
    #isoflop_dat["nqs_loss"] = isoflop_dat["NN_loss"] # placeholder
    # add a column chin_loss_value
    isoflop_dat["chin_loss"] = isoflop_dat.apply(lambda row: compute_chin_loss(row["actual_N"], row["B"], row["K"], 
                                                                             chin_model_dict, seq_len), axis=1)
    # save isoflop_dat to csv for debugging
    isoflop_dat.to_csv("isoflop_dat_debug.csv", index=False)
    # group by C
    # the color should be by grouped C values
    Cs = sorted(isoflop_dat["C"].unique().tolist())

    # Group C values that are within 70%-130% of each other
    def group_similar_Cs(C_list, tolerance=0.3):
        """Group C values that are within tolerance (e.g., 0.3 = 70%-130%) of each other."""
        if not C_list:
            return []
        groups = []
        current_group = [C_list[0]]
        for C in C_list[1:]:
            # Check if C is within tolerance of the first element in current group
            if C <= current_group[0] * (1 + tolerance) and C >= current_group[0] * (1 - tolerance):
                current_group.append(C)
            else:
                groups.append(current_group)
                current_group = [C]
        groups.append(current_group)
        return groups

    C_groups = group_similar_Cs(Cs, tolerance=0.3)

    # Create a mapping from original C to group representative (mean of group)
    C_to_group = {}
    group_representatives = []
    for group in C_groups:
        rep = np.mean(group)
        group_representatives.append(rep)
        for C in group:
            C_to_group[C] = rep

    # Add grouped C column to isoflop_dat
    isoflop_dat["C_group"] = isoflop_dat["C"].map(C_to_group)

    # the smallest groups are train, largest groups are test
    n_train_groups = 4
    n_test_groups = min(3, len(C_groups) - n_train_groups)
    train_group_reps = group_representatives[:n_train_groups]
    test_group_reps = group_representatives[-n_test_groups:] if n_test_groups > 0 else []

    # Use consistent color scale across train and test (continuous colors)
    all_colors = sns.color_palette(color_palette, len(group_representatives))[::-1]
    # Map each group representative to its color
    group_to_color = {rep: all_colors[i] for i, rep in enumerate(group_representatives)}

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # RIGHT PANEL: Chinchilla + LLM (actual) for test isoflops
    for group_rep in test_group_reps:
        subset = isoflop_dat[isoflop_dat["C_group"] == group_rep].copy()
        subset = subset.sort_values(by="N")
        color = group_to_color[group_rep]

        # Scatter for actual LLM loss
        axs[2].scatter(subset["N"], subset["NN_loss"], color=color, marker='o', s=40, alpha=0.8)

        # Line for Chinchilla predictions
        axs[2].plot(subset["N"], subset["chin_loss"], color=color, linestyle='--', linewidth=2.5, alpha=0.7)

        # Add label at the end of Chinchilla curve with group C value
        label_C = int(round(group_rep))
        axs[2].text(subset["N"].values[-1] * 1.05, subset["chin_loss"].values[-1],
                    f"  {label_C} PF", color=color, fontsize=10, fontweight='bold', va='center')

    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("N (Parameters)", fontweight='bold')
    axs[2].set_ylabel("Loss", fontweight='bold')
    axs[2].set_title("Heldout IsoFLOPs (Chinchilla)", fontweight='bold')

    # Sparse y-axis ticks (keep a few labels)
    from matplotlib.ticker import LogLocator
    axs[2].yaxis.set_major_locator(LogLocator(base=10, numticks=4))

    # Expand axis range for more blank space
    x_min, x_max = axs[2].get_xlim()
    y_min, y_max = axs[2].get_ylim()
    axs[2].set_xlim(x_min * 0.5, x_max * 2)
    axs[2].set_ylim(y_min * 0.95, y_max * 1.3)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['left'].set_linewidth(2)
    axs[2].spines['bottom'].set_linewidth(2)

    # INSET: Training isoflop curves (top right) - Chinchilla only
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset_right = inset_axes(axs[2], width="40%", height="40%", loc='upper right',
                          bbox_to_anchor=(-0.05, -0.05, 1, 1), bbox_transform=axs[2].transAxes)

    for group_rep in train_group_reps:
        subset = isoflop_dat[isoflop_dat["C_group"] == group_rep].copy()
        subset = subset.sort_values(by="N")
        color = group_to_color[group_rep]

        # Scatter for actual LLM loss
        ax_inset_right.scatter(subset["N"], subset["NN_loss"], color=color, marker='o', s=15, alpha=0.8)

        # Line for Chinchilla predictions
        ax_inset_right.plot(subset["N"], subset["chin_loss"], color=color, linestyle='--', linewidth=2.0, alpha=0.7)

    ax_inset_right.set_xscale("log")
    ax_inset_right.set_yscale("log")
    max_train_C = max(train_group_reps)
    max_train_C_label = int(round(max_train_C))
    ax_inset_right.set_title(f"Training IsoFLOPs < {max_train_C_label} PF", fontsize=7, fontweight='bold')
    ax_inset_right.set_xticklabels([])
    ax_inset_right.set_yticklabels([])
    ax_inset_right.tick_params(axis='both', which='both', length=0)

    # MIDDLE PANEL: NQS + LLM (actual) for test isoflops
    for group_rep in test_group_reps:
        subset = isoflop_dat[isoflop_dat["C_group"] == group_rep].copy()
        subset = subset.sort_values(by="N")
        color = group_to_color[group_rep]

        # Scatter for actual LLM loss
        axs[1].scatter(subset["N"], subset["NN_loss"], color=color, marker='o', s=40, alpha=0.8)

        # Line for NQS predictions
        axs[1].plot(subset["N"], subset["nqs_loss"], color=color, linestyle='-', linewidth=2.5, alpha=0.7)

        # Add label at the end of NQS curve with group C value
        label_C = int(round(group_rep))
        axs[1].text(subset["N"].values[-1] * 1.05, subset["nqs_loss"].values[-1],
                    f"  {label_C} PF", color=color, fontsize=10, fontweight='bold', va='center')

    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("N (Parameters)", fontweight='bold')
    axs[1].set_ylabel("Loss", fontweight='bold')
    axs[1].set_title("Heldout IsoFLOPs (NQS)", fontweight='bold')

    # Sparse y-axis ticks
    axs[1].yaxis.set_major_locator(LogLocator(base=10, numticks=4))

    # Expand axis range for more blank space
    x_min, x_max = axs[1].get_xlim()
    y_min, y_max = axs[1].get_ylim()
    axs[1].set_xlim(x_min * 0.5, x_max * 2)
    axs[1].set_ylim(y_min * 0.95, y_max * 1.3)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_linewidth(2)
    axs[1].spines['bottom'].set_linewidth(2)

    # INSET: Training isoflop curves (top right) - NQS only
    ax_inset_mid = inset_axes(axs[1], width="40%", height="40%", loc='upper right',
                          bbox_to_anchor=(-0.05, -0.05, 1, 1), bbox_transform=axs[1].transAxes)

    for group_rep in train_group_reps:
        subset = isoflop_dat[isoflop_dat["C_group"] == group_rep].copy()
        subset = subset.sort_values(by="N")
        color = group_to_color[group_rep]

        # Scatter for actual LLM loss
        ax_inset_mid.scatter(subset["N"], subset["NN_loss"], color=color, marker='o', s=15, alpha=0.8)

        # Line for NQS predictions
        ax_inset_mid.plot(subset["N"], subset["nqs_loss"], color=color, linestyle='-', linewidth=2.0, alpha=0.7)

    ax_inset_mid.set_xscale("log")
    ax_inset_mid.set_yscale("log")
    ax_inset_mid.set_title(f"Training IsoFLOPs < {max_train_C_label} PF", fontsize=7, fontweight='bold')
    ax_inset_mid.set_xticklabels([])
    ax_inset_mid.set_yticklabels([])
    ax_inset_mid.tick_params(axis='both', which='both', length=0)

    # Create custom legend entries (centered at bottom for all panels)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label='LLM (Groundtruth)'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, alpha=0.7, label='NQS'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5, alpha=0.7, label='Chinchilla'),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=3, prop={'weight': 'bold'})

    # LEFT PANEL: NBK surface
    NBK_dat = NN_dat[NN_dat["type"] == "NBK"]
    NBK_dat = NBK_dat.copy()

    # Use fixed loss levels for boundaries
    loss_levels = [2.8, 3.0, 3.5, 4.0, 4.5]

    def assign_loss_bin(x):
        """Assign to bin based on loss value."""
        for i, level in enumerate(loss_levels):
            if x <= level:
                return i
        return len(loss_levels)

    NBK_dat["NN_loss_bin"] = NBK_dat["NN_loss"].apply(assign_loss_bin)

    # Get the number of unique bins present in the data
    unique_bins = sorted(NBK_dat["NN_loss_bin"].unique())
    n_bins = len(unique_bins)

    # Create color mapping for bins
    bin_colors = sns.color_palette(color_palette, n_bins)[::-1]
    bin_to_color = {b: bin_colors[i] for i, b in enumerate(unique_bins)}

    NBK_dat["color"] = NBK_dat["NN_loss_bin"].apply(lambda x: bin_to_color[x])
    sc = axs[0].scatter(NBK_dat["B"], NBK_dat["K"], c=NBK_dat["color"], s=50, alpha=0.7)

    # For contour levels, use the fixed loss levels
    contour_levels = loss_levels
    contour_colors = sns.color_palette(color_palette, len(contour_levels))[::-1]

    # get the product of N, B, K for each row
    NBK_dat["NBK_product"] = NBK_dat["actual_N"] * NBK_dat["B"] * NBK_dat["K"]
    # get the mean of NBK_product
    NBK_product_mean = NBK_dat["NBK_product"].mean()
    # run nqs loss on each row of the NBK_dat, and save to a new column nqs_loss_value
    NBK_dat["nqs_loss"] = NBK_dat.apply(lambda row: compute_nqs_loss(row["actual_N"], row["B"], row["K"],
                                                                             1.999,
                                                                             "constant",
                                                                             {"decay_at":[], "decay_amt":[], "B_decay_amt":[]},
                                                                             nqs_model_dict), axis=1)
    # get N, B, K values from the first row
    #nqs_losss = compute_nqs_loss(8000000,768,51200, 1.999, "constant", {"decay_at":[], "decay_amt":[], "B_decay_amt":[]}, nqs_model_dict)
    # save NBK_dat to csv for debugging
    #raise ValueError(nqs_losss)
    NBK_dat.to_csv("NBK_dat_debug.csv", index=False)
    #raise ValueError(quantile_min_loss, quantile_max_loss, quantile_median_loss)

    # get the range of B and K
    B_min = NBK_dat["B"].min()
    B_max = NBK_dat["B"].max()
    K_min = NBK_dat["K"].min()
    K_max = NBK_dat["K"].max()
    # make a B-K grid with 10 points in each direction
    B_grid = np.logspace(np.log10(B_min), np.log10(B_max), 20)
    K_grid = np.logspace(np.log10(K_min), np.log10(K_max), 20)
    B_mesh, K_mesh = np.meshgrid(B_grid, K_grid)
    # compute nqs loss on the grid, N = NBK_product_mean / (B * K) to the nearest integer
    # check how many unique values of N in the NBK_dat
    unique_N_values = NBK_dat["actual_N"].unique()
    len_unique_N = len(unique_N_values)
    # if there is only one unique value of N, set N_mesh to that value
    if len_unique_N == 1:
        N_mesh = unique_N_values[0] * np.ones_like(B_mesh, dtype=int)
    else:
        N_mesh = NBK_product_mean / (B_mesh * K_mesh)
    # round N_mesh to nearest integer that can be processed by jax
    N_mesh = np.round(N_mesh).astype(int)
    #
    nqs_loss_mesh = np.zeros_like(N_mesh, dtype=float)
    for i in range(N_mesh.shape[0]):
        for j in range(N_mesh.shape[1]):
            NBK_ij = N_mesh[i, j], B_mesh[i, j], K_mesh[i, j]
            # convert to int
            NBK_ij = (int(NBK_ij[0]), int(NBK_ij[1]), int(NBK_ij[2]))
            nqs_loss_mesh[i, j] = compute_nqs_loss(NBK_ij[0],
                                                   NBK_ij[1],
                                                   NBK_ij[2],
                                                   1.999,
                                                   "constant",
                                                   {"decay_at":[], "decay_amt":[], "B_decay_amt":[]},
                                                   nqs_model_dict)

    # Plot contour lines at the loss boundaries (solid lines for NQS)
    if len(contour_levels) > 0:
        cs = axs[0].contour(B_mesh, K_mesh, nqs_loss_mesh, levels=contour_levels,
                           colors=contour_colors if len(contour_colors) == len(contour_levels) else 'gray',
                           linestyles='solid', linewidths=2.5, alpha=0.7)
        # Add labels at the rightmost part of contour lines
        for i, level in enumerate(cs.levels):
            if i < len(cs.collections):
                paths = cs.collections[i].get_paths()
                if paths:
                    # Find the rightmost point across all paths for this level
                    max_x = -np.inf
                    best_pos = None
                    for path in paths:
                        vertices = path.vertices
                        if len(vertices) > 0:
                            # Find the point with maximum x
                            max_idx = np.argmax(vertices[:, 0])
                            if vertices[max_idx, 0] > max_x:
                                max_x = vertices[max_idx, 0]
                                best_pos = (vertices[max_idx, 0], vertices[max_idx, 1])
                    if best_pos is not None:
                        axs[0].text(best_pos[0] * 1.05, best_pos[1], f"  loss < {level:.1f}",
                                   fontsize=10, fontweight='bold', va='center', ha='left',
                                   color=contour_colors[i] if i < len(contour_colors) else 'gray')

    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Batch Size B", fontweight='bold')
    axs[0].set_ylabel("Steps K", fontweight='bold')
    axs[0].set_title("Heldout B-K Plane (NQS)", fontweight='bold')

    # Expand axis range for more blank space
    x_min, x_max = axs[0].get_xlim()
    y_min, y_max = axs[0].get_ylim()
    axs[0].set_xlim(x_min * 0.5, x_max * 2)
    axs[0].set_ylim(y_min * 0.5, y_max * 2)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_linewidth(2)
    axs[0].spines['bottom'].set_linewidth(2)
    #cbar = plt.colorbar(sc, ax=axs[2])
    #cbar.set_label("log LLM Loss")

    # save figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # Make room for legend below plot
    plt.savefig("figure1.png", dpi=300)
    plt.close()

    # Save NQS cache to disk
    _save_nqs_cache()

    return None



if __name__ == "__main__":

    seq_len = 128
    vocab_size = 3000
    loss_range = (1.0, 7.0)

    nqs_model_yaml_path = "/mfs1/u/chuning/scale_new/outputs/runs/saved_yamls/fitted_nqs_adam_cosine_7_multiple_Ns.yaml"
    chin_model_yaml_path = "/mfs1/u/chuning/scale_new/outputs/runs/saved_yamls/fitted_cp_adam_cosine.yaml"

    # color palette
    colors = "magma"

    # load the yaml files into dictionaries
    with open(nqs_model_yaml_path, 'r') as f:
        nqs_model_dict = yaml.safe_load(f)
    with open(chin_model_yaml_path, 'r') as f:
        chin_model_dict = yaml.safe_load(f)
    # print the dictionaries
    print("NQS model dict:")
    print(nqs_model_dict)
    print("Chin model dict:")
    print(chin_model_dict)
    
    #nqs_loss = compute_nqs_loss(8000000,768,51200, 1.999, "constant", {"decay_at":[], "decay_amt":[], "B_decay_amt":[]}, nqs_model_dict)

    #raise ValueError(nqs_loss)

    data_file_name = "combined_data.csv"
    # if the file exists in working directory, load it, otherwise, call load_data to create it
    if False: #os.path.exists(data_file_name):
        print("Data file found, loading data...")
        combined_NN_df = pd.read_csv(data_file_name)
    else:
        print("Data file not found, creating data...")
        isoflops_path = "/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-18-00-09_openwebtext2_pythia_adam_cosine/5_resource_allocation/to_get_nn_samples.csv"
        NBK_path = "/mfs1/u/chuning/scale_new/outputs/runs/Genome/2026-01-19-20-37_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
        #"/mfs1/u/chuning/scale_new/outputs/runs/Genome/2026-01-19-04-40_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
        #04-31 norm2
        #"/mfs1/u/chuning/scale_new/outputs/runs/EPC_investigation/2025-12-15-22-49_openwebtext2_pythia/6_critical_batch_size/h_samples_with_nn_loss.csv"
        #fix N, BK "/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-18-03-10_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
        #NBK "/mfs1/u/chuning/scale_new/outputs/runs/EPC_investigation/2025-12-15-22-49_openwebtext2_pythia/6_critical_batch_size/h_samples_with_nn_loss.csv"
        #"scale_new/outputs/runs/Genome/2026-01-18-15-19_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
        #"/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-18-02-48_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
        isotokens_path = "/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-18-00-27_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
        load_NN_data(isoflops_path, NBK_path, isotokens_path)
        combined_NN_df = pd.read_csv(data_file_name)

    # test compute_chin_loss
    print("Testing compute_chin_loss...")
    chin_loss = compute_chin_loss(1e7, 32, 2048, chin_model_dict, seq_len)
    print(f"Chin loss for N=1e7, B=32, K=2048: {chin_loss}")    

    # test compute_nqs_loss
    print("Testing compute_nqs_loss...")
    nqs_loss = compute_nqs_loss(1e7, 32, 2048, 1.999, "constant", {"decay_at":[], "decay_amt":[], "B_decay_amt":[]}, nqs_model_dict)
    print(f"NQS loss for N=1e7, B=32, K=2048: {nqs_loss}")

    figure1(combined_NN_df, seq_len, color_palette=colors)