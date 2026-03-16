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


def compute_chin_loss(N,B,K,model_dict, seq_len):
    """Compute the chin loss given N, B, K and model_dict"""
    # extract parameters from model_dict
    A = model_dict['A']
    B = model_dict['B']
    E = model_dict['E']
    N_power = model_dict['N_power']
    D_power = model_dict['D_power']
    D = seq_len * B * K 
    loss = E + A/(N**N_power) + B/((D**D_power))
    
    return loss

def compute_nqs_loss(N,B,K,init_lr,lr_sch_type, sch, model_dict, use_LRA = True):
    """Compute the nqs loss given N, B, K, sch and model_dict"""
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
    return out_dict['nqs_df']['nqs_risk'].values[0]



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
    #isoflop_dat["nqs_loss"] = isoflop_dat.apply(lambda row: compute_nqs_loss(row["actual_N"], row["B"], row["K"], 
     #                                                                        1.999, 
     #                                                                        "constant", 
     #                                                                        {"decay_at":[], "decay_amt":[], "B_decay_amt":[]}, 
      #                                                                       nqs_model_dict), axis=1)
    isoflop_dat["nqs_loss"] = isoflop_dat["NN_loss"] # placeholder
    # add a column chin_loss_value
    isoflop_dat["chin_loss"] = isoflop_dat.apply(lambda row: compute_chin_loss(row["actual_N"], row["B"], row["K"], 
                                                                             chin_model_dict, seq_len), axis=1)
    # save isoflop_dat to csv for debugging
    isoflop_dat.to_csv("isoflop_dat_debug.csv", index=False)
    # group by C
    # the color should be by grouped C values
    Cs = sorted(isoflop_dat["C"].unique().tolist())
    colors = sns.color_palette(color_palette, len(Cs))

    # the smallest 4 C values are the train set, and the largest 3 C values are the test set
    train_Cs = Cs[:4]
    test_Cs = Cs[-3:]
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # for each C, scatter the NN loss vs D and draw a line for nqs_loss vs D
    for i, C in enumerate(Cs):
        if C in train_Cs or C in test_Cs:
            subset = isoflop_dat[isoflop_dat["C"] == C].copy()
            axs[0].scatter(subset["N"], subset["NN_loss"], label=f"C={C} PFLOPS", color=colors[i])
            # sort by N
            subset = subset.sort_values(by="N")
            axs[0].plot(subset["N"], subset["nqs_loss"], color=colors[i], linestyle="-")
            # for the nqs_loss line, put a text label at the end, where the label is C value
            if C in train_Cs: # round C to 2 significant figures
                # check if C is the largest in train_Cs
                if C == max(train_Cs):
                    label_text = f"{C} PFLOPS (train)"
                elif C == min(train_Cs):
                    label_text = f"{C} PFLOPS (train)"
                else:
                    label_text = ""
            elif C in test_Cs:
                if C == max(test_Cs):
                    label_text = f"{C} PFLOPS (test)"
                elif C == min(test_Cs):
                    label_text = f"{C} PFLOPS (test)"
                else:
                    label_text = ""
            else:
                label_text = ""
            axs[0].text(subset["N"].values[-1], subset["nqs_loss"].values[-1], label_text, color=colors[i])
            axs[0].plot(subset["N"], subset["chin_loss"], color=colors[i], linestyle=":")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("N (Parameters)")
    axs[0].set_ylabel("NN Loss")
    axs[0].set_title("Isoflops Scaling")
    #axs[0].legend()
    # right panel: NBK surface
    NBK_dat = NN_dat[NN_dat["type"] == "NBK"]
    NBK_dat = NBK_dat.copy()
    # look at all NN_loss values and percentile them the [0,0.25], (0.25,0.75], (0.75,1.0] quantiles
    NN_loss_values = NBK_dat["NN_loss"].values  
    quantiles = np.percentile(NN_loss_values, [25,50,75])
    def assign_quantile(x):
        # make it general for any number of quantiles!
        for i, q in enumerate(quantiles):
            if x <= q:
                return i
        return len(quantiles)
    NBK_dat["NN_loss_quantile"] = NBK_dat["NN_loss"].apply(assign_quantile)
    # for the row with minimum NN_loss, assign quantile 0
    min_loss_idx = NBK_dat["NN_loss"].idxmin()
    min_loss = NBK_dat.at[min_loss_idx, "NN_loss"]
    #NBK_dat.at[min_loss_idx, "NN_loss_quantile"] = 0

    # find the min loss within each quantile
    quantile_min_loss = NBK_dat.groupby("NN_loss_quantile")["NN_loss"].min()
    quantile_max_loss = NBK_dat.groupby("NN_loss_quantile")["NN_loss"].max()
    # find the median loss within each quantile
    quantile_median_loss = NBK_dat.groupby("NN_loss_quantile")["NN_loss"].median()
    # group by quantile and assign a color from the color palette
    quantile_colors = sns.color_palette(color_palette, len(quantile_median_loss))
    NBK_dat["color"] = NBK_dat["NN_loss_quantile"].apply(lambda x: quantile_colors[x])
    sc = axs[1].scatter(NBK_dat["B"], NBK_dat["K"], c=NBK_dat["color"], s=50, alpha=0.7)

    # get the product of N, B, K for each row
    NBK_dat["NBK_product"] = NBK_dat["actual_N"] * NBK_dat["B"] * NBK_dat["K"]
    # get the mean of NBK_product
    NBK_product_mean = NBK_dat["NBK_product"].mean()
    # run nqs loss on each row of the NBK_dat, and save to a new column nqs_loss_value
    NBK_dat["nqs_loss"] = NBK_dat.apply(lambda row: compute_nqs_loss(row["N"], row["B"], row["K"], 
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
    contour_levels_min = quantile_min_loss
    contour_levels_max = quantile_max_loss
    # let contour levels be the midpoints of these ranges (geometric mean)
    contour_levels = quantile_min_loss

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
    # plot contour lines of nqs_loss_mesh at the percentiles of NN_loss_values
    # find the range of nqs_loss_mesh
    nqs_loss_min = nqs_loss_mesh.min()
    nqs_loss_max = nqs_loss_mesh.max()

    #raise ValueError(contour_levels, nqs_loss_min, nqs_loss_max, min_loss, max(NN_loss_values))

    # let the color be the color for the quantile

    cs = axs[1].contour(B_mesh, K_mesh, nqs_loss_mesh, levels=contour_levels, colors=quantile_colors, linestyles='dashed')
    #axs[1].clabel(cs, inline=True, fontsize=10, fmt="NQS: %.2f")

    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Batch Size B")
    axs[1].set_ylabel("Sequence Length K")
    axs[1].set_title("NBK Scaling")
    #cbar = plt.colorbar(sc, ax=axs[1])
    #cbar.set_label("log NN Loss")

    # save figure   
    plt.tight_layout()
    plt.savefig("figure1.png", dpi=300)
    plt.close()

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