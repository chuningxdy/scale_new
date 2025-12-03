# fit scaling model
# save the nqs in the dir

from omegaconf import DictConfig, OmegaConf
import hydra
import pandas as pd
import jax.numpy as jnp
import os
from scipy.optimize import curve_fit
import numpy as np
from a_scale.archiving_utils import archive_wrapper, df_row_to_dict
#, create_archive
from a_scale.run_nn import train_nn
from a_scale import nn
from a_scale.nqs.nqs_utils import fit_nqs
from a_scale.design_architecture import convert_to_effective_params
#remove_embedding_params, 
from pyDOE import lhs 
from a_scale.nqs.nqs import to_x, to_nqs, NQS
import pickle

#from a_scale.sampling import h_sampler_time_compute_budget, sample_a_b_mb

# Helper function to bridge pandas.df and jnp arrays
# make sure each element in x_data is a jnp array
def series_to_jnp(series):
        series.apply(lambda x: float(x))
        return jnp.array(series.to_numpy())


# ----------------- Zeroth Order Scaling Model -----------------


def zeroth_order_scaling_model(number_of_initializations,
                                 seed,
                                 method,
                                 a_range,
                                 b_range,
                                 ma_range,
                                 mb_range,
                                 c_range,
                                 sigma_range,
                                 gtol,
                                 max_steps,
                               h_sampler_training, 
                               neural_net, 
                               scaling_model_dir, 
                               fitted_nqs_file,
                               nn_archive_file,
                               loss_threshold,
                               ignore_small_batches,
                               effective_model_size_factor,
                               include_embedding_params,                           
                               vocab_size =3000):
    
    '''
    as this function is instantiated, so is h_sampler_training and nqs_sampler_grid_search
    it remains to do:
    0. execute the samplers
    1. run neural_net for samples in h_sampler_training to get trained hypers + nn loss 
    2. for each nqs sampled by nqs_sampler_grid_search, 
        run nqs bayes,bias,var for samples in h_sampler_training to get trained nqs + nqs bayes,bias,var
    3. find the nqs whose bayes, bias, var best matches nn loss
    '''

    if method not in ["latin_hypercube"]: #, "grid_search"]:
        raise ValueError("method not implemented")

    burn_in = 2
    need_to_group_K = False
    
    # create the scaling_model_dir if it does not exist
    if not os.path.exists(scaling_model_dir):
        os.makedirs(scaling_model_dir)

    h_samples_dict = h_sampler_training()
    h_samples = h_samples_dict["samples"]
    ct_summary = h_samples_dict["ct_summary"]
    h_samples.to_csv(scaling_model_dir + "h_samples.csv", index = False)
    ct_summary.to_csv(scaling_model_dir + "training_cost.csv", index = False)
    # read the h_samples from the saved file
    h_samples = pd.read_csv(scaling_model_dir + "h_samples.csv")


    param_ranges_raw = {
        'a': (a_range[0], a_range[1]),
        'b': (b_range[0], b_range[1]),
        'ma': (ma_range[0], ma_range[1]),
        'mb': (mb_range[0], mb_range[1]),
        'c': (c_range[0], c_range[1]),
        'sigma': (sigma_range[0], sigma_range[1])
    }

    

    ls = []
    h_i_dicts = []

    # group h_samples by all columns except for K
    # give each group a unique id
    # make a dictionary with keys being the unique id and values being a list of all
    # the Ks in that group
    # make another dictionary with keys being the row number and values being the group id


    group_cols = list(h_samples.columns)
    group_cols.remove("K")

    # group by all columns except for K
    h_samples_grouped = h_samples.copy()
    h_samples_grouped["group_id"] = h_samples_grouped.groupby(group_cols).ngroup()
    # get the unique group ids
    group_ids = h_samples_grouped["group_id"].unique()
    # create a dictionary with keys being the group id and values being a list of all the Ks in that group
    group_id_dict = {}
    for group_id in group_ids:
        group_id_dict[group_id] = h_samples_grouped[h_samples_grouped["group_id"] == group_id]["K"].tolist()
    
    #raise ValueError(f"group_id_dict: {group_id_dict}")
    
    # create a dictionary with keys being the row number and values being the group id
    row_num_dict = {}
    for i in range(len(h_samples_grouped)):
        row_num_dict[i] = h_samples_grouped.iloc[i]["group_id"]

    use_ckpts = False
    
    if need_to_group_K:
        for i in range(len(h_samples)):
            h_i_dict = df_row_to_dict(h_samples, i)
            nn_i_dict = dict(neural_net.copy())

            lookup_dict = {
                            (200000, 549, 3000): (200000, 549, 1000),
                            (200000, 325, 1000): (200000, 325, 1000),
                            (2200000, 196, 5000): (2200000, 196, 4000),
                            (100000, 165, 5000): (100000, 165, 4000),
                            (1000000, 1078, 4000): (1000000, 1078, 4000),
                            (900000, 80, 5000): (900000, 80, 4000),
                            (1300000, 209, 22000): (1300000, 209, 10000),
                            (100000, 240, 17000): (100000, 240, 10000),
                            (4600000, 1625, 1000): (4600000, 1625, 1000),
                            (300000, 87, 32000): (300000, 87, 30000),
                            (100000, 63, 8000): (100000, 63, 4000),
                            (600000, 53, 87000): (600000, 53, 70000),
                            (300000, 65, 4000): (300000, 65, 4000),
                            (300000, 55, 7000): (300000, 55, 4000),
                            (2500000, 125, 7000): (2500000, 125, 4000),
                            (100000, 768, 1000): (100000, 768, 1000),
                            (400000, 129, 5000): (400000, 129, 4000),
                            (8300000, 540, 2000): (8300000, 540, 1000),
                            (1100000, 1412, 2000): (1100000, 1412, 1000),
                            (600000, 52, 4000): (600000, 52, 4000),
                            (6200000, 247, 1000): (6200000, 247, 1000),
                            (200000, 447, 2000): (200000, 447, 1000),
                            (1700000, 72, 46000): (1700000, 72, 30000),
                            (5200000, 210, 2000): (5200000, 210, 1000),
                            (1000000, 865, 1000): (1000000, 865, 1000),
                            (5900000, 2048, 2000): (5900000, 2048, 1000),
                            (200000, 341, 10000): (200000, 341, 4000),
                            (2400000, 112, 43000): (2400000, 112, 30000),
                            (2200000, 33, 3000): (2200000, 33, 1000),
                            (4500000, 44, 9000): (4500000, 44, 4000),
                            (500000, 106, 45000): (500000, 106, 30000),
                            (200000, 736, 13000): (200000, 736, 10000),
                            (900000, 47, 26000): (900000, 47, 10000),
                            (2500000, 87, 70000): (2500000, 87, 70000),
                            (200000, 139, 11000): (200000, 139, 4000),
                            (800000, 1502, 2000): (800000, 1502, 1000),
                            (700000, 890, 2000): (700000, 890, 1000),
                            (3300000, 306, 17000): (3300000, 306, 10000),
                            (400000, 511, 1000): (400000, 511, 1000),
                            (200000, 79, 4000): (200000, 79, 4000),
                            (300000, 324, 12000): (300000, 324, 10000),
                            (100000, 51, 15000): (100000, 51, 10000),
                            (2300000, 176, 4000): (2300000, 176, 4000),
                            (300000, 74, 46000): (300000, 74, 30000),
                            (500000, 54, 19000): (500000, 54, 10000),
                            (1200000, 54, 10000): (1200000, 54, 4000),
                            (800000, 227, 1000): (800000, 227, 1000),
                            (3900000, 955, 8000): (3900000, 955, 4000),
                            (100000, 42, 54000): (100000, 42, 30000),
                            (200000, 57, 5000): (200000, 57, 4000),
                            (700000, 1440, 3000): (700000, 1440, 1000),
                            (300000, 1065, 1000): (300000, 1065, 1000),
                            (1600000, 381, 4000): (1600000, 381, 4000),
                            (500000, 874, 4000): (500000, 874, 4000),
                            (700000, 541, 17000): (700000, 541, 10000),
                            (100000, 99, 26000): (100000, 99, 10000),
                            (7700000, 134, 2000): (7700000, 134, 1000),
                            (7300000, 39, 67000): (7300000, 39, 30000),
                            (400000, 148, 27000): (400000, 148, 10000),
                            (7000000, 55, 2000): (7000000, 55, 1000),
                            (1100000, 41, 3000): (1100000, 41, 1000),
                            (500000, 740, 7000): (500000, 740, 4000),
                            (4900000, 215, 1000): (4900000, 215, 1000),
                            (200000, 429, 1000): (200000, 429, 1000),
                            (1900000, 86, 43000): (1900000, 86, 30000),
                            (3700000, 236, 1000): (3700000, 236, 1000),
                            (1600000, 136, 12000): (1600000, 136, 10000),
                            (2700000, 486, 7000): (2700000, 486, 4000),
                            (8200000, 1223, 2000): (8200000, 1223, 1000),
                            (800000, 46, 11000): (800000, 46, 4000),
                            (400000, 1007, 1000): (400000, 1007, 1000),
                            (400000, 120, 33000): (400000, 120, 30000),
                            (8600000, 148, 4000): (8600000, 148, 4000),
                            (100000, 938, 3000): (100000, 938, 1000),
                            (800000, 33, 58000): (800000, 33, 30000),
                            (600000, 816, 2000): (600000, 816, 1000),
                            (200000, 44, 32000): (200000, 44, 30000),
                            (500000, 133, 28000): (500000, 133, 10000),
                            (2100000, 203, 19000): (2100000, 203, 10000),
                            (6200000, 40, 25000): (6200000, 40, 10000),
                            (600000, 140, 15000): (600000, 140, 10000),
                            (1500000, 53, 5000): (1500000, 53, 4000),
                            (5800000, 398, 6000): (5800000, 398, 4000),
                            (5300000, 407, 11000): (5300000, 407, 4000),
                            (1300000, 46, 89000): (1300000, 46, 70000),
                            (200000, 116, 4000): (200000, 116, 4000),
                            (300000, 109, 26000): (300000, 109, 10000),
                            (1600000, 294, 19000): (1600000, 294, 10000),
                            (4900000, 43, 12000): (4900000, 43, 10000),
                            (800000, 168, 33000): (800000, 168, 30000),
                            (7000000, 92, 6000): (7000000, 92, 4000),
                            (200000, 132, 5000): (200000, 132, 4000),
                            (4600000, 45, 46000): (4600000, 45, 30000),
                            (3300000, 597, 15000): (3300000, 597, 10000),
                        }

            # create a reverse lookup dict
            reverse_lookup_dict = {v: k for k, v in lookup_dict.items()}
            


            # update the K value in the h_i_dict with the value from the reverse lookup dict
            # first check if the K value is in the lookup dict
            to_get_K = int(h_i_dict["K"])
            to_lookup_tuple = (h_i_dict["N"], h_i_dict["B"], h_i_dict["K"])
            if to_lookup_tuple in reverse_lookup_dict:
                h_i_dict_to_get_nn = h_i_dict.copy()
                h_i_dict_to_get_nn["K"] = reverse_lookup_dict[to_lookup_tuple][2]
            else:
                raise ValueError(f"to_lookup_tuple not in lookup_dict: {to_lookup_tuple}")

            msg, nn_out = archive_wrapper(train_nn)(
                    nn_i_dict, h_i_dict_to_get_nn, nn_archive_file)
            nn_loss_df_i = nn_out["loss_curve_df"]
            actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0])
            nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] >= burn_in]
            # remove duplicates
            nn_loss_df_i = nn_loss_df_i.drop_duplicates()
            

            # ckpts_to_get_nqs is the max ckpt in the nn_loss_df_i
            #ckpts_to_get_nqs = list(nn_loss_df_i["ckpt"].values)


            ckpts_to_get_nqs = [to_get_K]
            #raise ValueError(f"ckpts_to_get_nqs: {ckpts_to_get_nqs}")
        
        
            for cpt in ckpts_to_get_nqs:
                    nn_loss_i_ckpt = nn_loss_df_i[nn_loss_df_i["ckpt"] == cpt]
                    h_i_dict_ckpt = h_i_dict.copy()
                    h_i_dict_ckpt.update({"K": int(cpt)})
                    # add nn loss to ls
                    ls.append(nn_loss_i_ckpt["loss"].values[0])
                    h_i_dicts.append(h_i_dict_ckpt)
    
    elif not use_ckpts:
        for i in range(len(h_samples)):
            h_i_dict = df_row_to_dict(h_samples, i)
            nn_i_dict = dict(neural_net.copy())
            msg, nn_out = archive_wrapper(train_nn)(
                    nn_i_dict, h_i_dict, nn_archive_file)
            nn_loss_df_i = nn_out["loss_curve_df"]
            actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0])
            nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] >= burn_in]
            # remove duplicates
            nn_loss_df_i = nn_loss_df_i.drop_duplicates()
            

            # ckpts_to_get_nqs is the max ckpt in the nn_loss_df_i
            ckpts_to_get_nqs = list(nn_loss_df_i["ckpt"].values)
            ckpts_to_get_nqs = [max(ckpts_to_get_nqs)]
            #raise ValueError(f"ckpts_to_get_nqs: {ckpts_to_get_nqs}")
        
        
            for cpt in ckpts_to_get_nqs:
                    nn_loss_i_ckpt = nn_loss_df_i[nn_loss_df_i["ckpt"] == cpt]
                    h_i_dict_ckpt = h_i_dict.copy()
                    h_i_dict_ckpt.update({"K": int(cpt)})

                    # check if loss is a valid number (not NaN or inf)
                    if not np.isfinite(nn_loss_i_ckpt["loss"].values[0]):
                        # add the dict and loss to a list of invalid losses
                        invalid_loss_dict = {
                            "h_i_dict": h_i_dict_ckpt,
                            "nn_loss": nn_loss_i_ckpt["loss"].values[0],
                            "actual_N": actual_N
                        }
                        # add the invalid loss dict to a list
                        if "invalid_losses" not in locals():
                            invalid_losses = []
                        invalid_losses.append(invalid_loss_dict)
                    
                    else:
                        # add nn loss to ls
                        ls.append(nn_loss_i_ckpt["loss"].values[0])
                        h_i_dicts.append(h_i_dict_ckpt)
    
    else:
        for i in range(len(h_samples)):
            # look up the group id for the row number i
            group_id_i = row_num_dict[i]
            # look up the list of Ks for the group id
            group_id_K_list = group_id_dict[group_id_i]
            # convert to a list of ints
            group_id_K_list = [int(x) for x in group_id_K_list]
            #raise ValueError(f"group_id_K_list: {group_id_K_list}")
            # check if the K value of the row number i is the maximum in the list
            is_max_K = int(h_samples.iloc[i]["K"]) == max(group_id_K_list)
            # if it is not the maximum, skip this row
            if is_max_K:
                
                h_i_dict = df_row_to_dict(h_samples, i)
                nn_i_dict = dict(neural_net.copy())
                msg, nn_out = archive_wrapper(train_nn)(
                    nn_i_dict, h_i_dict, nn_archive_file)
                nn_loss_df_i = nn_out["loss_curve_df"]
                actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0])
                nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] >= burn_in]
                # remove duplicates
                nn_loss_df_i = nn_loss_df_i.drop_duplicates()

                
                # get the checkpoints that are in the group_id_K_list
                nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"].isin(group_id_K_list)]
                #nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] == nn_loss_df_i["ckpt"].max()]
                # check if the ckpts contains all the Ks in the group_id_K_list

                # drop index
                nn_loss_df_i = nn_loss_df_i.reset_index(drop=True)

            
                if not set(group_id_K_list).issubset(set(nn_loss_df_i["ckpt"].values)):
                    raise ValueError("not all Ks in the group_id_K_list are in the nn_loss_df_i: " + str(group_id_K_list) + " " + str(nn_loss_df_i["ckpt"].values))
                
                ckpts_to_get_nqs = list(nn_loss_df_i["ckpt"].values)
                use_actual_N_for_nqs = True
                if use_actual_N_for_nqs:
                    h_i_dict.update({"N": actual_N})

                for cpt in ckpts_to_get_nqs:
                    nn_loss_i_ckpt = nn_loss_df_i[nn_loss_df_i["ckpt"] == cpt]
                    h_i_dict_ckpt = h_i_dict.copy()
                    h_i_dict_ckpt.update({"K": int(cpt)})
                    # add nn loss to ls
                    ls.append(nn_loss_i_ckpt["loss"].values[0])
                    h_i_dicts.append(h_i_dict_ckpt)
    
    # save invalid losses to a json file
    if "invalid_losses" in locals():
        import json
        with open(scaling_model_dir + "invalid_losses.json", "w") as f:
            json.dump(invalid_losses, f, indent=4)
    #    raise ValueError("invalid losses found, saved to " + scaling_model_dir + "invalid_losses.json")
    # count the number of h_i_dicts
    num_h_i_dicts = len(h_i_dicts)
    #raise ValueError(f"num_h_i_dicts: {num_h_i_dicts}")
    
    if not include_embedding_params:
        # loop thru h_i_dicts and remove embedding params
        h_i_dicts_to_run = h_i_dicts.copy()
        for k in range(len(h_i_dicts)):
            h_i_dicts_to_run[k] = h_i_dicts[k].copy() #remove_embedding_params(h_i_dicts[k], vocab_size)

    else:
        raise ValueError("include_embedding_params is not implemented yet")
    
    if effective_model_size_factor != 1.0:
        # loop thru h_i_dicts and scale the model size
        h_i_dicts_to_run = h_i_dicts.copy()
        for k in range(len(h_i_dicts)):
            h_i_dicts_to_run[k] = convert_to_effective_params(h_i_dicts[k], effective_model_size_factor)


    # filter out where loss is greater than threshold
    #loss_threshold = 4.0
    
    h_i_dicts_to_run = [h_i_dicts_to_run[i] for i in range(len(h_i_dicts_to_run)) if ls[i] < loss_threshold]
    ls = [x for x in ls if x < loss_threshold]

    # filter out isotoken runs where batch size are much smaller than optimal
    # do this by: sort h_i_dicts_to_run by compute budget, 
    # then sort by batch size (from largest to smallest),
    # if the loss start increasing for two consecutive batch sizes, start filtering out the runs with smaller batch sizes
    if ignore_small_batches:
        # create a list same size as h_i_dicts_to_run,
        # this will be the mask for filtering out runs
        filter_mask = [False] * len(h_i_dicts_to_run)
        
        # add ls to the h_i_dicts_to_run into each dict
        for i in range(len(h_i_dicts_to_run)):
            h_i_dicts_to_run[i]["loss"] = ls[i]
        # within each model size and compute budget, sort by batch size (from largest to smallest)
        h_i_dicts_to_run = sorted(h_i_dicts_to_run, key=lambda x: (x["N"], x["C"], x["lr"], x["B"]), reverse=True)
        # traverse the list, at each new compute budget, start a new counter
        # the counter accounts for the number of consecutive runs where the loss is increasing
        # if the counter reaches 2, set filter_mask to True for all runs in that compute budget
        current_compute_budget = None
        current_model_size = None
        current_lr = None
        consecutive_increasing_count = 0
        for i in range(len(h_i_dicts_to_run)):
            
            h_i_dict = h_i_dicts_to_run[i]
            # reset the counter if the compute budget changes or N changes
            if current_compute_budget is None or h_i_dict["C"] != current_compute_budget or h_i_dict["N"] != current_model_size or h_i_dict["lr"] != current_lr:
                # reset the counter
                consecutive_increasing_count = 0
                current_compute_budget = h_i_dict["C"]
                current_model_size = h_i_dict["N"]
                current_lr = h_i_dict["lr"]
            # check if the loss is increasing compared to the previous run
            elif h_i_dict["loss"] > h_i_dicts_to_run[i - 1]["loss"] - 0.05:
                consecutive_increasing_count += 1
            
            # if the counter reaches 2, set filter_mask to True for all runs in that compute budget
            if consecutive_increasing_count >= 1:
                filter_mask[i] = True
    
        # redefine ls using the sorted h_i_dicts_to_run
        ls = []
        # pop out ls in h_i_dicts_to_run for each dict
        for i in range(len(h_i_dicts_to_run)):
            ls.append(h_i_dicts_to_run[i].pop("loss"))

        # combine filtered_mask, h_i_dicts_to_run, and ls into a data frame and save it to a csv file
        filtered_df = pd.DataFrame({
            "h_i_dict": h_i_dicts_to_run,
            "loss": ls,
            "filter_mask": filter_mask
        })
        filtered_df.to_csv(scaling_model_dir + "filtered_h_i_dicts.csv", index=False)
        #raise ValueError("filtered out runs with small batch sizes, saved to " + scaling_model_dir + "filtered_h_i_dicts.csv")
    
        # remove h_i_dicts_to_run and ls where filter_mask is True
        h_i_dicts_to_run = [h_i_dicts_to_run[i] for i in range(len(h_i_dicts_to_run)) if not filter_mask[i]]
        ls = [ls[i] for i in range(len(ls)) if not filter_mask[i]]

        
    # raise ValueError with the number filtered out, as well as the number of h_i_dicts_to_run and the number of ls
    num_filtered_out = num_h_i_dicts - len(h_i_dicts_to_run)
    if num_filtered_out > 0:
        print(f"filtered out {num_filtered_out} h_i_dicts with loss greater than {loss_threshold}, "
                         f"remaining {len(h_i_dicts_to_run)} h_i_dicts to run, and {len(ls)} nn losses")
    else:
        print(f"no h_i_dicts filtered out, remaining {len(h_i_dicts_to_run)} h_i_dicts to run, and {len(ls)} nn losses")
    
    
    # save ls to a csv file
    ls_df = pd.DataFrame({"loss": ls})
    ls_df.to_csv(scaling_model_dir + "nn_losses.csv", index=False)
    # check if ls_df contains any NaN or inf values
    if ls_df.isnull().values.any() or np.isinf(ls_df.values).any():
        raise ValueError("nn losses contain NaN or inf values, saved to " + scaling_model_dir + "nn_losses.csv")
    

    

    #raise ValueError("saved nn losses to a csv file at " + scaling_model_dir + "nn_losses.csv")
    #raise ValueError("h_i_dicts_to_run: " + str(h_i_dicts_to_run) )
    
    # where h_dicts_to_run is adamw optimizer,
    # set lr to 2.0
    for h_i_dict in h_i_dicts_to_run:
        if h_i_dict["optimizer"] == "adamw":
            h_i_dict["lr"] = 1.999 #1.999
    
    # raise ValueError with the h_i_dicts_to_run 
    #raise ValueError(f"h_i_dicts_to_run: {h_i_dicts_to_run}")

    fitted_nqs_dict, eval_metric_value, trajectories_formatted = fit_nqs(
         h_dicts = h_i_dicts_to_run,
         nn_losses = ls, 
         seed = seed,
         number_of_initializations = number_of_initializations,
         param_ranges_raw = param_ranges_raw,
         gtol = gtol,
         max_steps = max_steps)
                                                

    # save the best nqs to the fitted_nqs_file
    OmegaConf.save(fitted_nqs_dict, fitted_nqs_file)

    # save the trajectories formatted into a pickle file
    with open(scaling_model_dir + "trajectories_formatted.pickle", "wb") as f:
        pickle.dump(trajectories_formatted, f)

    # read fitted_nqs from the saved file
    fitted_nqs = OmegaConf.load(fitted_nqs_file)
    # save the trajectories to the scaling_model_dir
    # save the formatted trajectories to the scaling_model_dir
    # return the best fitted nqs
    return fitted_nqs, eval_metric_value

            

 