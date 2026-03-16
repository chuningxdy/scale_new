
import os
import pandas as pd
from omegaconf import OmegaConf
from a_scale.archiving_utils import archive_wrapper, df_row_to_dict
from a_scale.run_nn import train_nn
from a_scale.plotting_utils import actual_vs_fitted_plot
from a_scale.plotting_utils import plot_loss_curves
import json
import numpy as np
from a_scale.design_architecture import convert_to_effective_params

#remove_embedding_params, 

def loss_estimation(neural_net, 
                    a_scaling_model,
                    chinchilla_1,
                    chinchilla_2,
                    h_sampler_train,
                    h_sampler_train_chinchilla,
                    h_sampler_test,
                    nqs_simulator,
                    eval_metric,
                    loss_estimation_dir, 
                    fitted_nqs,
                    fitted_baselines,
                    #fitted_nqs_file,
                    nn_archive_file,
                    nqs_archive_file,
                    test_last_ckpt_only,
                    effective_model_size_factor,
                    include_embedding_params,
                    loss_threshold,
                    ignore_small_batches,
                    vocab_size = 3000):
    
    burn_in = 2
    need_to_group_K = False


    if not os.path.exists(loss_estimation_dir):
        os.makedirs(loss_estimation_dir)

    # start a tab to record the eval metrics
    # this is a data frame with columns: model_name, train_eval_metric, test_eval_metric
    eval_metrics_df = pd.DataFrame(columns = ["model_name", "train_eval_metric", "test_eval_metric"])

    
    # get training and test samples of h
    h_samples_train_dict = h_sampler_train()
    h_samples_train = h_samples_train_dict["samples"]
    h_samples_train.to_csv(loss_estimation_dir + "h_samples_train.csv", index = False)
    h_samples_train = pd.read_csv(loss_estimation_dir + "h_samples_train.csv")

    h_samples_test_dict = h_sampler_test()
    h_samples_test = h_samples_test_dict["samples"]
    ct_summary_test = h_samples_test_dict["ct_summary"]
    h_samples_test.to_csv(loss_estimation_dir + "h_samples_test.csv", index = False)
    ct_summary_test.to_csv(loss_estimation_dir + "test_run_cost.csv", index = False)
    h_samples_test = pd.read_csv(loss_estimation_dir + "h_samples_test.csv")
    


    h_samples_chinchilla_dict = h_sampler_train_chinchilla()
    h_samples_chinchilla = h_samples_chinchilla_dict["samples"]
    h_samples_chinchilla.to_csv(loss_estimation_dir + "h_samples_chinchilla.csv", index = False)
    h_samples_chinchilla = pd.read_csv(loss_estimation_dir + "h_samples_chinchilla.csv")


    
    # --------------- NQS Loss Estimation ----------------
    #raise ValueError("NQS loss estimation is not implemented yet")
    fitted_nqs_dict = OmegaConf.to_container(fitted_nqs)

    nqs_lookup_dict = fitted_nqs_dict #= {k: fitted_nqs_dict[k] for k in ['a', 'b', 'm_b']}
    

    # for each row in h_samples, get the nn loss
    started = False
    for j, h_samples in enumerate([h_samples_train, h_samples_test]):

        if j == 0:
            h_samples_type = "train"
        else:
            h_samples_type = "test"
            #raise ValueError("test has these many rows: " + str(len(h_samples)))
        
        group_cols = list(h_samples.columns)
        # if D in group_cols, and there are nan values in D, remove D from group_cols
        if "D" in group_cols:
            if h_samples["D"].isna().any():
                group_cols.remove("D")
        # do same for actual_N
        if "actual_N" in group_cols:
            if h_samples["actual_N"].isna().any():
                group_cols.remove("actual_N")
        #group_cols.remove("K")

        # group by all columns except for K
        h_samples_grouped = h_samples.copy()
        h_samples_grouped["group_id"] = h_samples_grouped.groupby(group_cols).ngroup()
        # get the unique group ids
        group_ids = h_samples_grouped["group_id"].unique()
        #if h_samples_type == "test":
        #    raise ValueError(h_samples_grouped[group_cols].isna().any(axis=1).sum())
        
        # create a dictionary with keys being the group id and values being a list of all the Ks in that group
        group_id_dict = {}
        for group_id in group_ids:
            group_id_dict[group_id] = h_samples_grouped[h_samples_grouped["group_id"] == group_id]["K"].tolist()
        
        #raise ValueError(f"group_id_dict: {group_id_dict}")
        
        # create a dictionary with keys being the row number and values being the group id
        row_num_dict = {}
        for ii in range(len(h_samples_grouped)):
            row_num_dict[ii] = h_samples_grouped.iloc[ii]["group_id"]
        

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
                
                # convert the ith row of h_samples to a dictionary
                h_i_dict = df_row_to_dict(h_samples, i)
                # h_samples.iloc[i].to_dict()
                #raise ValueError("h_i_dict: " + str(h_i_dict) +"h_samples_i: " + str(h_samples.iloc[i]))
                # -- get NN loss --
                nn_i_dict = dict(neural_net.copy())
                #raise ValueError("nn_i_dict: " + str(nn_i_dict))
                #if h_samples_type == "train": raise ValueError("nn_i_dict: " + str(nn_i_dict) + "\n" + "h_i_dict: " + str(h_i_dict))
                
                
                if h_samples_type == "train" and need_to_group_K: 
                    #if h_samples_type == "test": raise ValueError("getting NN_loss")
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
                    to_lookup_tuple = (int(h_i_dict["N"]), int(h_i_dict["B"]), int(h_i_dict["K"]))
                    if to_lookup_tuple in reverse_lookup_dict:
                        h_i_dict_to_get_nn = h_i_dict.copy()
                        h_i_dict_to_get_nn["K"] = reverse_lookup_dict[to_lookup_tuple][2]
                    else:
                        raise ValueError(f"Tuple {to_lookup_tuple} not in lookup dict")


                    msg, nn_out = archive_wrapper(train_nn)(
                            nn_i_dict, h_i_dict_to_get_nn, nn_archive_file)
                    
                elif h_samples_type == "test" or not need_to_group_K:
                    #if h_samples_type == "train": raise ValueError("getting NN_loss")
                    #h_i_dict_to_get_nn = h_i_dict.copy()
                    #h_i_dict_to_get_nn["K"] = int(h_i_dict["K"])
                    msg, nn_out = archive_wrapper(train_nn)(
                            nn_i_dict, h_i_dict, nn_archive_file)  
                    #raise ValueError("h_i_dict: " + str(h_i_dict) )
                #if h_samples_type == "test": raise ValueError("nn_out: " + str(nn_out))
                #raise ValueError("nn_out: " + str(nn_out))
                nn_loss_df_i = nn_out["loss_curve_df"] # a dataframe with columns ckpt, loss
                actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0]) # the actual N (may differ from requested N in h_i_dict)
                nn_loss_df_i["actual_N"] = actual_N
                nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] > burn_in]
                # remove duplicates
                nn_loss_df_i = nn_loss_df_i.drop_duplicates()

                if h_samples_type == "test":
                    nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] == nn_loss_df_i["ckpt"].max()]
                    #raise ValueError("nn_loss_df_i: " + str(nn_loss_df_i))
                if h_samples_type == "train":
                    # get the checkpoints that are in the group_id_K_list
                    #nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"].isin(group_id_K_list)]
                    if need_to_group_K:
                        nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"].isin([to_get_K])]
                    else:
                        nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] == nn_loss_df_i["ckpt"].max()]
                    # check if the ckpts contains all the Ks in the group_id_K_list

                    # drop index
                    nn_loss_df_i = nn_loss_df_i.reset_index(drop=True)
                
                    if False:
                        if not set(group_id_K_list).issubset(set(nn_loss_df_i["ckpt"].values)):
                         #raise ValueError(str(nn_loss_df_i))
                            raise ValueError("not all Ks in the group_id_K_list are in the nn_loss_df_i: " + str(group_id_K_list) + " " + str(nn_loss_df_i["ckpt"].values))
                    
                ckpts_to_get_nqs = list(nn_loss_df_i["ckpt"].values)
                
                #if h_samples_type == "train":
                    # filter nn_loss_df_i for ckpts in [8000, 80000, 800000]
                    #nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"].isin([8000, 80000, 800000])]
                    # get the last ckpt
                #    nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] == nn_loss_df_i["ckpt"].max()]
                
                #if h_samples_type == "test":
                    # keep the last ckpt
                #    nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] == nn_loss_df_i["ckpt"].max()]
                
                ckpts_to_get_nqs = list(nn_loss_df_i["ckpt"].values)

                # -- get NQS loss --
                #raise ValueError("NQS loss estimation is not implemented yet")
                use_actual_N_for_nqs = False
                if use_actual_N_for_nqs:
                    h_i_dict.update({"N": actual_N})
                
                
                # empty dataframe
                

                for cpt in ckpts_to_get_nqs:
                    h_i_dict.update({"K": int(cpt)})
                    # run nqs bayes,bias,var for the ith sample
                    if not include_embedding_params:
                        #if h_i_dict["N"] <= 250000:
                        #    raise ValueError("h_i_dict: " + str(h_i_dict))
                        # remove embedding params from h_i_dict
                        h_i_dict_to_run = h_i_dict.copy() #remove_embedding_params(h_i_dict, vocab_size)
                    
                    else:
                        raise ValueError("include_embedding_params is not implemented yet")
                        #h_i_dict_to_run = h_i_dict.copy()
                    if effective_model_size_factor != 1.0:
                        h_i_dict_to_run = convert_to_effective_params(h_i_dict_to_run, effective_model_size_factor)

                    #raise ValueError("h_i_dict_to_run: " + str(h_i_dict_to_run))
                    msg, nqs_out = archive_wrapper(nqs_simulator)(
                        nqs_lookup_dict, h_i_dict_to_run, nqs_archive_file) # a dictionary with values dataframes
                    if cpt == ckpts_to_get_nqs[0]:
                        nqs_bbv_df_i = nqs_out["nqs_df"] # a dataframe with columns nqs_iter, bayes_risk, nqs_bias, nqs_var
                    else:
                        nqs_bbv_df_i = pd.concat([nqs_bbv_df_i, nqs_out["nqs_df"]], axis = 0)
                    #raise ValueError("nqs_bbv_df_i: " + str(nqs_bbv_df_i))
           
           
                #raise ValueError("done getting nqs")
                # -- merge NN and NQS loss --
                eval_df_i = pd.merge(nn_loss_df_i, nqs_bbv_df_i, 
                                        left_on = 'ckpt', right_on = 'nqs_iter', how = 'inner')
                eval_df_i.drop(columns = ['nqs_iter'], inplace = True)
                eval_df_i["h_samples_type"] = h_samples_type

                # if test set, print nn_loss_df_i and nqs_bbv_df_i in error
                #if h_samples_type == "test":
                #    raise ValueError("nn_loss_df_i: " + str(nn_loss_df_i) + "\n" + "nqs_bbv_df_i: " + str(nqs_bbv_df_i))
                
                #raise ValueError("eval_df_i: " + str(eval_df_i))
                # add h_i_dict to eval_df_i
                for key in h_i_dict.keys():
                    valuee = h_i_dict[key]
                    # if valuee is a dictionary, convert it to a string
                    if isinstance(valuee, dict):
                        valuee = json.dumps(valuee)
                    eval_df_i[key] = valuee

                
                # if eval_df does not exist, let it be eval_df_i
                
                if not started:
                    eval_df = eval_df_i
                    started = True
                else:
                    eval_df = pd.concat([eval_df, eval_df_i], axis = 0)
                print("i: " + str(i) + " j: " + str(j) + " h_samples_type: " + h_samples_type)
                

    # get len of eval_df
    #raise ValueError("len(eval_df): " + str(len(eval_df)))
        #raise ValueError("eval_df: " + str(eval_df))

   



    eval_df["nqs_loss"] = eval_df["nqs_risk"] #eval_df.apply(lambda row: calc_nqs_from_bbv(row, fitted_nqs_dict), axis = 1)
    eval_df.to_csv(loss_estimation_dir + "eval_df.csv", index = False)

    # filter out rows with loss greater than loss_threshold
    eval_df = eval_df[eval_df["loss"] < loss_threshold].copy()
    
    # within each model size and compute budget, sort by batch size (from largest to smallest)
    eval_df = eval_df.sort_values(by = ["N", "C", "B"], ascending = [True, False, False]).reset_index(drop = True)
    # at each new N, C, set a counter to zero
    # if the loss is at least the previous line's loss minus 0.01, add one to the counter
    eval_df["loss_counter"] = 0
    for row in range(len(eval_df)):
        if row == 0:
            eval_df.at[row, "loss_counter"] = 0
        # if new N, C, reset the counter to 0
        elif eval_df.at[row, "N"] != eval_df.at[row - 1, "N"] or eval_df.at[row, "C"] != eval_df.at[row - 1, "C"]:
                eval_df.at[row, "loss_counter"] = 0
        else:
            if eval_df.at[row, "loss"] >= eval_df.at[row - 1, "loss"] - 0.05:
                eval_df.at[row, "loss_counter"] = eval_df.at[row - 1, "loss_counter"] + 1
            else:
                eval_df.at[row, "loss_counter"] = 0
    
    # save the eval_df to a csv file
    eval_df.to_csv(loss_estimation_dir + "eval_df_counter.csv", index = False)
    #raise ValueError("check eval_df_counter.csv for the loss counter")

    # delete rows with loss_counter > 0
   # eval_df = eval_df[eval_df["loss_counter"] == 0].copy()
    # drop the loss_counter column
    eval_df.drop(columns = ["loss_counter"], inplace = True)

    # get loss curves
    #loss_curve_plot_file = loss_estimation_dir + "loss_curves.png"
    #plot_loss_curves(eval_df, loss_curve_plot_file)


    eval_df_test = eval_df[eval_df["h_samples_type"] == "test"]
    #if test_last_ckpt_only:
        # for the test set, keep only the last ckpt
        #eval_df_test = eval_df_test[eval_df_test["ckpt"] == eval_df_test["ckpt"].max()]
    

    
    eval_df_train = eval_df[eval_df["h_samples_type"] == "train"]
    
    
        
    if eval_df_test.shape[0] == 0:
        eval_df_test = eval_df_train[0:2].copy()
        eval_df_test["h_samples_type"] = "test"
        #raise ValueError("eval_df_test is:" + str(eval_df_test))
        #raise ValueError("eval_df_test is empty")

    eval_df = pd.concat([eval_df_train, eval_df_test], axis = 0)
    # raise an error that lists the h_samlpes_type in eval_df
    #raise ValueError("h_samples_type in eval_df: " + str(eval_df["h_samples_type"].unique()))

    actual_values = eval_df[eval_df["h_samples_type"] == "train"]["loss"].values
    fitted_values = eval_df[eval_df["h_samples_type"] == "train"]["nqs_loss"].values
    # the colors for train - use 1 for cases where N == 10000000, 0 otherwise
    #train_colors = np.where(eval_df[eval_df["h_samples_type"] == "train"]["B"] == 128, 1, 0)
    train_colors = np.where(eval_df[eval_df["h_samples_type"] == "train"]["N"] == 8419936, 1, 0)
    #raise ValueError("train_colors: " + str(train_colors) + "\n" + "eval_df_N: " + str(eval_df["N"]))

    test_actual_values = eval_df[eval_df["h_samples_type"] == "test"]["loss"].values
    test_colors = np.where(eval_df[eval_df["h_samples_type"] == "test"]["N"] == 8419936, 1, 0)
    test_fitted_values = eval_df[eval_df["h_samples_type"] == "test"]["nqs_loss"].values
    # the colors for test - use 1 for cases where N == 10000000, 0 otherwise
   # test_colors = np.where(eval_df[eval_df["h_samples_type"] == "test"]["B"] == 128, 1, 0)

    train_eval_metric = eval_metric(actual_values, fitted_values)
    test_eval_metric = eval_metric(test_actual_values, test_fitted_values)
    # add the eval metrics to the eval_metrics_df; use concat
    evals = pd.DataFrame({"model_name": "nqs",
                          "train_eval_metric": train_eval_metric,
                          "test_eval_metric": test_eval_metric}, 
                          index = [0])
    eval_metrics_df = pd.concat([eval_metrics_df, evals], axis = 0)
    
    output_file = loss_estimation_dir + "actual_vs_fitted.png"

    

    actual_vs_fitted_plot(actual_values, fitted_values, output_file,
                          test_actual_values = test_actual_values, test_fitted_values = test_fitted_values,
                          train_colors =train_colors, test_colors = test_colors)
    

  
    # -------------- Baselines ----------------


    # look through the dict fitted_baselines
    # for each item in fitted_baselines

    for baseline_name, fitted_baseline in fitted_baselines.items():
        
        # create a subdir with the key
        baseline_dir = loss_estimation_dir + baseline_name + "/"
        if not os.path.exists(baseline_dir):
            os.makedirs(baseline_dir)

        if baseline_name == "chinchilla_2":
            h_samples = h_samples_chinchilla
        elif baseline_name == "chinchilla_1":
            h_samples = h_samples_train


        # make sure the values of actual_N is populated for h_samples
        # if there are nan values, populate them with the value from N
        #h_samples["actual_N"] = h_samples.apply(lambda row: row["N"] if pd.isna(row["actual_N"]) else row["actual_N"], axis = 1)
        # raise an error if h_samples["actual_N"] still has nan values
        # raise an error if h_samples["N"] still has nan values



        group_cols = list(h_samples.columns)
        #group_cols.remove("K")

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
        for ii in range(len(h_samples_grouped)):
            row_num_dict[ii] = h_samples_grouped.iloc[ii]["group_id"]
        

        
        started = False
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
                    # convert the ith row of h_samples to a dictionary
                    h_i_dict = df_row_to_dict(h_samples, i)
                    #h_samples.iloc[i].to_dict()
                    nn_i_dict = dict(neural_net.copy())
                    #nn_cfg.update({'h': h_i_cfg})
                    msg, nn_out = archive_wrapper(train_nn)(
                        nn_i_dict, h_i_dict, nn_archive_file) # a dictionary with values dataframes
                    
                    nn_loss_df_i = nn_out["loss_curve_df"] # a dataframe with columns ckpt, loss
                    actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0]) # the actual N (may differ from requested N in h_i_dict)
                    nn_loss_df_i["actual_N"] = actual_N
                    nn_loss_df_i["h_samples_type"] = "train"
                    # filter for ckpt larger than burn_in
                    nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] > burn_in].copy()
                    
                    # only keep the last ckpt
                    #nn_loss_df_i = nn_loss_df_i.tail(1).copy()

                    # get the last ckpt, ignore nas
                    last_ckpt = nn_loss_df_i["ckpt"].max()
                    # append the last_ckpt to the group_id_K_list and call it group_id_K_list_lu
                    group_id_K_list_lu = group_id_K_list.copy()
                    group_id_K_list_lu.append(last_ckpt)
                    # remove duplicates
                    group_id_K_list_lu = list(set(group_id_K_list_lu))
                    #raise ValueError("group_id_K_list" + str(group_id_K_list) + " last_ckpt: " + str(last_ckpt) + " group_id_K_list_lu: " + str(group_id_K_list_lu))
                    # get the checkpoints that are in the group_id_K_list
                    nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"].isin(group_id_K_list_lu)]
                    #nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] == nn_loss_df_i["ckpt"].max()]
                    # check if the ckpts contains all the Ks in the group_id_K_list

                    # drop index
                    nn_loss_df_i = nn_loss_df_i.reset_index(drop=True)
                
                    if not set(group_id_K_list).issubset(set(nn_loss_df_i["ckpt"].values)):
                        # if group_id_K_list contains only one element which is 37500, and nn_loss_df_i["ckpt"].values contains only one element which is 37000
                        # let pass, and set group_id_K_list to nn_loss_df_i["ckpt"].values
                        if len(group_id_K_list) == 1 and len(nn_loss_df_i["ckpt"].values) == 1 and group_id_K_list[0] == 37500 and nn_loss_df_i["ckpt"].values[0] == 37000:
                            pass
                        else:
                            raise ValueError("not all Ks in the group_id_K_list are in the nn_loss_df_i: " + str(group_id_K_list) + " " + str(nn_loss_df_i["ckpt"].values))
                    
                    # add h_i_dict to eval_df_i, by looping through the keys of h_i_dict and adding them as columns
                    for key in h_i_dict.keys():
                        #nn_loss_df_i[key] = h_i_dict[key]
                        # add a column to nn_loss_df_i with the key as the column name
                        # set every row to the same value which is h_i_dict[key]
                        nn_loss_df_i.loc[:, key] = h_i_dict[key]

                    if not started:
                        nn_loss_df = nn_loss_df_i
                        started = True
                    else:
                        nn_loss_df = pd.concat([nn_loss_df, nn_loss_df_i], axis = 0)
                
        # in eval_df, replace the rows with h_samples_type = "train" with
        #  the rows in nn_loss_df 
        eval_df_tst = eval_df[eval_df["h_samples_type"] == "test"]
       # rows_with_10M = eval_df_tst[eval_df_tst["N"] == 10000000]
        #raise ValueError("rows_with_10M: " + str(rows_with_10M))
        eval_df = pd.concat([nn_loss_df, eval_df_tst], axis = 0)

        if eval_df["N"].isna().any():
            raise ValueError("h_samples has nan values in N column")
        if eval_df["actual_N"].isna().any():
            # set actual_N to N where actual_N is nan
            eval_df["actual_N"] = eval_df.apply(lambda row: row["N"] if pd.isna(row["actual_N"]) else row["actual_N"], axis = 1)
        # display rows with N == 10000000
        #rows_with_10M = eval_df[eval_df["N"] == 10000000]
        #raise ValueError("rows_with_10M: " + str(rows_with_10M))

        eval_df["D"] = eval_df["ckpt"] * eval_df["B"] * 128
        fitted_baseline_dict = OmegaConf.to_container(fitted_baseline)
        # predict loss with the fitted baseline (chinchilla)
        def chin_predict_loss(row, fitted_baseline_dict):
            # raise an error that prints the type of actual_N and D
            # if actual_N or D is not int or float
            if not (isinstance(row["actual_N"], int) or isinstance(row["actual_N"], float)):
                raise ValueError("actual_N: " + str(type(row["actual_N"])) + "D: " + str(type(row["D"])) +
                                 "\n" + str(row))
                                 
            if row["actual_N"] <= 0 or row["D"] <= 0:
                return None
            E = fitted_baseline_dict["E"]
            A = fitted_baseline_dict["A"]
            alpha = fitted_baseline_dict["N_power"]
            B = fitted_baseline_dict["B"]
            beta = fitted_baseline_dict["D_power"]

            
            #    raise ValueError("row: " + str(row) + "return_Value" + str(E + A / row["actual_N"]**alpha + B / row["D"]**beta))
            return E + A / row["actual_N"]**alpha + B / row["D"]**beta
        
        
        eval_df["chin_loss"] = eval_df.apply(lambda row: chin_predict_loss(row, fitted_baseline_dict), axis = 1)
        # save the eval_df to a csv file
        eval_df.to_csv(baseline_dir + "eval_df.csv", index = False)

        actual_values = eval_df[eval_df["h_samples_type"] == "train"]["loss"].values
        train_colors = np.where(eval_df[eval_df["h_samples_type"] == "train"]["N"] ==8419936 , 1, 0)
        
        test_actual_values = eval_df[eval_df["h_samples_type"] == "test"]["loss"].values
        test_colors = np.where(eval_df[eval_df["h_samples_type"] == "test"]["N"] == 8419936, 1, 0)
       # raise ValueError("test_colors: " + str(test_colors))

        chin_fitted_values = eval_df[eval_df["h_samples_type"] == "train"]["chin_loss"].values
        chin_fitted_values_test = eval_df[eval_df["h_samples_type"] == "test"]["chin_loss"].values

        chin_train_eval_metric = eval_metric(actual_values, chin_fitted_values)
        chin_test_eval_metric = eval_metric(test_actual_values, chin_fitted_values_test)






        # add the eval metrics to the eval_metrics_df
        evals = pd.DataFrame({"model_name": baseline_name,
                              "train_eval_metric": chin_train_eval_metric,
                              "test_eval_metric": chin_test_eval_metric}, 
                              index = [0])
        eval_metrics_df = pd.concat([eval_metrics_df, evals], axis = 0)
        
        output_file = baseline_dir + "actual_vs_fitted.png"


        actual_vs_fitted_plot(actual_values, chin_fitted_values, output_file,
                          test_actual_values = test_actual_values, test_fitted_values = test_fitted_values,
                          train_colors =train_colors, test_colors = test_colors)
        
        actual_vs_fitted_plot(actual_values, chin_fitted_values, output_file,
                              test_actual_values = test_actual_values, 
                              test_fitted_values = chin_fitted_values_test,
                              train_colors = train_colors,
                              test_colors = test_colors)

    
    # save the eval_metrics_df to a csv file
    eval_metrics_df.to_csv(loss_estimation_dir + "eval_metrics_df.csv", index = False)
    return eval_metrics_df
    