
import os
import pandas as pd
from omegaconf import OmegaConf
from a_scale.archiving_utils import archive_wrapper, df_row_to_dict
from a_scale.run_nn import train_nn, build_nn
#from a_scale.plotting_utils import deepseek_plot
import json
import numpy as np
# import timer
import time
#from a_scale.design_architecture import ModelArchitecture, remove_embedding_params, convert_to_effective_params
from a_scale.design_architecture import convert_to_effective_params

import matplotlib.pyplot as plt
from a_scale.plotting_utils import deepseek_plot


def hyper_selection(fitted_nqs,
                    neural_net,
                    nqs_simulator,
                    hyper_selection_dir,
                    nn_archive_file,
                    nqs_archive_file,
                    h_sampler_hyper_selection_deepseek,
                    h_sampler_hyper_selection_NQS,
                    h_sampler_hyper_selection_test,
                    test,
                    deepseek,
                    min_B,
                    max_B,
                    min_lr,
                    max_lr,
                    momentum,
                    end_lr,
                    lr_schedule,
                    optimizer,
                    step_decay_schedule,
                    seq_len,
                    vocab_size,

                    effective_model_size_factor,
                    include_embedding_params
                    ):
    

    def actual_N(N):
       nn_dict = dict(neural_net.copy())
       h_dict = {}
       # add key "N" to h_dict
       h_dict["N"] = int(N)
       actual_N_value = build_nn(nn_dict, h_dict)
        # get actual N using the design architecture class
       # archi = ModelArchitecture.design_model_architecture(N, vocab_size)
       # actual_N_value = ModelArchitecture.calculate_params_simplified(hidden_size=archi["hidden_size"], num_layers=archi["num_hidden_layers"]) + vocab_size * archi["hidden_size"]
       return actual_N_value

    out_dir = hyper_selection_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # run the deepseek sampler
    h_sampler_deepseek_dict = h_sampler_hyper_selection_deepseek()
    h_samples_deepseek = h_sampler_deepseek_dict['samples'] # a dataframe of hypers
    # save the dataframe h_samples_deepseek as a csv file in the out_dir
    h_samples_deepseek.to_csv(out_dir + "/h_samples_deepseek.csv", index=False)
    h_samples_deepseek = pd.read_csv(out_dir + "/h_samples_deepseek.csv")

    # run the NQS sampler
    h_sampler_NQS_dict = h_sampler_hyper_selection_NQS()
    h_samples_NQS = h_sampler_NQS_dict['samples'] # a dataframe of hypers
    # save the dataframe h_samples_NQS as a csv file in the out_dir
    h_samples_NQS.to_csv(out_dir + "/h_samples_NQS.csv", index=False)
    h_samples_NQS = pd.read_csv(out_dir + "/h_samples_NQS.csv")

    # run the test sampler
    h_sampler_test_dict = h_sampler_hyper_selection_test()
    h_samples_test = h_sampler_test_dict['samples'] # a dataframe of hypers
    # save the dataframe h_samples_test as a csv file in the out_dir
    h_samples_test.to_csv(out_dir + "/h_samples_test.csv", index=False)
    h_samples_test = pd.read_csv(out_dir + "/h_samples_test.csv")

    # collet nqs loss for each hyper
    fitted_nqs_dict = OmegaConf.to_container(fitted_nqs)
    h_samples_NQS_with_nqs_loss = h_samples_NQS.copy()


    for i in range(len(h_samples_NQS)):
        h_i_dict = df_row_to_dict(h_samples_NQS, i)
        if not include_embedding_params:
            # remove embedding params from h_i_dict
            #h_i_dict_to_run = remove_embedding_params(h_i_dict, vocab_size)
            h_i_dict_to_run = h_i_dict.copy()
        else:
            raise ValueError("include_embedding_params is not implemented yet")
            #h_i_dict_to_run = h_i_dict
            
        if effective_model_size_factor != 1.0:
            h_i_dict_to_run = convert_to_effective_params(h_i_dict, effective_model_size_factor)

        msg, nqs_out = archive_wrapper(nqs_simulator)(
                    fitted_nqs_dict, h_i_dict_to_run, nqs_archive_file)
        nqs_df_i = nqs_out["nqs_df"]
        nqs_loss_i = nqs_df_i['nqs_risk'].iloc[-1]
        h_samples_NQS_with_nqs_loss.loc[i, 'nqs_loss'] = nqs_loss_i

    # save the dataframe h_samples_NQS_with_nqs_loss as a csv file in the out_dir
    h_samples_NQS_with_nqs_loss.to_csv(out_dir + "/h_samples_NQS_with_nqs_loss.csv", index=False)
    h_samples_NQS_with_nqs_loss = pd.read_csv(out_dir + "/h_samples_NQS_with_nqs_loss.csv")

    # find the optimal hyper parameters using h_samples_NQS_with_nqs_loss
    # first sort by nqs_loss, then take the first row
    nqs_opt = h_samples_NQS_with_nqs_loss.sort_values('nqs_loss').iloc[0]
    nqs_opt = pd.DataFrame([nqs_opt])
    nqs_opt.to_csv(out_dir + "/nqs_opt.csv", index=False)

    # test the optimal hyperparameters in NN 

    h_samples_test_NQS_with_nn_loss = h_samples_test.copy()
    # update B, K, lr with the optimal values from nqs_opt
    h_samples_test_NQS_with_nn_loss["B"] = nqs_opt["B"].values[0]
    h_samples_test_NQS_with_nn_loss["K"] = nqs_opt["K"].values[0]
    h_samples_test_NQS_with_nn_loss["lr"] = nqs_opt["lr"].values[0]
    # run NN for each hyper in h_samples_test_NQS_with_nn_loss

    for i in range(len(h_samples_test_NQS_with_nn_loss)):
        h_i_dict = df_row_to_dict(h_samples_test_NQS_with_nn_loss, i)
        # -- get NN loss --
        nn_i_dict = dict(neural_net.copy())
        msg, nn_out = archive_wrapper(train_nn)(
            nn_i_dict, h_i_dict, nn_archive_file)
        nn_loss_df_i = nn_out["loss_curve_df"]
        actual_N_value = int((nn_out["actual_N_df"])["actual_N"].values[0])
        h_samples_test_NQS_with_nn_loss.loc[i, 'actual_N'] = actual_N_value
        h_samples_test_NQS_with_nn_loss.loc[i, 'NN_loss'] = nn_loss_df_i['loss'].iloc[-1]

    # save the dataframe h_samples_test_NQS_with_nn_loss as a csv file in the out_dir
    h_samples_test_NQS_with_nn_loss.to_csv(out_dir + "/h_samples_test_NQS_with_nn_loss.csv", index=False)
    h_samples_test_NQS_with_nn_loss = pd.read_csv(out_dir + "/h_samples_test_NQS_with_nn_loss.csv")


    h_samples_test_NQS_with_nn_loss["type"] = "NQS"


    # ----- Deepseek -----


    h_samples_deepseek_with_nn_loss = h_samples_deepseek.copy()

    # run NN for each hyper in h_samples_deepseek

    for i in range(len(h_samples_deepseek)):
        h_i_dict = df_row_to_dict(h_samples_deepseek, i)
        # -- get NN loss --
        nn_i_dict = dict(neural_net.copy())
        msg, nn_out = archive_wrapper(train_nn)(
            nn_i_dict, h_i_dict, nn_archive_file) # a dictionary with values dataframes
        nn_loss_df_i = nn_out["loss_curve_df"] # a dataframe with columns ckpt, loss
        actual_N_value = int((nn_out["actual_N_df"])["actual_N"].values[0]) # the actual N (may differ from requested N in h_i_dict)
        h_samples_deepseek_with_nn_loss.loc[i, 'actual_N'] = actual_N_value
        h_samples_deepseek_with_nn_loss.loc[i, 'NN_loss'] = nn_loss_df_i['loss'].iloc[-1]

    # save the dataframe h_samples_deepseek_with_nn_loss as a csv file in the out_dir
    h_samples_deepseek_with_nn_loss.to_csv(out_dir + "/h_samples_deepseek_with_nn_loss.csv", index=False)
    h_samples_deepseek_with_nn_loss = pd.read_csv(out_dir + "/h_samples_deepseek_with_nn_loss.csv")

    # compute C for each hyper in h_samples_deepseek_with_nn_loss
    h_samples_deepseek_with_nn_loss["D"] = seq_len * h_samples_deepseek_with_nn_loss["B"] * h_samples_deepseek_with_nn_loss["K"]
    h_samples_deepseek_with_nn_loss["C"] = 6 * h_samples_deepseek_with_nn_loss["N"]/ 1e9 * h_samples_deepseek_with_nn_loss["D"] 
    
    def round_significant_digit(x, digits = 2):
        x = round(x)
        return round(x, digits - len(str(int(x))))
    # keep only the top 2 significant digits of a number

    h_samples_deepseek_with_nn_loss["C"] = h_samples_deepseek_with_nn_loss["C"].apply(lambda x: round_significant_digit(x, 2))
    


    # find the optimal lr, B for each C

    def deepseek_opt_calc(h_samples_deepseek_with_nn_loss, y_axis = "B"):
        # group by C, and within each group, sort by nn_loss, retain all rows
        # Sort by NN_loss within each group
        df = h_samples_deepseek_with_nn_loss.copy()
        df = df.sort_values(by = ["C", "NN_loss"])

        # Compute NN_loss ratio within each group
        # remove na valus
        df_valid = df.copy()
        df_valid = df_valid.dropna(subset=["NN_loss"])
        # replace na in df with inf
        df = df.fillna(np.inf)
        df["NN_loss_ratio"] = df["NN_loss"] / df.groupby("C")["NN_loss"].transform("min")
        df["keep"] = df["NN_loss_ratio"] < 1.0025
        df_keep = df[df["keep"] == True].copy()
        # only keep the rows where NN_loss_ratio is less than 1.01
        
        # fit a regression line to the data, x = C, y = y_axis - log scale
        # return the function that takes in C and returns the optimal value of y_axis for that C
        x = np.log(df_keep["C"])
        y = np.log(df_keep[y_axis])
        z = np.polyfit(x, y, 1)
        f = lambda C: np.exp(np.poly1d(z)(np.log(C)))


        # plot the data and the regression line
        fig, ax = plt.subplots()
        ax.scatter(df_valid["C"], df_valid[y_axis], color = "grey", alpha = 0.5)
        ax.scatter(df_keep["C"], df_keep[y_axis], color = "green", alpha = 1.0)
        r_squared = np.corrcoef(x, y)[0, 1]**2
        # get unique C values
        C_vals = df_keep["C"].unique()

        ax.plot(C_vals, [f(C) for C in C_vals],
                color = "red",
                label = "y = " + str(round(np.exp(z[1]), 2)) + " * x^" + str(round(z[0], 2)) + "\nR^2 = " + str(round(r_squared, 2)))
        ax.legend()
        ax.set_xlabel("C")
        ax.set_ylabel(y_axis)
        ax.set_title("Optimal " + y_axis + " vs. C")
        # log scale
        ax.set_xscale("log")
        ax.set_yscale("log")


        plt.savefig(out_dir + "/deepseek_opt_" + y_axis + ".png")

        return f, df


    get_deepseek_opt_B, df_B = deepseek_opt_calc(h_samples_deepseek_with_nn_loss,
                 y_axis = "B")
    
    get_deepseek_opt_lr, df_lr = deepseek_opt_calc(h_samples_deepseek_with_nn_loss,
                 y_axis = "lr")

    deepseek_opt_B = int(get_deepseek_opt_B(test["C"]))
    deepseek_opt_lr = get_deepseek_opt_lr(test["C"])
    deepseek_opt_K = test["C"]/6/actual_N(test["N"])*1e9/seq_len/deepseek_opt_B
    deepseek_opt_K = int(round(deepseek_opt_K))


    # Test the optimal hyperparameters in NN 

    h_samples_test_deepseek_with_nn_loss = h_samples_test.copy()
    # update B, K, lr with the optimal values from deepseek_opt_B, deepseek_opt_lr
    h_samples_test_deepseek_with_nn_loss["B"] = deepseek_opt_B
    h_samples_test_deepseek_with_nn_loss["K"] = deepseek_opt_K
    h_samples_test_deepseek_with_nn_loss["lr"] = deepseek_opt_lr

    # save h_samples_test_deepseek_with_nn_loss as a csv file in the out_dir, call it deepseek_opt
    h_samples_test_deepseek_with_nn_loss.to_csv(out_dir + "/deepseek_opt.csv", index=False)
    h_samples_test_deepseek_with_nn_loss = pd.read_csv(out_dir + "/deepseek_opt.csv")
    
    # run NN for each hyper in h_samples_test_deepseek_with_nn_loss
    for i in range(len(h_samples_test_deepseek_with_nn_loss)):
        h_i_dict = df_row_to_dict(h_samples_test_deepseek_with_nn_loss, i)
        # -- get NN loss --
        nn_i_dict = dict(neural_net.copy())
        msg, nn_out = archive_wrapper(train_nn)(
            nn_i_dict, h_i_dict, nn_archive_file)
        nn_loss_df_i = nn_out["loss_curve_df"]
        actual_N_value = int((nn_out["actual_N_df"])["actual_N"].values[0])
        h_samples_test_deepseek_with_nn_loss.loc[i, 'actual_N'] = actual_N_value
        h_samples_test_deepseek_with_nn_loss.loc[i, 'NN_loss'] = nn_loss_df_i['loss'].iloc[-1]

    h_samples_test_deepseek_with_nn_loss["type"] = "deepseek"
    # save the dataframe h_samples_test_deepseek_with_nn_loss as a csv file in the out_dir
    h_samples_test_deepseek_with_nn_loss.to_csv(out_dir + "/h_samples_test_deepseek_with_nn_loss.csv", index=False)
    h_samples_test_deepseek_with_nn_loss = pd.read_csv(out_dir + "/h_samples_test_deepseek_with_nn_loss.csv")
    # combine the dataframes h_samples_test_NQS_with_nn_loss and h_samples_test_deepseek_with_nn_loss
    h_samples_test_with_nn_loss = pd.concat([h_samples_test_NQS_with_nn_loss, h_samples_test_deepseek_with_nn_loss], axis=0)
    # save the dataframe h_samples_test_with_nn_loss as a csv file in the out_dir
    h_samples_test_with_nn_loss.to_csv(out_dir + "/h_samples_test_with_nn_loss.csv", index=False)
    h_samples_test_with_nn_loss = pd.read_csv(out_dir + "/h_samples_test_with_nn_loss.csv")

    h_samples_test_with_nn_loss["D"] = h_samples_test_with_nn_loss["B"] * h_samples_test_with_nn_loss["K"] * seq_len
    h_samples_test_with_nn_loss["C"] = 6 * h_samples_test_with_nn_loss["N"]/ 1e9 * h_samples_test_with_nn_loss["D"]
    h_samples_test_with_nn_loss["C"] = h_samples_test_with_nn_loss["C"].apply(lambda x: round_significant_digit(x, 2))

    deepseek_plot(h_samples_deepseek_with_nn_loss,
                    h_samples_test_with_nn_loss,
                    deepseek_opt_calc,
                    y_axis = "B",
                    out_file = out_dir + "/deepseek_opt_B_w_NNloss.png")

    deepseek_plot(h_samples_deepseek_with_nn_loss,
                    h_samples_test_with_nn_loss,
                    deepseek_opt_calc,
                    y_axis = "lr",
                    out_file = out_dir + "/deepseek_opt_lr_w_NNloss.png")


    return None
