
import os
import pandas as pd
from omegaconf import OmegaConf
from a_scale.archiving_utils import archive_wrapper, df_row_to_dict
from a_scale.run_nn import train_nn, build_nn
import json
import numpy as np
# import timer
import time
#from a_scale.design_architecture import ModelArchitecture, remove_embedding_params, convert_to_effective_params
from a_scale.design_architecture import convert_to_effective_params

import matplotlib.pyplot as plt
from a_scale.plotting_utils import IsoX_plot

def round_significant_digit(x, digits = 2):
        # find out how many digits are in the number by logging it base 10
        if x == 0:
            return 0
        else:
            return round(x, digits - int(np.floor(np.log10(abs(x)))) - 1)
        
def critical_batch_size(slope_at_critical_batch_size,
                    fitted_nqs,
                    neural_net,
                    nqs_simulator,
                    critical_batch_size_dir,
                    nn_archive_file,
                    nqs_archive_file,
                    h_sampler,

                    seq_len,
                    vocab_size,

                    effective_model_size_factor,
                    include_embedding_params,
                    use_lr_adapt,
                    lr_adapt_tolerance
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

    out_dir = critical_batch_size_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # run the test sampler
    h_sampler_dict = h_sampler()
    h_samples = h_sampler_dict['samples'] # a dataframe of hypers
    # save the dataframe h_samples_test as a csv file in the out_dir
    h_samples.to_csv(out_dir + "/h_samples.csv", index=False)
    h_samples = pd.read_csv(out_dir + "/h_samples.csv")
    

    # ------------- Collet NQS Loss for each hyper --------------

    fitted_nqs_dict = OmegaConf.to_container(fitted_nqs)
    h_samples_with_nqs_loss = h_samples.copy()


    for i in range(len(h_samples_with_nqs_loss)):
        h_i_dict = df_row_to_dict(h_samples_with_nqs_loss, i)
        if not include_embedding_params:
            # remove embedding params from h_i_dict
            #h_i_dict_to_run = remove_embedding_params(h_i_dict, vocab_size)
            h_i_dict_to_run = h_i_dict.copy()
        else:
            raise ValueError("include_embedding_params is not implemented yet")
            #h_i_dict_to_run = h_i_dict
            
        if effective_model_size_factor != 1.0:
            h_i_dict_to_run = convert_to_effective_params(h_i_dict, effective_model_size_factor)
        
        if use_lr_adapt:
            #pass
            h_i_dict_to_run["lr_schedule"] = "optimized"
            h_i_dict_to_run["step_decay_schedule"] = "na"
        
        #raise ValueError(h_i_dict_to_run)

        msg, nqs_out = archive_wrapper(nqs_simulator)(
                    fitted_nqs_dict, h_i_dict_to_run, nqs_archive_file)
        nqs_df_i = nqs_out["nqs_df"]
        nqs_loss_i = nqs_df_i['nqs_risk'].iloc[-1]
        h_samples_with_nqs_loss.loc[i, 'nqs_loss'] = nqs_loss_i

    # save the dataframe h_samples_NQS_with_nqs_loss as a csv file in the out_dir
    h_samples_with_nqs_loss.to_csv(out_dir + "/h_samples_with_nqs_loss.csv", index=False)
    h_samples_with_nqs_loss = pd.read_csv(out_dir + "/h_samples_with_nqs_loss.csv")



    # ----- Run NN for each hyper in h_samples -----------------

    h_samples_with_nn_loss = h_samples_with_nqs_loss.copy()

    for i in range(len(h_samples_with_nn_loss)):
        h_i_dict = df_row_to_dict(h_samples_with_nn_loss, i)
        # -- get NN loss --
        nn_i_dict = dict(neural_net.copy())
        msg, nn_out = archive_wrapper(train_nn)(
            nn_i_dict, h_i_dict, nn_archive_file) # a dictionary with values dataframes
        nn_loss_df_i = nn_out["loss_curve_df"] # a dataframe with columns ckpt, loss
        # drop duplicates in the nn_loss_df_i dataframe
        nn_loss_df_i = nn_loss_df_i.drop_duplicates()
        actual_N_value = int((nn_out["actual_N_df"])["actual_N"].values[0]) # the actual N (may differ from requested N in h_i_dict)
        h_samples_with_nn_loss.loc[i, 'actual_N'] = actual_N_value
        h_samples_with_nn_loss.loc[i, 'NN_loss'] = nn_loss_df_i['loss'].iloc[-1]

    # save the dataframe h_samples_with_nn_loss as a csv file in the out_dir
    h_samples_with_nn_loss.to_csv(out_dir + "/h_samples_with_nn_loss.csv", index=False)
    h_samples_with_nn_loss = pd.read_csv(out_dir + "/h_samples_with_nn_loss.csv")

    # compute C for each hyper in h_samples_with_nn_loss
    h_samples_with_nn_loss["D"] = seq_len * h_samples_with_nn_loss["B"] * h_samples_with_nn_loss["K"]
    h_samples_with_nn_loss["C"] = 6 * h_samples_with_nn_loss["N"]/ 1e9 * h_samples_with_nn_loss["D"] 
    
    def round_significant_digit(x, digits = 2):
        x = round(x)
        return round(x, digits - len(str(int(x))))
    # keep only the top 2 significant digits of a number

    h_samples_with_nn_loss["C"] = h_samples_with_nn_loss["C"].apply(lambda x: round_significant_digit(x, 2))
    
    # save the dataframe h_samples_with_nn_loss as a csv file in the out_dir
    h_samples_with_nn_loss.to_csv(out_dir + "/h_samples_with_nn_loss.csv", index=False)


    # calculate D, C
    h_samples_with_nn_loss["D"] = seq_len * h_samples_with_nn_loss["B"] * h_samples_with_nn_loss["K"]
    h_samples_with_nn_loss["C"] = 6 * h_samples_with_nn_loss["N"]/ 1e9 * h_samples_with_nn_loss["D"]

    # round C to 2 significant digits
    h_samples_with_nn_loss["C"] = h_samples_with_nn_loss["C"].apply(lambda x: round_significant_digit(x, 2))



    IsoX_plot(h_samples_with_nn_loss, 
              x_axis_name = 'B',
              seq_len = seq_len, models_to_plot = ['nqs'], 
              output_file = out_dir + "/IsoX_plot.png")
    

    return None
