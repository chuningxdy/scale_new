
import os
import pandas as pd
from omegaconf import OmegaConf
from a_scale.archiving_utils import archive_wrapper, df_row_to_dict
from a_scale.run_nn import train_nn, build_nn
from a_scale.plotting_utils import risk_contour_plot, isoflop_plot
import json
import numpy as np
# import timer
import time
from a_scale.design_architecture import convert_to_effective_params
#remove_embedding_params, 

def round_to_significant(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(np.floor(np.log10(abs(x))))-1)
    
def resource_allocation(h_sampler_nn,
                        h_sampler_nqs,
                        fitted_nqs, 
                        fitted_baselines,
                        seq_len,
                        neural_net,
                        nqs_simulator,
                        resource_allocation_dir,
                        nn_archive_file,
                        nqs_archive_file,
                        B,
                        lr, 
                        momentum,
                        end_lr,
                        lr_schedule,
                        optimizer,
                        step_decay_schedule,
                        max_iters_per_run,
                        effective_model_size_factor,
                        include_embedding_params,

                        use_lr_adapt,
                        lr_adapt_tolerance,
                        vocab_size = 3000
                        ):



    def get_actual_N(N):
       nn_dict = dict(neural_net.copy())
       h_dict = {}
       # add key "N" to h_dict
       h_dict["N"] = int(N)
       actual_N_value = build_nn(nn_dict, h_dict)
        # get actual N using the design architecture class
       # archi = ModelArchitecture.design_model_architecture(N, vocab_size)
       # actual_N_value = ModelArchitecture.calculate_params_simplified(hidden_size=archi["hidden_size"], num_layers=archi["num_hidden_layers"]) + vocab_size * archi["hidden_size"]
       return actual_N_value
    
    out_dir = resource_allocation_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # time the h_sampler
    #start = time.time()
    h_samples_dict = h_sampler_nqs()
    h_samples = h_samples_dict['samples'] # a dataframe of hypers
    # round column C to 3 significant figures
    h_samples['C'] = h_samples['C'].apply(lambda x: round_to_significant(x, 3))
    # round to nearest 10000
    h_samples["C"] = h_samples["C"].apply(lambda x: round(x, -4))
    h_samples["C"] = h_samples["C"].replace(58900000.0, 59000000.0)
    # get unique values of C where type is 'random_isoflop'
    #unique_C_values = h_samples[h_samples['type'] == 'random_isoflop']['C'].unique()
    #raise ValueError("STOP HERE: unique C values: ", unique_C_values)
    # save the dataframe h_samples as a csv file in the out_dir
    
   

    # check if h_samples has a column 'type', if not, add it with value 'random_isoflop'
    if 'type' not in h_samples.columns:
        h_samples['type'] = 'random_isoflop'
    # get the unique C values where  'random_isoflop'
    unique_C_values = h_samples[h_samples['type'] == 'random_isoflop']['C'].unique()
    #raise ValueError("STOP HERE: unique C values: ", unique_C_values)
    h_samples.to_csv(out_dir + "/h_samples.csv", index=False)
    h_samples = pd.read_csv(out_dir + "/h_samples.csv")
    # remove duplicates from the data frame, remember to reset the index
    h_samples = h_samples.drop_duplicates()
    h_samples = h_samples.reset_index(drop=True)
    #end = time.time()
    #raise ValueError("STOP HERE: time to sample hypers: ", end - start)

    # collet nqs loss for each hyper
    fitted_nqs_dict = OmegaConf.to_container(fitted_nqs)

    h_samples_with_nqs_loss = h_samples.copy()
    for i in range(len(h_samples)):
        h_i_dict = df_row_to_dict(h_samples, i)

        # -- compute actual_N only if needed
      #  actual_N = h_i_dict.get("N") * 1.2 ## FOR NOW!!!
        
        # -- get nqs loss
      #  use_actual_N_for_nqs = True
       # if use_actual_N_for_nqs:
       #     h_i_dict.update({"N": actual_N})
        
        if not include_embedding_params:
            # remove embedding params from h_i_dict
            h_i_dict_to_run = h_i_dict.copy()
            #h_i_dict_to_run = remove_embedding_params(h_i_dict, vocab_size)
        else:
            raise ValueError("include_embedding_params is not implemented yet")
            #h_i_dict_to_run = h_i_dict.copy()
            
        if effective_model_size_factor != 1.0:
            # convert to effective params
            h_i_dict_to_run = convert_to_effective_params(h_i_dict_to_run, effective_model_size_factor)

        if use_lr_adapt:
            #pass
            h_i_dict_to_run["lr_schedule"] = "optimized"
            h_i_dict_to_run["step_decay_schedule"] = "na"
            
        msg, nqs_out = archive_wrapper(nqs_simulator)(
                    fitted_nqs_dict, h_i_dict_to_run, nqs_archive_file)
        
     #   h_samples_with_nqs_loss.loc[i, 'actual_N'] = actual_N
        nqs_df_i = nqs_out["nqs_df"]
        nqs_loss_i = nqs_df_i['nqs_risk'].iloc[-1]
        h_samples_with_nqs_loss.loc[i, 'nqs_loss'] = nqs_loss_i
    
    # save the dataframe h_samples_with_nqs_loss as a csv file in the out_dir
    h_samples_with_nqs_loss["D"] = h_samples_with_nqs_loss["B"] * h_samples_with_nqs_loss["K"] * seq_len
    h_samples_with_nqs_loss.to_csv(out_dir + "/h_samples_with_nqs_loss.csv", index=False)
    h_samples_with_nqs_loss = pd.read_csv(out_dir + "/h_samples_with_nqs_loss.csv")


    opts = {}
    # find optimal allocation for nqs
    # keep only type == 'random_isoflop'
    iso_samples = h_samples_with_nqs_loss.copy()
    iso_samples = h_samples_with_nqs_loss[h_samples_with_nqs_loss['type'] == 'random_isoflop']
    # group by C, and within each group, sort by nqs_loss, keep the first row
    nqs_opts = iso_samples.sort_values('nqs_loss').groupby('C').first().reset_index()
    nqs_opts["N_raw"] = nqs_opts["N"]
    nqs_opts["K_raw"] = nqs_opts["K"]
    # iterate over the rows of nqs_opts, 
    for i in range(len(nqs_opts)):
        # replace N with actual_N
        N_raw = nqs_opts.loc[i, 'N_raw']
        actual_N_value = get_actual_N(N_raw)
        K_raw = nqs_opts.loc[i, 'K_raw']
        updated_K = K_raw * N_raw / actual_N_value
        nqs_opts.loc[i, 'N'] = nqs_opts.loc[i, 'N_raw'] #int(actual_N_value)
        nqs_opts.loc[i, 'K'] = nqs_opts.loc[i, 'K_raw'] #int(updated_K)

    nqs_opts["D"] = nqs_opts["B"] * nqs_opts["K"] * seq_len

    # add nqs: nqs_opts to opts_dict
    opts['nqs'] = nqs_opts

    # get the unique C values
    C_vals = nqs_opts['C'].unique()

    # get NN loss for some isoflop samples
    # filter for rows where N is close to nqs_opts['N']
    # for now, let the range be 900K+

    to_get_nn_samples_dict = h_sampler_nn()
    to_get_nn_samples = to_get_nn_samples_dict['samples'] # a dataframe of hypers
    # save the dataframe to_get_nn_samples as a csv file in the out_dir
    to_get_nn_samples.to_csv(out_dir + "/to_get_nn_samples.csv", index=False)
    to_get_nn_samples = pd.read_csv(out_dir + "/to_get_nn_samples.csv")
    # remove duplicates from the data frame, remember to reset the index
    to_get_nn_samples = to_get_nn_samples.drop_duplicates()
    to_get_nn_samples = to_get_nn_samples.reset_index(drop=True)
    # remove the rows where N is not in the range of 900K+ and C is in C_vals
    to_get_nn_samples = to_get_nn_samples[
       # (to_get_nn_samples['N'] > 900000) # & (to_get_nn_samples['N'] < 4000000)
        #& 
        (to_get_nn_samples['C'].isin(C_vals))]
    #to_get_nn_samples = iso_samples.copy()

    # reset index
    #to_get_nn_samples = to_get_nn_samples.reset_index(drop=True)
    #to_get_nn_samples = to_get_nn_samples[
     #   (to_get_nn_samples['N'] > 900000) # & (to_get_nn_samples['N'] < 4000000)
     #   & (to_get_nn_samples['C'].isin(C_vals))]
    # reset index
    to_get_nn_samples = to_get_nn_samples.reset_index(drop=True)
    
    # run the nn for each of these samples
    # for each row in to_get_nn_samples, run a NN with corresponding hyperparameters N, K, B
    for kk in range(len(to_get_nn_samples)):
        # convert the ith row of h_samples to a dictionary
        h_i_dict = df_row_to_dict(to_get_nn_samples, kk)
        # check if K > 1mil, if so raise an error
        if h_i_dict['K'] > max_iters_per_run:
            # move to the next row
            to_get_nn_samples.loc[kk, 'NN_loss'] = np.nan
            continue
        
        
            # -- get NN loss --
        nn_i_dict = dict(neural_net.copy())
        msg, nn_out = archive_wrapper(train_nn)(
            nn_i_dict, h_i_dict, nn_archive_file)
        nn_loss_df_i = nn_out["loss_curve_df"] # a dataframe with columns ckpt, loss
        # remove duplicates from the dataframe
        nn_loss_df_i = nn_loss_df_i.drop_duplicates()
        actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0]) # the actual N (may differ from requested N in h_i_dict)
        #if actual_N == 979072.0:
        #     raise ValueError("kk: ", kk, "\n to_get_nn_samples: ", to_get_nn_samples)
        to_get_nn_samples.loc[kk, 'actual_N'] = actual_N
        to_get_nn_samples.loc[kk, 'NN_loss'] = nn_loss_df_i['loss'].iloc[-1]

    # save the dataframe to_get_nn_samples as a csv file in the out_dir
    to_get_nn_samples.to_csv(out_dir + "/to_get_nn_samples.csv", index=False)
    to_get_nn_samples = pd.read_csv(out_dir + "/to_get_nn_samples.csv")

           


    # for each C, find the optimal params in chin
    # baselines is a dict of dicts {chin1: fitted_chin1, chin2: fitted_chin2}
    # loop over the baselines
    for key, value in fitted_baselines.items():
        # call a function that takes in the value (cp_cfg_dict) and for each C, outputs the optimal hyperparameters
        chin_opts = pd.DataFrame()
        for C in C_vals:
            C_chin_allocation = chin_allocation(value, C, B, seq_len)
            # convert the dict to a dataframe
            C_chin_allocation = pd.DataFrame([C_chin_allocation])
            chin_opts = pd.concat([chin_opts, C_chin_allocation], axis=0)
        opts[key] = chin_opts
    
    #raise ValueError("STOP HERE")
    for mod, opt in opts.items():
        # drop index and reset the index
        #print("mod: ", mod)
        #print("opt: ", opt)
        
        opt_samples = opt.copy()
        # add columns for lr, momentum, end_lr, lr_schedule, optimizer, step_decay_schedule
        opt_samples['lr'] = lr
        opt_samples['momentum'] = momentum
        opt_samples['end_lr'] = end_lr
        opt_samples['lr_schedule'] = lr_schedule
        opt_samples['optimizer'] = optimizer

        # convert step_decay_schedule to a dictionary of lists
        step_decay_schedule = {key: list(value) for key, value in step_decay_schedule.items()}
        opt_samples['step_decay_schedule'] = json.dumps(step_decay_schedule)

        # save opt with file name mod + "_opt_samples.csv"
        opt_samples.to_csv(out_dir + "/" + mod + "_opt_samples.csv", index=False)
       
        # round C to 3 significant figures
        opt_samples["C"] = opt_samples["C"].apply(lambda x: round_to_significant(x, 3))
        # for each row in nqs_opts, run a NN with corresponding hyperparameters N, K, B
        for i in range(len(opt_samples)):
            # convert the ith row of h_samples to a dictionary
                opt_samples = opt_samples.reset_index(drop=True)
                opts[mod] = opt_samples
                #raise ValueError("opt_samples: ", opt_samples)
                h_i_dict = df_row_to_dict(opt_samples, i)
                # check if K > 1mil, if so raise an error
                if h_i_dict['K'] > max_iters_per_run:
                    # move to the next row
                    opt_samples.loc[i, 'NN_loss'] = np.nan
                    continue
                
                # -- get NN loss --
              #  nn_i_dict = dict(neural_net.copy())
              #  msg, nn_out = archive_wrapper(train_nn)(
              #      nn_i_dict, h_i_dict, nn_archive_file) # a dictionary with values dataframes
              #  nn_loss_df_i = nn_out["loss_curve_df"] # a dataframe with columns ckpt, loss
                # remove duplicates from the dataframe
              #  nn_loss_df_i = nn_loss_df_i.drop_duplicates()
              #  actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0]) # the actual N (may differ from requested N in h_i_dict)
              #  opt_samples.loc[i, 'actual_N'] = actual_N
              #  opt_samples.loc[i, 'NN_loss'] = nn_loss_df_i['loss'].iloc[-1]

    # save the dataframe as a csv file 
   # h_samples.to_csv("h_samples.csv", index=False)
   # h_samples = pd.read_csv("h_samples.csv")


    grid_samples = h_samples_with_nqs_loss[h_samples_with_nqs_loss['type'] == 'grid']
    isoflop_samples = h_samples_with_nqs_loss[h_samples_with_nqs_loss['type'] == 'random_isoflop']


    # save the opts dataframe to csv
    opt_allocations_file = os.path.join(out_dir, "opt_allocations.csv")
    # convert the dictionary to a dataframe, where the key becomes a column, 
    # and the values are the rest of the columns
    opt_df = pd.DataFrame()

    for key, value in opts.items():
        value['type'] = key
        opt_df = pd.concat([opt_df, value], axis=0)
    opt_df.to_csv(opt_allocations_file, index=False)
    #raise ValueError("STOP HERE")


    # plot the risk contour plot
    #outfile = os.path.join(out_dir, "isoloss_plot.png")
    #risk_contour_plot(grid_samples, seq_len, opts, outfile)

    # check if fitted_baseline has key "chinchilla_2"
    if "chinchilla_2" in fitted_baselines:
        chin_loss_from_N, chin_loss_from_D = calc_chin_loss_from_NK(
            fitted_baselines["chinchilla_2"], seq_len)
    elif "chinchilla_1" in fitted_baselines:
        chin_loss_from_N, chin_loss_from_D = calc_chin_loss_from_NK(
            fitted_baselines["chinchilla_1"], seq_len)
    else:
        chin_loss_from_N, chin_loss_from_D = None, None
    
    # plot the isoflop plot
    outfile = os.path.join(out_dir, "isoflop_plot.png")
    isoflop_plot(isoflop_samples, seq_len, opts, outfile, 
                 x_axis="N",
                 to_get_nn_samples=to_get_nn_samples,
                 get_chin_loss_from_x =chin_loss_from_N)

    return None

   

def calc_chin_loss_from_NK(cp_cfg, seq_len):
    # get the params from the chinchilla config
    alpha = cp_cfg["N_power"]
    beta = cp_cfg["D_power"]
    A = cp_cfg["A"]
    B = cp_cfg["B"]
    E = cp_cfg["E"]

    def chin_loss_from_N(C, N_in):
        # compute K from N
        # we know C = 6 * N * B * K * seq_len/ 1bill
        D_calc = C /6 *1000 / N_in *1000 *1000 #/seq_len
        # compute the loss
        loss = E + A/(N_in **alpha) + B/(D_calc**beta)
        #raise ValueError("N_in: ", N_in, "D_calc: ", D_calc)
        return loss
    
    def chin_loss_from_D(C, D_in):
        # compute N from K
        N_calc = C /6 *1000 *1000 / D_in  *1000 
        D_calc = D_in #/seq_len
        # compute the loss
        loss = E + A/(N_calc**alpha) + B/(D_in **beta)
        return loss
    

    return chin_loss_from_N, chin_loss_from_D


def chinchilla_parametric_hyper_selection_func(compute_budget, params):

        G = params["G"]
        a = params["a"]
        b = params["b"]

        billion = 1000000000
        ND = compute_budget/6
        opt_N = G * ND**a * billion**a
        opt_D = 1/G * ND**b * billion**b
        # round to the nearest integer
        opt_N = round(opt_N)
        opt_D = round(opt_D)
        # create a dictionary of the optimal N, D
        opt_N_D = {"N": opt_N, "D": opt_D}
        return opt_N_D
    
def get_chinchilla_parametric_hyper_selection_params(cp_cfg):

    alpha = cp_cfg["N_power"]
    beta = cp_cfg["D_power"]
    A = cp_cfg["A"]
    B = cp_cfg["B"]

    # optimal N = ((a * A) /(b * B))**(1/(a + b)) * (C**(b/(a + b))) * (billion**(b/(a + b)))
    G = ((alpha * A) /(beta * B))**(1/(alpha + beta))
    a = beta / (alpha + beta)
    b = alpha / (alpha + beta)

    params = {"G": G, "a": a, "b": b}
    
    return params

def chin_allocation(cp_cfg, compute_budget, B, seq_len):
    params = get_chinchilla_parametric_hyper_selection_params(cp_cfg)
    compute_budget_over_seq_len = compute_budget/seq_len
    opt_N_D = chinchilla_parametric_hyper_selection_func(compute_budget_over_seq_len, params)
    N = opt_N_D["N"]
    D = opt_N_D["D"] 
    B = B
    K = int(D/B)
  #  D = D * seq_len
    C = 6*N/1000*B/1000*K/1000*seq_len
    # round C to nearest thousand
    C = round(C, -3)

    selected_hypers = {"N": N, "K": K, "B": B, "D": D, "C": C}

    return selected_hypers
