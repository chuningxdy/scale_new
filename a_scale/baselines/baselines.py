
import pandas as pd
from a_scale.run_nn import train_nn
from a_scale.archiving_utils import archive_wrapper, df_row_to_dict
import jax.numpy as jnp
import os
from scipy.optimize import curve_fit
from omegaconf import OmegaConf
import json
from a_scale.baselines.chin import train_chin

def series_to_jnp(series):
        series.apply(lambda x: float(x))
        return jnp.array(series.to_numpy())

def chinchilla_parametric(
        h_sampler_training,
        chinchilla_dir,
        neural_net,
        nn_archive_file,
        eval_metric,
        use_last_x_ckpt = 1):
    
    # create chinchilla_dir if it does not exist
    if not os.path.exists(chinchilla_dir):
        os.makedirs(chinchilla_dir)

    h_samples_dict = h_sampler_training()
    h_samples = h_samples_dict["samples"]
    ct_summary = h_samples_dict["ct_summary"]
    h_samples.to_csv(chinchilla_dir + "h_samples.csv", index=False)
    ct_summary.to_csv(chinchilla_dir + "training_cost.csv", index=False)
    # read the h_samples from the saved file
    h_samples = pd.read_csv(chinchilla_dir + "h_samples.csv")

           # for each i in h_samples, run nqs bayes,bias,var with nqs_j and h_i
    for i in range(len(h_samples)):
            # convert the ith row of h_samples to a dictionary
            h_i_dict = df_row_to_dict(h_samples, i)
            #h_samples.iloc[i].to_dict()
            nn_i_dict = dict(neural_net.copy())
      
            
            #nn_cfg.update({'h': h_i_cfg})
            msg, nn_out = archive_wrapper(train_nn)(
                 nn_i_dict, h_i_dict, nn_archive_file) # a dictionary with values dataframes
            
            nn_loss_df_i = nn_out["loss_curve_df"] # a dataframe with columns ckpt, loss
            # remove duplicates
            nn_loss_df_i = nn_loss_df_i.drop_duplicates()
            actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0]) # the actual N (may differ from requested N in h_i_dict)
            nn_loss_df_i["actual_N"] = actual_N

            # add h_i_dict to eval_df_i, by looping through the keys of h_i_dict and adding them as columns
            for key in h_i_dict.keys():
                valuee = h_i_dict[key]
                if isinstance(valuee, list):
                     valuee = json.dumps(valuee)
                nn_loss_df_i[key] = valuee

            # keep the last 3 rows of nn_loss_df_i
            nn_loss_df_i = nn_loss_df_i.tail(use_last_x_ckpt)

            if i == 0:
                dat = nn_loss_df_i
            else:
                dat = pd.concat([dat, nn_loss_df_i], axis = 0)
    
    # D = ckpt * B
    dat['D'] = dat['ckpt'] * dat['B'] * 128
    # # filter for where N > 0 and D > 0
    dat = dat[(dat['actual_N'] > 0) & (dat['D'] > 0)]

    # save the dataframe to a csv file
    dat.to_csv(chinchilla_dir + "fit_chinchilla_data.csv", index = False)

    def chinchilla_loss_func(x, E, A, alpha, B, beta):
        return jnp.log(E + A / x[0]**alpha + B / x[1]**beta)
    
    x_data = [dat['actual_N'], dat['D']]
    y_data = jnp.log(series_to_jnp(dat['loss']))
    # convert x_data to jnp arrays
    x_data = [series_to_jnp(series) for series in x_data]
    

    use_chinchilla_paper_method = True
    if use_chinchilla_paper_method: # use the chinchilla paper method

        N = dat['actual_N']
        D = dat['D']
        losses = dat['loss']
        fitted_params, best_init_params, results_dict = train_chin(N, D, losses)
        fitted_cp = fitted_params
        # convert all entries to float
        fitted_cp = {key: float(value) for key, value in fitted_cp.items()}
        # convert the dictionary to an OmegaConf object
        best_init_params = {key: float(value) for key, value in best_init_params.items()}
        with open(chinchilla_dir + "best_init_params.json", "w") as f:
            json.dump(best_init_params, f)

    if not use_chinchilla_paper_method: # use the scipy curve_fit method

        # save the x_data and y_data to a csv file
        # combine x_data and y_data
        xy_data = x_data + [y_data]
        xy_data = pd.DataFrame(xy_data).T
        xy_data.columns = ['actual_N', 'D', 'loss']
        xy_data.to_csv(chinchilla_dir + "xy_data.csv", index = False)


        # create boundaries for the parameters
        bounds = (jnp.zeros(5), [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf])
        initial_guess = [0.1, 1, 0.5, 1, 0.5]
        # set maxfev to 10000
        popt, _ = curve_fit(chinchilla_loss_func,
                            x_data,
                            y_data, 
                            bounds = bounds,
                            p0 = initial_guess,
                            maxfev = 10000)

        # create a dictionary for fitted_cp
        fitted_cp = {'E': popt[0], 
                    'A': popt[1], 
                    'N_power': popt[2], 
                    'B': popt[3], 
                    'D_power': popt[4]}
    # convert all entries to float
    
    fitted_cp = {key: float(value) for key, value in fitted_cp.items()}
    # convert the dictionary to an OmegaConf object
    fitted_cp = OmegaConf.create(fitted_cp)
    # save the fitted_cp to a yaml file as config
    OmegaConf.save(fitted_cp, chinchilla_dir + "fitted_cp.yaml")
    
    # calculated the fitted values
    dat['chin_loss'] = jnp.exp(chinchilla_loss_func(x_data, fitted_cp.E, fitted_cp.A, 
                                                    fitted_cp.N_power, fitted_cp.B, 
                                                    fitted_cp.D_power))
    chin_loss = series_to_jnp(dat['chin_loss'])
    
    # save the dataframe to a csv file
    dat.to_csv(chinchilla_dir + "fit_chinchilla_data.csv", index = False)
    # save the results_dict to json
    #with open(chinchilla_dir + "results_dict.json", "w") as f:
    #    json.dump(results_dict, f)
    # save the best_init_params to json


    train_eval_metric = eval_metric(series_to_jnp(dat['loss']), chin_loss)


    return fitted_cp, train_eval_metric