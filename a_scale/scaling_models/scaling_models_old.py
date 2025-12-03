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
#from a_scale.sampling import h_sampler_time_compute_budget, sample_a_b_mb

# Helper function to bridge pandas.df and jnp arrays
# make sure each element in x_data is a jnp array
def series_to_jnp(series):
        series.apply(lambda x: float(x))
        return jnp.array(series.to_numpy())


# ----------------- Zeroth Order Scaling Model -----------------

def zeroth_order_scaling_model(h_sampler_training, 
                               nqs_sampler_grid_search, 
                               nqs_simulator,
                               neural_net, 
                               eval_metric,
                               scaling_model_dir, 
                               fitted_nqs_file,
                               nn_archive_file,
                               nqs_archive_file):
    
    '''
    as this function is instantiated, so is h_sampler_training and nqs_sampler_grid_search
    it remains to do:
    0. execute the samplers
    1. run neural_net for samples in h_sampler_training to get trained hypers + nn loss 
    2. for each nqs sampled by nqs_sampler_grid_search, 
        run nqs bayes,bias,var for samples in h_sampler_training to get trained nqs + nqs bayes,bias,var
    3. find the nqs whose bayes, bias, var best matches nn loss
    '''

    burn_in = 2
    
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

    nqs_samples_dict = nqs_sampler_grid_search()
    nqs_samples = nqs_samples_dict["samples"]
    nqs_samples.to_csv(scaling_model_dir + "nqs_samples.csv", index = False)
    # read the nqs_samples from the saved file
    nqs_samples = pd.read_csv(scaling_model_dir + "nqs_samples.csv")

    
    # for each nqs sampled by nqs_sampler_grid_search,
    # run nqs bayes,bias,var for samples in h_sampler_training to get trained nqs + nqs bayes,bias,var
    search_results = nqs_samples.copy()
    # create a subfolder under scaling_model_dir, called trials
    trials_dir = scaling_model_dir + "trials/"
    if not os.path.exists(trials_dir):
        os.makedirs(trials_dir)

 

    for j in range(len(nqs_samples)):
        # create a subdirectory for the jth nqs
        j_dir = trials_dir + str(j) + '/'
        #   write this using column name row name     search_results.at[j, "nqs_dir"] = j_dir
        search_results.loc[j, "nqs_dir"] = j_dir
        search_results.to_csv(scaling_model_dir + "search_results.csv", index = False)


        if not os.path.exists(j_dir):
            os.makedirs(j_dir)

        # read the jth row of nqs_samples
        nqs_j_dict = df_row_to_dict(nqs_samples, j)
        #nqs_samples.iloc[j].to_dict()
        # convert the dictionary to OmegaConf
        #nqs_j = OmegaConf.create(nqs_j)

        data_to_fit_nqs_j = pd.DataFrame(columns = ['ckpt', 'loss', 'bayes_risk', 'nqs_bias', 'nqs_var'])

        
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
            actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0]) # the actual N (may differ from requested N in h_i_dict)
            # filter out the burn-in period
            nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] >= burn_in]
            ckpts_to_get_nqs = list(nn_loss_df_i["ckpt"].values)

            use_actual_N_for_nqs = True
            #raise ValueError(h_i_dict)
            if use_actual_N_for_nqs:
                h_i_dict.update({"N": actual_N})

            for cpt in ckpts_to_get_nqs:
                h_i_dict.update({"K": int(cpt)})
                # run nqs bayes,bias,var for the ith sample
                msg, nqs_out = archive_wrapper(nqs_simulator)(
                    nqs_j_dict, h_i_dict, nqs_archive_file) # a dictionary with values dataframes
                if cpt == ckpts_to_get_nqs[0]:
                    nqs_bbv_df_ij = nqs_out["nqs_df"] # a dataframe with columns nqs_iter, bayes_risk, nqs_bias, nqs_var
                else:
                    nqs_bbv_df_ij = pd.concat([nqs_bbv_df_ij, nqs_out["nqs_df"]], axis = 0)
            
            # run nqs bayes,bias,var for the ith sample
            #msg, nqs_out = archive_wrapper(nqs_simulator)(
            #     nqs_j_dict, h_i_dict, nqs_archive_file) # a dictionary with values dataframes
            #nqs_bbv_df_ij = nqs_out["nqs_df"] # a dataframe with columns nqs_iter, bayes_risk, nqs_bias, nqs_var
            
            # left join nn_loss_df and nqs_bbv_df on ckpt = nqs_iter
            curve_fit_df_ij = pd.merge(nn_loss_df_i, nqs_bbv_df_ij, 
                                       left_on = 'ckpt', right_on = 'nqs_iter', how = 'inner')
            # drop the nqs_iter column
            curve_fit_df_ij.drop(columns = ['nqs_iter'], inplace = True)
            # concat to the data_to_fit_nqs_i
            if i == 0:
                data_to_fit_nqs_j = curve_fit_df_ij
            else:
                data_to_fit_nqs_j = pd.concat([data_to_fit_nqs_j, curve_fit_df_ij], axis = 0)

        # fit the scaling model using data_to_fit_nqs_i
        bayes_risk = data_to_fit_nqs_j ["bayes_risk"]
        biases_at_iters = data_to_fit_nqs_j ["nqs_bias"]
        variances_at_iters = data_to_fit_nqs_j ["nqs_var"]
        losses = data_to_fit_nqs_j ["loss"]
        # fit: logloss = log(c0 + c1 * nqs_bias + c2 * nqs_var)
        def func(x, c0, c1, c2):
            return jnp.log(c0 + c1* x[0] + c2 * x[1])
        # use curve_fit to fit the function, with constraints c0 > 0, c1 > 0, c2 > 0
        x_data = [bayes_risk + biases_at_iters, variances_at_iters]
        
        x_data = [series_to_jnp(series) for series in x_data]
        losses = series_to_jnp(losses)

        bounds = (jnp.zeros(3), [jnp.inf, jnp.inf, jnp.inf])
        popt, _ = curve_fit(func, 
                            x_data,
                            jnp.log(losses), 
                            bounds = bounds)

       
        eps = popt[0]
        ma_over_mb = popt[1]
        sigma = popt[2]**0.5

        # convert the fitted values to float
        eps = float(eps)
        ma_over_mb = float(ma_over_mb)
        sigma = float(sigma)

        nqs_j_dict.update({'m_a': nqs_j_dict['m_b']*ma_over_mb,
                            'eps': eps, 'sigma': sigma})
        nqs_j_file = j_dir + "nqs_dict.yaml"
        OmegaConf.save(nqs_j_dict, nqs_j_file)

        # get the fitted values
        # make sure x_data and popt are numpy arrays
        fitted_values = jnp.exp(func(x_data, *popt))# exponential of the fitted values
        # add the fitted values to the dataframe
        data_to_fit_nqs_j["fitted_values"] = fitted_values
        # save data_to_fit_nqs_i to a csv file
        data_to_fit_nqs_j.to_csv(j_dir + "data_to_fit_nqs.csv", index = False)


        # get a goodness of fit measure
        eval_metric_value = eval_metric(data_to_fit_nqs_j["loss"].values, fitted_values)

        # add the goodness of fit measure to the grid search space dataframe
        search_results.loc[j, "eval_metric"] = eval_metric_value

    # sort the search_results by the column eval_metric, in ascending order
    search_results.sort_values(by = "eval_metric", inplace = True)
    # save the search_results to a csv file
    search_results.to_csv(scaling_model_dir + "search_results.csv", index = False)
    # find the path to the best nqs (path of first row of search_results)
    best_nqs_path = search_results.iloc[0]["nqs_dir"] + "nqs_dict.yaml"
    # read the best nqs
    best_nqs = OmegaConf.load(best_nqs_path)
    # save the best_nqs to the fitted_nqs_file
    OmegaConf.save(best_nqs, fitted_nqs_file)
    # get the eval_metric value of the best nqs
    best_eval_metric = search_results.iloc[0]["eval_metric"]
    
    # return the best_nqs
    return best_nqs, best_eval_metric



if __name__ == "__main__":
    None
    