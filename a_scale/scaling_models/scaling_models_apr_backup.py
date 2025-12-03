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
from a_scale.design_architecture import remove_embedding_params, convert_to_effective_params

#from a_scale.sampling import h_sampler_time_compute_budget, sample_a_b_mb

# Helper function to bridge pandas.df and jnp arrays
# make sure each element in x_data is a jnp array
def series_to_jnp(series):
        series.apply(lambda x: float(x))
        return jnp.array(series.to_numpy())


# ----------------- Zeroth Order Scaling Model -----------------

def zeroth_order_scaling_model_apr_backup(h_sampler_training, 
                               nqs_sampler_grid_search, 
                               nqs_simulator,
                               neural_net, 
                               eval_metric,
                               scaling_model_dir, 
                               fitted_nqs_file,
                               nn_archive_file,
                               nqs_archive_file,
                               include_embedding_params = False,
                               effective_model_size_factor = 1.0,
                               vocab_size =3000,
                               previous_search_results = None):
    
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
    # add a column eval_metric to search_results
    search_results["eval_metric"] = np.nan
    # create a subfolder under scaling_model_dir, called trials
    trials_dir = scaling_model_dir + "trials/"
    if not os.path.exists(trials_dir):
        os.makedirs(trials_dir)

    ls = []
    h_i_dicts = []

    for i in range(len(h_samples)):
        h_i_dict = df_row_to_dict(h_samples, i)
        nn_i_dict = dict(neural_net.copy())
        msg, nn_out = archive_wrapper(train_nn)(
            nn_i_dict, h_i_dict, nn_archive_file)
        nn_loss_df_i = nn_out["loss_curve_df"]
        actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0])
        nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] >= burn_in]

        
        # temp: keep the last two ckpts only
        # nn_loss_df_i = nn_loss_df_i.tail(2)

        # sort from smallest to largest ckpt
        # count the total number of ckpts
        # keep the ckpts at the 1/4, 1/2, 3/4, and 1/1 quartiles
        nn_loss_df_i = nn_loss_df_i.sort_values(by = ["ckpt"])
        # count the total number of ckpts
        total_ckpts = len(nn_loss_df_i)
        # find the quartiles
        quartiles = [int(total_ckpts * 0.25), int(total_ckpts * 0.5), int(total_ckpts * 0.75), total_ckpts]
        # create a col with row number
        nn_loss_df_i["row_num"] = range(1, total_ckpts + 1)
        # filter the rows with row_num in quartiles
        nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["row_num"].isin(quartiles)]
        # drop the row_num column
        nn_loss_df_i = nn_loss_df_i.drop(columns = ["row_num"])

        #raise ValueError("nn_loss_df_i is " + str(nn_loss_df_i))



   
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
    

    if previous_search_results is not None:
        existing_search_results = pd.read_csv(previous_search_results)
        # concatenate the existing search results with the new search results
        search_results = pd.concat([existing_search_results, search_results], ignore_index = True)
        # group by the columns [a,b,m_a,m_b,eps,sigma] and
        # take the non-nan value of eval_metric
        search_results = search_results.groupby(["a","b","m_a","m_b","eps","sigma"]).agg({"eval_metric": "min"}).reset_index()
    
    # save the search_results to a csv file
    search_results.to_csv(scaling_model_dir + "existing_search_results.csv", index = False)
    # drop the eval_metric and rename to search_results_nqs
    search_results_nqs = search_results.drop(columns = ["eval_metric"])
    for j in range(len(search_results)):
        #raise ValueError("search_results length is " + str(len(search_results)))
        
        if np.isnan(search_results.loc[j, "eval_metric"]):
                    # create a subdirectory for the jth nqs
            j_dir = trials_dir + str(j) + '/'
            if not os.path.exists(j_dir):
                os.makedirs(j_dir)
            # use fit_nqs to fit the nqs
            nqs_sample_j = df_row_to_dict(search_results_nqs, j)

            
            if not include_embedding_params:
                # loop thru h_i_dicts and remove embedding params
                h_i_dicts_to_run = h_i_dicts.copy()
                for k in range(len(h_i_dicts)):
                    h_i_dicts_to_run[k] = remove_embedding_params(h_i_dicts[k], vocab_size)


            if effective_model_size_factor != 1.0:
                # loop thru h_i_dicts and scale the model size
                h_i_dicts_to_run = h_i_dicts.copy()
                for k in range(len(h_i_dicts)):
                    h_i_dicts_to_run[k] = convert_to_effective_params(h_i_dicts[k], effective_model_size_factor)

            fitted_nqs_dict_j, eval_metric_value = fit_nqs(h_dicts = h_i_dicts_to_run, 
                                        nn_losses = ls, 
                                        nqs_init = nqs_sample_j)
            fitted_nqs_file_j = trials_dir + str(j) + "/fitted_nqs.yaml"
            OmegaConf.save(fitted_nqs_dict_j, fitted_nqs_file_j)
            # save the eval_metric_value in the folder as a csv file 
            eval_metric_file = trials_dir + str(j) + "/eval_metric.csv"
            pd.DataFrame([eval_metric_value]).to_csv(eval_metric_file, index = False)

            # update the search_results with the eval_metric_value
            search_results.loc[j, "eval_metric"] = eval_metric_value
            # save the search_results to a csv file
            search_results.to_csv(scaling_model_dir + "search_results.csv", index = False)
            

        else:
            eval_metric_value = search_results.loc[j, "eval_metric"]
            # extract the directory name of the previous_search_results
            previous_dir = os.path.dirname(previous_search_results)
            # the previous best fitted nqs name is "fitted_nqs.yaml"
            # note that this is the best run from the set of previous nqs init samples
            # which is not necessarily equal run j; however, this is sufficient for our purpose,
            # which is to retrieve the previous best in case the new samples are not better
            previous_best_fitted_nqs_path = previous_dir + "/fitted_nqs.yaml"
            fitted_nqs_file_j = previous_best_fitted_nqs_path

        


        if j == 0:
            best_eval_metric = eval_metric_value
            best_fitted_nqs_path = fitted_nqs_file_j
        else:
            if eval_metric_value < best_eval_metric:
                best_eval_metric = eval_metric_value
                best_fitted_nqs_path = fitted_nqs_file_j
        
    # sort the search_results by the column eval_metric, in ascending order
    search_results.sort_values(by = "eval_metric", inplace = True)
    # save the search_results to a csv file
    search_results.to_csv(scaling_model_dir + "search_results.csv", index = False)
    # load the best nqs
    best_nqs = OmegaConf.load(best_fitted_nqs_path)
    # save the best nqs to the fitted_nqs_file
    OmegaConf.save(best_nqs, fitted_nqs_file)
    # return the best fitted nqs
    return best_nqs, best_eval_metric

            

 