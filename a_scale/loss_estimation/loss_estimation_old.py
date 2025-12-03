
import os
import pandas as pd
from omegaconf import OmegaConf
from a_scale.archiving_utils import archive_wrapper, df_row_to_dict
from a_scale.run_nn import train_nn
from a_scale.plotting_utils import actual_vs_fitted_plot
from a_scale.plotting_utils import plot_loss_curves
import json
import numpy as np

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
                    test_last_ckpt_only):
    
    burn_in = 2
    
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
    
    fitted_nqs_dict = OmegaConf.to_container(fitted_nqs)

    nqs_lookup_dict = {k: fitted_nqs_dict[k] for k in ['a', 'b', 'm_b']}
    
    # for each row in h_samples, get the nn loss
    for j, h_samples in enumerate([h_samples_train, h_samples_test]):
        if j == 0:
            h_samples_type = "train"
        else:
            h_samples_type = "test"
        for i in range(len(h_samples)):
            # convert the ith row of h_samples to a dictionary
            h_i_dict = df_row_to_dict(h_samples, i)
            # h_samples.iloc[i].to_dict()

            # -- get NN loss --
            nn_i_dict = dict(neural_net.copy())
            msg, nn_out = archive_wrapper(train_nn)(
                 nn_i_dict, h_i_dict, nn_archive_file) # a dictionary with values dataframes
            nn_loss_df_i = nn_out["loss_curve_df"] # a dataframe with columns ckpt, loss
            actual_N = int((nn_out["actual_N_df"])["actual_N"].values[0]) # the actual N (may differ from requested N in h_i_dict)
            nn_loss_df_i["actual_N"] = actual_N
            # filter for ckpt larger than burn_in
            nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] > burn_in]
            ckpts_to_get_nqs = list(nn_loss_df_i["ckpt"].values)

            # -- get NQS loss --
            use_actual_N_for_nqs = True
            if use_actual_N_for_nqs:
                h_i_dict.update({"N": actual_N})
            
            # empty dataframe
            

            for cpt in ckpts_to_get_nqs:
                h_i_dict.update({"K": int(cpt)})
                # run nqs bayes,bias,var for the ith sample
                msg, nqs_out = archive_wrapper(nqs_simulator)(
                    nqs_lookup_dict, h_i_dict, nqs_archive_file) # a dictionary with values dataframes
                if cpt == ckpts_to_get_nqs[0]:
                    nqs_bbv_df_i = nqs_out["nqs_df"] # a dataframe with columns nqs_iter, bayes_risk, nqs_bias, nqs_var
                else:
                    nqs_bbv_df_i = pd.concat([nqs_bbv_df_i, nqs_out["nqs_df"]], axis = 0)
            
            # -- merge NN and NQS loss --
            eval_df_i = pd.merge(nn_loss_df_i, nqs_bbv_df_i, 
                                       left_on = 'ckpt', right_on = 'nqs_iter', how = 'inner')
            eval_df_i.drop(columns = ['nqs_iter'], inplace = True)
            eval_df_i["h_samples_type"] = h_samples_type

            # if test set, print nn_loss_df_i and nqs_bbv_df_i in error
            #if h_samples_type == "test":
            #    raise ValueError("nn_loss_df_i: " + str(nn_loss_df_i) + "\n" + "nqs_bbv_df_i: " + str(nqs_bbv_df_i))
            
            # add h_i_dict to eval_df_i
            for key in h_i_dict.keys():
                valuee = h_i_dict[key]
                # if valuee is a dictionary, convert it to a string
                if isinstance(valuee, dict):
                    valuee = json.dumps(valuee)
                eval_df_i[key] = valuee

            

            if j==0 and i == 0:
                eval_df = eval_df_i
            else:
                eval_df = pd.concat([eval_df, eval_df_i], axis = 0)
    
    def calc_nqs_from_bbv(row, fitted_nqs_dict):
        m_a = fitted_nqs_dict["m_a"]
        eps = fitted_nqs_dict["eps"]
        sigma = fitted_nqs_dict["sigma"]
        m_b = fitted_nqs_dict["m_b"]
        nqs = eps + m_a/m_b * (row["bayes_risk"] + row["nqs_bias"]) + sigma**2 * row["nqs_var"]
        return nqs



    eval_df["nqs_loss"] = eval_df.apply(lambda row: calc_nqs_from_bbv(row, fitted_nqs_dict), axis = 1)
    eval_df.to_csv(loss_estimation_dir + "eval_df.csv", index = False)


    # get loss curves
    loss_curve_plot_file = loss_estimation_dir + "loss_curves.png"
    plot_loss_curves(eval_df, loss_curve_plot_file)


    eval_df_test = eval_df[eval_df["h_samples_type"] == "test"]
    if test_last_ckpt_only:
        # for the test set, keep only the last ckpt
        eval_df_test = eval_df_test[eval_df_test["ckpt"] == eval_df_test["ckpt"].max()]
    
    # raise an error that prints eval_df_test
    if eval_df_test.shape[0] == 0:
        raise ValueError("eval_df_test is empty")
    
    eval_df_train = eval_df[eval_df["h_samples_type"] == "train"]
    eval_df = pd.concat([eval_df_train, eval_df_test], axis = 0)
    

    actual_values = eval_df[eval_df["h_samples_type"] == "train"]["loss"].values
    fitted_values = eval_df[eval_df["h_samples_type"] == "train"]["nqs_loss"].values

    test_actual_values = eval_df[eval_df["h_samples_type"] == "test"]["loss"].values
    test_fitted_values = eval_df[eval_df["h_samples_type"] == "test"]["nqs_loss"].values

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
                          train_param_counts = None)
    

  
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
        if True:
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
                    nn_loss_df_i["actual_N"] = actual_N
                    nn_loss_df_i["h_samples_type"] = "train"
                    # filter for ckpt larger than burn_in
                    nn_loss_df_i = nn_loss_df_i[nn_loss_df_i["ckpt"] > burn_in].copy()
                    # only keep the last 3 ckpts
                    nn_loss_df_i = nn_loss_df_i.tail(3).copy()

                    # add h_i_dict to eval_df_i, by looping through the keys of h_i_dict and adding them as columns
                    for key in h_i_dict.keys():
                        #nn_loss_df_i[key] = h_i_dict[key]
                        # add a column to nn_loss_df_i with the key as the column name
                        # set every row to the same value which is h_i_dict[key]
                        nn_loss_df_i.loc[:, key] = h_i_dict[key]

                    if i == 0:
                        nn_loss_df = nn_loss_df_i
                    else:
                        nn_loss_df = pd.concat([nn_loss_df, nn_loss_df_i], axis = 0)
                
                # in eval_df, replace the rows with h_samples_type = "train" with
                #  the rows in nn_loss_df 
                eval_df_tst = eval_df[eval_df["h_samples_type"] == "test"]
                eval_df = pd.concat([nn_loss_df, eval_df_tst], axis = 0)



        eval_df["D"] = eval_df["ckpt"] * eval_df["B"]
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
            return E + A / row["actual_N"]**alpha + B / row["D"]**beta
        
        eval_df["chin_loss"] = eval_df.apply(lambda row: chin_predict_loss(row, fitted_baseline_dict), axis = 1)
        # save the eval_df to a csv file
        eval_df.to_csv(baseline_dir + "eval_df.csv", index = False)

        actual_values = eval_df[eval_df["h_samples_type"] == "train"]["loss"].values
        test_actual_values = eval_df[eval_df["h_samples_type"] == "test"]["loss"].values

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
                              test_actual_values = test_actual_values, 
                              test_fitted_values = chin_fitted_values_test,
                              train_param_counts = None)
    
    # save the eval_metrics_df to a csv file
    eval_metrics_df.to_csv(loss_estimation_dir + "eval_metrics_df.csv", index = False)
    return eval_metrics_df
    