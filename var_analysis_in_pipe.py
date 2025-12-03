import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import os
import sys

if __name__ == "__main__":

    # check if path and chin_path are given in the command line arguments

    #if len(sys.argv) >= 1:
    path = sys.argv[1]
    #path = 'outputs/runs/2025-11-25-20-07_openwebtext2_pythia_sgd_tol_0dot1/'

    chin_path = 'outputs/runs/2025-09-22-14-42_openwebtext2_pythia_nqsplusplus_sgd/4_loss_estimation/chinchilla_2/eval_df.csv'

    def kendall_tau_b_scipy(y, yhat, nan_policy="omit", alternative="two-sided"):
        """
        Kendall's tau-b (tie-corrected) via SciPy.
        
        Returns
        -------
        tau : float
            Kendall's tau-b correlation.
        pvalue : float
            P-value for the hypothesis test (H0: tau == 0), using
            exact method when feasible.
        """
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(yhat, dtype=float)

        # Drop NaNs if requested
        if nan_policy == "omit":
            mask = np.isfinite(y) & np.isfinite(yhat)
            y, yhat = y[mask], yhat[mask]

        n = y.size
        if n < 2:
            return np.nan, np.nan

        # Prefer exact test for small n with no ties; else let SciPy choose.
        no_ties = (np.unique(y).size == n) and (np.unique(yhat).size == n)
        method = "exact" if (n <= 50 and no_ties) else "auto"

        res = kendalltau(
            y, yhat,
            variant="b"#,            # tau-b (tie-corrected)
            #method=method,          # "exact" if feasible, else SciPy decides
            #nan_policy=nan_policy,  # "omit" | "propagate" | "raise"
            #alternative=alternative # "two-sided" | "less" | "greater"
        )
        # SciPy returns KendalltauResult(correlation=..., pvalue=...)
        tau = getattr(res, "correlation", getattr(res, "statistic", np.nan))
        return tau, res.pvalue




    log_scale = True

    # read a csv files into dataframes
    # isoflop_nqs = outputs/runs/2025-08-26-21-48_openwebtext2_pythia_pow_dot58_valid_test_adam/5_resource_allocation/h_samples_with_nqs_loss.csv
    # isoflop_nn = outputs/runs/2025-08-26-21-48_openwebtext2_pythia_pow_dot58_valid_test_adam/5_resource_allocation/to_get_nn_samples.csv
    # isotoken_nqs_nn = outputs/runs/2025-08-26-21-48_openwebtext2_pythia_pow_dot58_valid_test_adam/6_critical_batch_size/h_samples_with_nn_loss.csv

   # path = 'outputs/runs/2025-11-25-20-07_openwebtext2_pythia_sgd_tol_0dot1/'
    #'outputs/runs/2025-11-12-14-01_openwebtext2_pythia_cosine_good_lr_adapt/'



   # chin_path = 'outputs/runs/2025-09-22-14-42_openwebtext2_pythia_nqsplusplus_sgd/4_loss_estimation/chinchilla_2/eval_df.csv'
    #'outputs/runs/2025-11-20-21-21_openwebtext2_pythia_cosine_chin_evaluation/4_loss_estimation/chinchilla_2/eval_df.csv'
    #'outputs/runs/2025-09-13-16-09_openwebtext2_pythia_balanced_dot03_dot75/4_loss_estimation/chinchilla_2/eval_df.csv'
    #'outputs/runs/2025-09-08-20-23_openwebtext2_pythia_eval_chin/4_loss_estimation/chinchilla_2/eval_df.csv'
    chin_valid_path = chin_path #'outputs/runs/2025-09-14-22-07_openwebtext2_pythia_fit_chin_on_all/4_loss_estimation/chinchilla_2/eval_df.csv'
    #'outputs/runs/2025-09-13-22-22_openwebtext2_pythia_dot3/4_loss_estimation/chinchilla_2/eval_df.csv'
    #'outputs/runs/2025-09-11-17-10_openwebtext2_pythia_eval_chin_valid_and_test/4_loss_estimation/chinchilla_2/eval_df.csv'
    #'outputs/runs/2025-09-08-21-27_openwebtext2_pythia_eval_chin_valid/4_loss_estimation/chinchilla_2/eval_df.csv'

    isoflop_nqs_path = path + "5_resource_allocation/h_samples_with_nqs_loss.csv"
    isoflop_nn_path = path + "5_resource_allocation/to_get_nn_samples.csv"
    isotoken_nqs_nn_path = path + "6_critical_batch_size/h_samples_with_nn_loss.csv"
    isoflop_chin_path = chin_path
    isoflop_chin_valid_path = chin_valid_path

    with open(path + 'var_analysis_readme.txt', 'w') as f:
        f.write(f"isoflop_nqs_path: {isoflop_nqs_path}\n")
        f.write(f"isoflop_nn_path: {isoflop_nn_path}\n")
        f.write(f"isotoken_nqs_nn_path: {isotoken_nqs_nn_path}\n")
        f.write(f"isoflop_chin_path: {isoflop_chin_path}\n")
        f.write(f"isoflop_chin_valid_path: {isoflop_chin_valid_path}\n")
        f.write(f"log_scale: {log_scale}\n")

    isoflop_nqs = pd.read_csv(isoflop_nqs_path)

    isoflop_nn = pd.read_csv(isoflop_nn_path)
    isoflop_chin = pd.read_csv(isoflop_chin_path)
    isoflop_chin_valid = pd.read_csv(isoflop_chin_valid_path)


    # check if isotoken_path exists 
    if os.path.exists(isotoken_nqs_nn_path):
        isotoken_nqs_nn = pd.read_csv(isotoken_nqs_nn_path)



    join_keys = ['actual_N','B','K','C']
    # print the columns of isoflop_nqs
    print("isoflop_nqs columns:", isoflop_nqs.columns)
    print("isoflop_nn columns:", isoflop_nn.columns)
    print("isoflop_chin columns:", isoflop_chin.columns)
    #raise ValueError("stop here")
    # left join isoflop_nqs and isoflop_nn on join_keys, keeping the key fields, and loss and nqs_loss
    # drop all other columns from isoflop_nn except for the join keys and loss and nqs_loss
    # dont use suffixes
    isoflop_nqs = isoflop_nqs[['N','B','K','C'] + ['nqs_loss']]
    # group by C and give count of rows
    print("isoflop_nqs summary by C")
    print(isoflop_nqs.groupby('C').size())
    #raise ValueError("stop here")
    isoflop_nn = isoflop_nn[join_keys + ['NN_loss']]
    isoflop_chin = isoflop_chin[join_keys + ['chin_loss']]
    # drop duplicates in isoflop_chin based on join_keys
    isoflop_chin = isoflop_chin.drop_duplicates(subset=join_keys)
    isoflop_chin_valid = isoflop_chin_valid[join_keys + ['chin_loss']]
    # rename chin_loss in isoflop_chin_valid to chin_loss_valid
    isoflop_chin_valid = isoflop_chin_valid.rename(columns={'chin_loss': 'chin_loss_valid'})
    # remove duplicates in isoflop_chin_valid
    isoflop_chin_valid = isoflop_chin_valid.drop_duplicates(subset=join_keys)
    # group by C and give count of rows
    print("isoflop_chin_valid summary by C")
    print(isoflop_chin_valid.groupby('C').size())
    #raise ValueError("stop here")
    isoflop_df = pd.merge(isoflop_nqs, isoflop_nn, on=['C','B','K'], how='left')
    # count how many riws in isoflop_df, isoflop_nqs, isoflop_nn
    print("isoflop_df rows:", len(isoflop_df))
    print("isoflop_nqs rows:", len(isoflop_nqs))
    print("isoflop_nn rows:", len(isoflop_nn))
    #raise ValueError("stop here")
    isoflop_df = pd.merge(isoflop_df, isoflop_chin, on=join_keys, how='left')
    isoflop_df = pd.merge(isoflop_df, isoflop_chin_valid, on=join_keys, how='left')
    # add a field called "type" to isoflop_df, with value "isoflop"
    isoflop_df['type'] = 'isoflop'
    # summarize isoflop_df by C and give coutnt of rows
    print("isoflop_df summary by C")
    print(isoflop_df.groupby('C').size())
    #raise ValueError("stop here")



    # check if isotoken_nqs_nn_path exists
    if os.path.exists(isotoken_nqs_nn_path):

        # now concatenate isotoken_nqs_nn to isoflop_df
        # only keep the columns join_keys and NN_loss and nqs_loss
        isotoken_nqs_nn = isotoken_nqs_nn[join_keys + ['NN_loss', 'nqs_loss']]
        # add a field called "type" to isotoken_nqs_nn, with value "isotoken"
        isotoken_nqs_nn['type'] = 'isotoken'
        isoflop_df = pd.concat([isoflop_df, isotoken_nqs_nn], ignore_index=True)




    # add a field with log NN_loss and log nqs_loss
    isoflop_df['log_NN_loss'] = isoflop_df['NN_loss'].apply(lambda x: None if x <= 0 else np.log(x))
    isoflop_df['log_nqs_loss'] = isoflop_df['nqs_loss'].apply(lambda x: None if x <= 0 else np.log(x))
    isoflop_df['log_chin_loss'] = isoflop_df['chin_loss'].apply(lambda x: None if x <= 0 else np.log(x))
    isoflop_df['log_chin_loss_valid'] = isoflop_df['chin_loss_valid'].apply(lambda x: None if x <= 0 else np.log(x))

    nn_field = 'log_NN_loss' if log_scale else 'NN_loss'
    nqs_field = 'log_nqs_loss' if log_scale else 'nqs_loss'
    chin_field = 'log_chin_loss' if log_scale else 'chin_loss'
    chin_valid_field = 'log_chin_loss_valid' if log_scale else 'chin_loss_valid'

    # check if there are any NaN values in nn_field or nqs_field or chin_field
    print(f"Number of NaN values in {nn_field}: {isoflop_df[nn_field].isna().sum()}")
    print(f"Number of NaN values in {nqs_field}: {isoflop_df[nqs_field].isna().sum()}")
    print(f"Number of NaN values in {chin_field}: {isoflop_df[chin_field].isna().sum()}")
    print(f"Number of NaN values in {chin_valid_field}: {isoflop_df[chin_valid_field].isna().sum()}")

    # count isoflop vs isotoken rows
    print("Number of isoflop rows:", (isoflop_df['type'] == 'isoflop').sum())
    print("Number of isotoken rows:", (isoflop_df['type'] == 'isotoken').sum())



    # if the actual_N = 10171392.0, then set N = 1.0e7
    isoflop_df['N'] = isoflop_df['actual_N'].apply(lambda x: 1.0e7 if x == 10171392.0 else x)
    # if C = 950000000, then set C = 940000000
    isoflop_df['C'] = isoflop_df['C'].apply(lambda x: 940000000 if x == 950000000 else x)

    isoflop_df['test_train_split'] = isoflop_df['C'].apply(lambda x: 'train' if x < 2e8 else 'valid' if x < 4*2e8 else 'test')
    # if valid and isotoken, then set to test
    isoflop_df.loc[(isoflop_df['test_train_split'] == 'valid') & (isoflop_df['type'] == 'isotoken'), 'test_train_split'] = 'test'

    # show the first few rows of the dataframe
    print("isoflop_df")
    print(isoflop_df.head())

    # save isoflop_df to a csv file
    isoflop_df.to_csv(path + 'isoflop_df.csv', index=False)

    # group by C and type, compute the mean of the field NN_loss, save in new field NN_loss_mean_by_C
    isoflop_df['NN_loss_mean_by_C'] = isoflop_df.groupby(['C', 'type'])[nn_field].transform('mean')
    # reset index
    isoflop_df = isoflop_df.reset_index(drop=True)

    print("isoflop_df with NN_loss_mean_by_C")
    print(isoflop_df[['C', 'NN_loss_mean_by_C']].drop_duplicates())

    # create a new column called sqr_diff_NN_with_NN_mean_by_C, which is the square of the difference between NN_loss and NN_loss_mean_by_C
    isoflop_df['sqr_diff_NN_with_NN_mean_by_C'] = (isoflop_df[nn_field] - isoflop_df['NN_loss_mean_by_C']) ** 2

    # create a new column called sqr_diff_NN_with_NQS, which is the square of the difference between NN_loss and NQS_loss
    isoflop_df['sqr_diff_NN_with_NQS'] = (isoflop_df[nn_field] - isoflop_df[nqs_field]) ** 2

    # create a new column called sqr_diff_NN_with_chin, which is the square of the difference between NN_loss and chin_loss
    isoflop_df['sqr_diff_NN_with_chin'] = (isoflop_df[nn_field] - isoflop_df[chin_field]) ** 2
    # print nn_field and chin_field where sqr_diff_NN_with_chin is NaN
    print("Rows where sqr_diff_NN_with_chin is NaN:")
    print(isoflop_df[isoflop_df['sqr_diff_NN_with_chin'].isna()][[nn_field, chin_field, 'sqr_diff_NN_with_chin']])
    # raise error if there are any NaN values in sqr_diff_NN_with_chin
    #if isoflop_df['sqr_diff_NN_with_chin'].isna().sum() > 0:
    #   raise ValueError("There are NaN values in sqr_diff_NN_with_chin")
    # create a new column called sqr_diff_NN_with_chin_valid, which is the square of the difference between NN_loss and chin_loss_valid
    isoflop_df['sqr_diff_NN_with_chin_valid'] = (isoflop_df[nn_field] - isoflop_df[chin_valid_field]) ** 2

    # create a column test_train_split, which is "train" if C < 2e8
    # else if C < 4*2e8 then "valid", else "test"



    # group by test_train_split and type, find the lowest 2 values of loss in nn_field, 
    # add a column that indictes if the row is one of the lowest 2 values of loss in nn_field for that group
    isoflop_df['is_lowest_2_NN_loss'] = isoflop_df.groupby(['test_train_split', 'type'])[nn_field] \
        .transform(lambda x: x.isin(x.nsmallest(2))).astype(int)    
    # add a column se at lowest 2 values of loss in nn_field for that group
    isoflop_df['se_at_lowest_2_nqs_vs_NN_loss'] = isoflop_df['is_lowest_2_NN_loss'] * isoflop_df['sqr_diff_NN_with_NQS'] * 0.5

    # do same for chin_field
    isoflop_df['se_at_lowest_2_chin_vs_NN_loss'] = isoflop_df['is_lowest_2_NN_loss'] * isoflop_df['sqr_diff_NN_with_chin'] * 0.5
    #print("isoflop_df with is_lowest_2_NN_loss")
    #print(isoflop_df[['C', nn_field, 'test_train_split', 'is_lowest_2_NN_loss']].sort_values(by=['test_train_split', nn_field]))
    #raise ValueError("stop here")
    # print the valid isoflop where is_lowest_2_NN_loss is 1
    print("isoflop_df valid rows with is_lowest_2_NN_loss = 1")
    check = isoflop_df[(isoflop_df['test_train_split'] == 'valid') & (isoflop_df['type'] == 'isoflop') & (isoflop_df['is_lowest_2_NN_loss'] == 1)][['C', nn_field, nqs_field, 'nqs_loss', 'is_lowest_2_NN_loss']]
    # in check, calculate the average squared distance between nn_field and nqs_field
    sqr_distance_nn_nqs = (check[nn_field] - check[nqs_field]) ** 2
    print(sqr_distance_nn_nqs.mean())

    # print the values in sqr_diff_NN_with_chin
    print("sqr_diff_NN_with_chin:")
    print(isoflop_df['sqr_diff_NN_with_chin'])
    #raise ValueError("stop here")

    # compute the mean of sqr_diff_NN_with_NN_mean_by_C and sqr_diff_NN_with_NQS, grouped by test_train_split and type
    var_analysis = isoflop_df.groupby(['type', 'test_train_split']).agg(
        var_NN_with_NN_mean_by_C = ('sqr_diff_NN_with_NN_mean_by_C', 'mean'),
        var_NN_with_NQS = ('sqr_diff_NN_with_NQS', 'mean'),
        # calculate the average of var_NN_with_NQS only for rows where is_lowest_2_NN_loss is 1
        # do this by summing se_at_lowest_2_NN_loss and dividing by 2
        mse_NN_with_NQS_lowest_2 = ('se_at_lowest_2_nqs_vs_NN_loss', 'sum'),
        mse_NN_with_chin_lowest_2 = ('se_at_lowest_2_chin_vs_NN_loss', 'sum'),
        # divide mse_NN_with_NQS_lowest_2 
        var_NN_with_chin = ('sqr_diff_NN_with_chin', 'mean'),
        var_NN_with_chin_valid = ('sqr_diff_NN_with_chin_valid', 'mean'),
        # get the count of rows in each group
        count = ('sqr_diff_NN_with_NN_mean_by_C', 'count')
    ).reset_index()

    # in var_analysis, filter for validation and isoflop and get var_NN_with_NQS_lowest_2
    print("var_analysis for valid isoflop with var_NN_with_NQS_lowest_2")
    print(var_analysis[(var_analysis['test_train_split'] == 'valid') & (var_analysis['type'] == 'isoflop')][['var_NN_with_NQS', 'mse_NN_with_NQS_lowest_2']])
    #raise ValueError("stop here")

    print("var_analysis")
    # compute the var explained = 1 - var_NN_with_NQS / var_NN_with_NN_mean_by_C
    var_analysis['var_explained_nqs'] = 1 - var_analysis['var_NN_with_NQS'] / var_analysis['var_NN_with_NN_mean_by_C']
    var_analysis['var_explained_chin'] = 1 - var_analysis['var_NN_with_chin'] / var_analysis['var_NN_with_NN_mean_by_C']
    var_analysis['var_explained_chin_valid'] = 1 - var_analysis['var_NN_with_chin_valid'] / var_analysis['var_NN_with_NN_mean_by_C']

    print(var_analysis)


    # compute the kendall tau-b correlation between NN_loss and nqs_loss, grouped by test_train_split and type
    for fld in [nqs_field, chin_field, chin_valid_field]:


        kendall_tau_b_results = []
        for (data_type, split), group in isoflop_df.groupby(['type', 'test_train_split']):
            tau, pvalue = kendall_tau_b_scipy(group[nn_field], group[fld])
            kendall_tau_b_results.append({
                'type': data_type,
                'test_train_split': split,
                'kendall_tau_b': tau,
                'pvalue': pvalue,
                'count': len(group)
            })

        kendall_tau_b_df = pd.DataFrame(kendall_tau_b_results)
        print("Kendall tau-b results")
        print(kendall_tau_b_df)
        # drop count and pvalue columns
        kendall_tau_b_df = kendall_tau_b_df.drop(columns=['pvalue', 'count'])
        # rename kendall_tau_b to kendall_tau_b_{fld}
        kendall_tau_b_df = kendall_tau_b_df.rename(columns={'kendall_tau_b': f'kendall_tau_b_{fld}'})
        #print(f"kendall_tau_b

        # jpoin kendall_tau_b_df to var_analysis on type and test_train_split
        var_analysis = pd.merge(var_analysis, kendall_tau_b_df, on=['type', 'test_train_split'], how='left')
        #print(f"var_analysis with kendall tau-b for {fld}")


    # transpose var_analysis, the column names becomes the values in a new column called metric
    var_analysis_t = var_analysis.melt(id_vars=['type', 'test_train_split'], var_name='metric', value_name='value')
    #print("var_analysis_t")
    #print(var_analysis_t)

    # save var_analysis_t to a csv file
    print("saving var_analysis_t to csv at ", path + 'var_analysis.csv')
    var_analysis_t.to_csv(path + 'var_analysis.csv', index=False)
    # document the file paths used to generate the var_analysis.csv file in a text file
    with open(path + 'var_analysis_readme.txt', 'w') as f:
        f.write(f"isoflop_nqs_path: {isoflop_nqs_path}\n")
        f.write(f"isoflop_nn_path: {isoflop_nn_path}\n")
        f.write(f"isotoken_nqs_nn_path: {isotoken_nqs_nn_path}\n")
        f.write(f"isoflop_chin_path: {isoflop_chin_path}\n")
        f.write(f"isoflop_chin_valid_path: {isoflop_chin_valid_path}\n")
        f.write(f"log_scale: {log_scale}\n")
    

    if False:



        # filter var_analysis_t to only keep rows where it is on train split
        var_analysis_t_train = var_analysis_t[var_analysis_t['test_train_split'].isin(['train','valid','test'])]
        # filter for type in isoflop
        var_analysis_t_train = var_analysis_t_train[var_analysis_t_train['type'].isin(['isoflop','isotoken'])]
        # filter for rows for var_explained, kendall_tau
        var_analysis_t_train = var_analysis_t_train[var_analysis_t_train['metric'].str.contains('var_explained')] #|kendall_tau|count|mse')]
        print("var_analysis_t_train")
        print(var_analysis_t_train)


        # print the section of the df where C >= 2e8
        #print("Rows with C >= 2e8")
        double_test = isoflop_df[isoflop_df['C'] >= 2e8*16][['C', 'type', nn_field, nqs_field,  "nqs_loss", "NN_loss"]]
        #print(double_test)
        # get variance explained for these rows
        sqr_distance_nn_nqs = (double_test[nn_field] - double_test[nqs_field]) ** 2
        sqr_distance_nn_mean = (double_test[nn_field] - double_test[nn_field].mean()) ** 2
        var_explained = 1 - sqr_distance_nn_nqs.sum() / sqr_distance_nn_mean.sum()
        #print(f"Variance explained by NQS for C >= 2e8*16: {var_explained}")



                        