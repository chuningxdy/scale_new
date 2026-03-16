# create csv files where each row is a config for NN/NQS
import os
import csv
import pandas as pd
import numpy as np


# base file
base_file = 'h_samples_resource_allocation_owt_all_balanced_adam_cosine.csv'
level = 236000000

# out file
out_file = 'h_samples_test_nqs_owt_adam_cosine_NBK_corrected_dense.csv'
# check if out file exists
if os.path.exists(out_file):
    # read the file
    df_existing = pd.read_csv(out_file)

    #40
    out_path = "/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-06-22-01_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
    #2026-01-05-19-55_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
    #"/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-02-17-18_openwebtext2_pythia_adam_cosine/4_loss_estimation/eval_df.csv" # LRA old
    #SN:"/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-01-19-49_openwebtext2_pythia_adam_cosine/4_loss_estimation/eval_df.csv"
    out_path2 = "/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-14-19-36_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"

    # left join with existing file to see if any rows match
    df_out = pd.read_csv(out_path)
    df_out2 = pd.read_csv(out_path2)
    # concatenate df_out and df_out2
    df_out = pd.concat([df_out, df_out2], ignore_index=True)
    # remove duplicates based on B, K, N
    df_out = df_out.drop_duplicates(subset=['B', 'K', 'N'])
    df_merged = pd.merge(df_existing, df_out, how='left', on=['B', 'K', 'N'], suffixes=('_existing', '_eval'))
    # rename N_existing to N
    #df_merged = df_merged.rename(columns={'N_existing': 'N'})
    # filter for rows where eval metric is not null
    
    df_matched = df_merged[~df_merged['NN_loss'].isnull()]
    eval_metric_col = 'NN_loss'
    print(f"Number of matching rows with eval metric: {len(df_matched)}")
    if len(df_matched) == 0:
        raise ValueError("No matching rows found. Exiting.")
    # make a plot of N on x-axis and B on y-axis
    # color is "loss" column from eval_df.csv
    import matplotlib.pyplot as plt
    plt.scatter(df_matched['K'], df_matched['B'], c=df_matched[eval_metric_col], cmap='viridis')
    # Plot scatter from df_existing, use grey color with low alpha
    plt.scatter(df_existing['K'], df_existing['B'], color='grey', alpha=0.3)
    plt.colorbar(label=eval_metric_col)
    plt.xlabel('K')
    plt.ylabel('B')
    plt.title(f'K vs B colored by {eval_metric_col}')
    plt.xscale('log')
    plt.yscale('log')
    # make a star at min loss point
    min_loss_idx = df_matched[eval_metric_col].idxmin()
    plt.scatter(df_matched.loc[min_loss_idx, 'K'], df_matched.loc[min_loss_idx, 'B'], color='red', marker='*', s=200, label=f'Min {eval_metric_col}')
    # for each K, make a circle where loss is minimum among that K
    unique_K = df_matched['K'].unique()
    for k in unique_K:
        df_k = df_matched[df_matched['K'] == k]
        min_loss_k_idx = df_k[eval_metric_col].idxmin()
        plt.scatter(df_k.loc[min_loss_k_idx, 'K'], df_k.loc[min_loss_k_idx, 'B'], 
                    #label=f'Min {eval_metric_col} for K={k}',
                     facecolors='none', edgecolors='red', s=100)
    
    # for each N, make a square where loss is minimum among that N
    unique_N = df_matched['N'].unique()
    for n in unique_N:
        df_n = df_matched[df_matched['N'] == n]
        min_loss_n_idx = df_n[eval_metric_col].idxmin()
        plt.scatter(df_n.loc[min_loss_n_idx, 'K'], df_n.loc[min_loss_n_idx, 'B'], 
                    #label=f'Min {eval_metric_col} for N={n}',
                    facecolors='none', edgecolors='blue', marker='s', s=100)
    

    # for each B, make a triangle where loss is minimum among that B
    unique_B = df_matched['B'].unique()
    for b in unique_B:
        df_b = df_matched[df_matched['B'] == b]
        min_loss_b_idx = df_b[eval_metric_col].idxmin()
        plt.scatter(df_b.loc[min_loss_b_idx, 'K'], df_b.loc[min_loss_b_idx, 'B'], 
                        #label=f'Min {eval_metric_col} for B={b}',
                        facecolors='none', edgecolors='green', marker='^', s=100)


    plt.legend()
    plt.savefig(f'K_vs_B_{eval_metric_col}_scatter_plot.png')
    plt.close()
    print(f"Out file {out_file} already exists with {len(df_existing)} rows.")
    

    # create a plot that is 3 by 3. For all three plots,
    # K on x-axis, B on y-axis, 
    # scatter points are all rows in df_matched
    # for the first plot, highlight min eval metric for each K with a circle
    # do this for both nqs_loss and nn_loss if both exist in df_matched;
    # use different colors for nn_loss (red) and nqs_loss (green) min evaluation points.

    # for the second plot, highlight min eval metric for each N with a square 
    # for the third plot, highlight min eval metric for each B with a triangle

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # use circle with no fill and grey edge for the scatters below
    scatter = axs[0].scatter(df_matched['K'], df_matched['B'], facecolors='grey', edgecolors='grey', alpha = 0.5)
   # fig.colorbar(scatter, ax=axs[0], label='NN_loss')
    scatter = axs[1].scatter(df_matched['K'], df_matched['B'], facecolors='grey', edgecolors='grey', alpha = 0.5)
   # fig.colorbar(scatter, ax=axs[1], label='NN_loss')
    scatter = axs[2].scatter(df_matched['K'], df_matched['B'], facecolors='grey', edgecolors='grey', alpha = 0.5)
   # fig.colorbar(scatter, ax=axs[2], label='NN_loss')
    for j in range(3):
        axs[j].set_xlabel('K')
        axs[j].set_ylabel('B')
        axs[j].set_xscale('log')
        axs[j].set_yscale('log')
    axs[0].set_title('Min NN_loss per K')
    axs[1].set_title('Min NN_loss per N')
    axs[2].set_title('Min NN_loss per B')

    # find global min nn_loss and nqs_loss
    global_min_nn_loss_idx = df_matched['NN_loss'].idxmin()
    axs[0].scatter(df_matched.loc[global_min_nn_loss_idx, 'K'], df_matched.loc[global_min_nn_loss_idx, 'B'], 
                   facecolors='red', edgecolors='red', marker='*', s=200, label='Global Min NN_loss')
    global_min_nqs_loss_idx = df_matched['nqs_loss'].idxmin()
    axs[0].scatter(df_matched.loc[global_min_nqs_loss_idx, 'K'], df_matched.loc[global_min_nqs_loss_idx, 'B'], 
                   facecolors='none', edgecolors='green', marker='*', s=700, label='Global Min NQS_loss')

    unique_K = df_matched['K'].unique()
    for k in unique_K:
    
        df_k = df_matched[df_matched['K'] == k]
        min_loss_k_idx = df_k['NN_loss'].idxmin()
        axs[0].scatter(df_k.loc[min_loss_k_idx, 'K'], df_k.loc[min_loss_k_idx, 'B'], 
                       facecolors='red', edgecolors='red', alpha=0.5, label='Min NN_loss')
        min_nqs_loss_k_idx = df_k['nqs_loss'].idxmin()
        axs[0].scatter(df_k.loc[min_nqs_loss_k_idx, 'K'], df_k.loc[min_nqs_loss_k_idx, 'B'], 
                       facecolors='none', edgecolors='green', s=100, alpha=1.0, label='Min NQS_loss')
    
  
    
    unique_N = df_matched['N'].unique()
    for n in unique_N:
        df_n = df_matched[df_matched['N'] == n]
        min_loss_n_idx = df_n['NN_loss'].idxmin()
        axs[1].scatter(df_n.loc[min_loss_n_idx, 'K'], df_n.loc[min_loss_n_idx, 'B'], 
                       facecolors='red', edgecolors='red',  alpha=0.5)
        min_nqs_loss_n_idx = df_n['nqs_loss'].idxmin()
        axs[1].scatter(df_n.loc[min_nqs_loss_n_idx, 'K'], df_n.loc[min_nqs_loss_n_idx, 'B'], 
                       facecolors='none', edgecolors='green',  s=100, alpha=1.0)
    unique_B = df_matched['B'].unique()
    for b in unique_B:
        df_b = df_matched[df_matched['B'] == b]
        min_loss_b_idx = df_b['NN_loss'].idxmin()
        axs[2].scatter(df_b.loc[min_loss_b_idx, 'K'], df_b.loc[min_loss_b_idx, 'B'], 
                       facecolors='red', edgecolors='red',  alpha=0.5, label='Min NN_loss')
        min_nqs_loss_b_idx = df_b['nqs_loss'].idxmin()
        axs[2].scatter(df_b.loc[min_nqs_loss_b_idx, 'K'], df_b.loc[min_nqs_loss_b_idx, 'B'], 
                       facecolors='none', edgecolors='green',  s=100, alpha=1.0, label='Min NQS_loss')
   
    # add legends
    #axs[0].legend()
    plt.savefig(f'constraints.png')
    plt.close()

    raise ValueError("Out file already exists. Exiting to avoid overwriting.")


# read base file
df_base = pd.read_csv(base_file)

# filter for rows where C = level
df_filtered = df_base[df_base['C'] == level]

# print rows
print(f"Filtered rows for C = {level}:")
print(df_filtered)


# get min max values of N, B, K
N_min = df_filtered['N'].min()
N_max = df_filtered['N'].max() * 2
B_min = df_filtered['B'].min()/2
B_max = df_filtered['B'].max()
K_min = df_filtered['K'].min()
K_max = df_filtered['K'].max() * 2 * 2 * 2  # double K_max to expand range

N_dense_max = 256000000
N_dense_min = 512000000 #64000000

B_dense_max = 1000
B_dense_min = 2000 #48

K_dense_max = 50 * 1024
K_dense_min = 50 * 2048 #50 * 128

# display min max values
print(f"N: min={N_min}, max={N_max}")
print(f"B: min={B_min}, max={B_max}")
print(f"K: min={K_min}, max={K_max}")

# round the B min/max to nearest multiple of 256
B_min_rounded = int(np.floor(B_min / 256) * 256)
B_max_rounded = int(np.ceil(B_max / 256) * 256)

# round the K min/max to nearest multiple of 50
K_min_rounded = int(np.floor(K_min / 50) * 50)
K_max_rounded = int(np.ceil(K_max / 50) * 50)
# display rounded min max values
print(f"B rounded: min={B_min_rounded}, max={B_max_rounded}")
print(f"K rounded: min={K_min_rounded}, max={K_max_rounded}")

# create a grid with N, B, K values log-spaced
# for N, double each step from N_min to N_max
N_values = []
n = N_min
while n <= N_max:
    N_values.append(n)
    n *= 2
n = N_min
while n <= N_max and n <= N_dense_max:
    if n < N_dense_min:
        # do nothing
        pass
    # check if n is within 10% of existing N values
    elif all(abs(n - existing_n) / existing_n > 0.1 for existing_n in N_values):
        N_values.append(n)
    n = int(np.round(n * np.sqrt(2),0))
N_values = list(set(N_values))  # remove duplicates
N_values.sort()
#raise ValueError(f"N values: {N_values}")

# for B, double each step from B_min_rounded to B_max_rounded
# start at 12
B_values = []
b = 12
while b <= B_max_rounded:
    B_values.append(b)
    b *= 2
B_values.append(b)  # ensure max is included
# remove B values smaller than B_min_rounded
B_values = [b for b in B_values if b >= B_min_rounded/2]
b = 12
while b <= B_max_rounded and b <= B_dense_max:
    if b < B_dense_min:
        # do nothing
        pass
    # check if b is within 10% of existing B values
    elif all(abs(b - existing_b) / existing_b > 0.1 for existing_b in B_values):
        B_values.append(b)
    b = int(np.round(b * np.sqrt(2),0))
B_values = list(set(B_values))  # remove duplicates
B_values.sort()
#raise ValueError(f"B values: {B_values}")

# for K, double each step from K_min_rounded to K_max_rounded
K_values = []
k = 50
while k <= K_max_rounded:

    K_values.append(k)
    k *= 2
K_values.append(k)  # ensure max is included
# remove K values smaller than K_min_rounded

k = 50
while k <= K_max_rounded and k <= K_dense_max:
    if k < K_dense_min:
        # do nothing
        pass
    # check if k is within 10% of existing K values
    elif all(abs(k - existing_k) / existing_k > 0.2 for existing_k in K_values):
        K_values.append(k)
    k = int(np.round(k * np.sqrt(2),0))
K_values = [k for k in K_values if k <= K_max_rounded * 2]
K_values = list(set(K_values))  # remove duplicates

K_values = [k for k in K_values if k >= 1000]
K_values.sort()
#raise ValueError(f"K values: {K_values}")

# display generated values
print(f"N values: {N_values}")
print(f"B values: {B_values}")
print(f"K values: {K_values}")

# create a table with all combinations of N, B, K
rows = []
for N in N_values:
    for B in B_values:
        for K in K_values:
            rows.append({'N': N, 'B': B, 'K': K})

# compute NBK product 
for row in rows:
    row['C'] = row['N'] * row['B']/1e9 * row['K'] * 6 * 128
    row['D'] = row['B'] * row['K'] * 128
    row['actual_N'] = row['N']
# print the first 10 rows
print("First 10 rows with computed C values:")
for row in rows[:10]:
    print(row)
# filter for where NBK is between 90% and 110% of level
rows_filtered = [row for row in rows if 0.90 * level <= row['C'] <= 1.10   * level]
# display filtered rows
print(f"Filtered rows for NBK within 90%-110% of {level}:")
for row in rows_filtered:
    print(row)

# make a scatter plot of B vs K (K on the x-axis, B on the y-axis)
import matplotlib.pyplot as plt

B_values_filtered = [row['B'] for row in rows_filtered]
K_values_filtered = [row['K'] for row in rows_filtered]

plt.scatter(K_values_filtered, B_values_filtered)
plt.xlabel('K')
plt.ylabel('B')
plt.title('Scatter plot of B vs K for filtered rows')

# make log-log scale
plt.xscale('log')
plt.yscale('log')
# save log-log plot
# save plot
plt.savefig('h_samples_resource_B_vs_K_scatter_plot.png')
plt.close()

# similarly, make a plot for B_vs_N
N_values_filtered = [row['N'] for row in rows_filtered]
plt.scatter(N_values_filtered, B_values_filtered)
plt.xlabel('N')
plt.ylabel('B')
plt.title('Scatter plot of B vs N for filtered rows')
# make log-log scale
plt.xscale('log')
plt.yscale('log')
# save log-log plot
plt.savefig('h_samples_resource_B_vs_N_scatter_plot.png')
plt.close()

write_to_file = True
if write_to_file:
    # now write the filtered rows to out_file, use values from df_base for other columns

    # first, get all column names from df_base
    column_names = df_base.columns.tolist()
    # for each column name not in N, B, K, C, actual_N set a default value from the first row of df_base
    default_values = df_base.iloc[0].to_dict()
    # now write to out_file
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_names)
        writer.writeheader()
        for row in rows_filtered:
            out_row = default_values.copy()
            out_row.update(row)
            writer.writerow(out_row)
    print(f"Wrote filtered configurations to {out_file}")
    # print number of rows written
    print(f"Number of configurations written: {len(rows_filtered)}")
