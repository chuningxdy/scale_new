# create csv files where each row is a config for NN/NQS
import os
import csv
import pandas as pd
import numpy as np


# base file
base_file = 'h_samples_resource_allocation_owt_all_balanced_adam_cosine.csv'
level = 236000000

# out file
out_file = 'h_samples_test_nqs_owt_adam_cosine_NBK_corrected.csv'
# check if out file exists
if os.path.exists(out_file):
    # read the file
    df_existing = pd.read_csv(out_file)

    #40
    out_path = "/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-04-23-23_openwebtext2_pythia_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv"
    #"/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-02-17-18_openwebtext2_pythia_adam_cosine/4_loss_estimation/eval_df.csv" # LRA old
    #SN:"/mfs1/u/chuning/scale_new/outputs/runs/LRA_investigation/2026-01-01-19-49_openwebtext2_pythia_adam_cosine/4_loss_estimation/eval_df.csv"

    # left join with existing file to see if any rows match
    df_out = pd.read_csv(out_path)
    df_merged = pd.merge(df_existing, df_out, how='left', on=['B', 'K', 'N'], suffixes=('_existing', '_eval'))
    # rename N_existing to N
    #df_merged = df_merged.rename(columns={'N_existing': 'N'})
    # filter for rows where eval metric is not null
    eval_metric_col = 'nqs_loss'
    df_matched = df_merged[~df_merged[eval_metric_col].isnull()]
    print(f"Number of matching rows with eval metric: {len(df_matched)}")
    if len(df_matched) == 0:
        raise ValueError("No matching rows found. Exiting.")
    # make a plot of N on x-axis and B on y-axis
    # color is "loss" column from eval_df.csv
    import matplotlib.pyplot as plt
    plt.scatter(df_matched['K'], df_matched['B'], c=df_matched[eval_metric_col], cmap='viridis')
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
K_max = df_filtered['K'].max() * 2  # double K_max to expand range

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

# for K, double each step from K_min_rounded to K_max_rounded
K_values = []
k = 50
while k <= K_max_rounded:
    K_values.append(k)
    k *= 2
K_values.append(k)  # ensure max is included
# remove K values smaller than K_min_rounded
K_values = [k for k in K_values if k >= K_min_rounded/2]

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
rows_filtered = [row for row in rows if 0.9 * level <= row['C'] <= 1.1   * level]
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
