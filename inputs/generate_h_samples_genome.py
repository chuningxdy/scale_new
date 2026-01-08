# create csv files where each row is a config for NN/NQS
import os
import csv
import pandas as pd
import numpy as np


# base file
base_file = 'h_samples_critical_batch_size_owt_all_balanced_adam_cosine_train.csv'

#level = 236000000/4

# out file
out_file = 'h_samples_critical_batch_size_genomes_all_balanced_adam_cosine_train.csv'

# read base file
df_base = pd.read_csv(base_file)

# compute token to parameter ratio R = 128*B*K/N
df_base['R'] = 128 * df_base['B'] * df_base['K'] / df_base['N']
# summarize R
print("Token to parameter ratio R summary:")
print(df_base['R'].describe())
# print R for the first 10 rows, only display N,B,K,R
print("First 10 rows of N, B, K, R:")
print(df_base[['N', 'B', 'K', 'R']].head(10))
#raise ValueError("Stopping after R summary")

#raise ValueError("Stopping after R summary")
# filter for rows where R > 10
df_filtered = df_base [df_base['R'] > 2]
print(f"Filtered base file to R > 2, new shape: {df_filtered.shape}")

# decrease N by a factor of 2 and increase K by a factor of 2
df_filtered['N'] = df_filtered['N'] // 2
df_filtered['K'] = df_filtered['K'] * 2

# divide all K values by 2048/128 (sequence lenghth goes from 128 to 2048, maintain same R)

df_filtered['B'] = df_filtered['B'] // (2048 // 128)
print("Adjusted K values for sequence length change from 128 to 2048.")

# recompute D = B * K * 2048
df_filtered['D'] = df_filtered['B'] * df_filtered['K'] * 2048
# compute R as D / N
df_filtered['R'] = df_filtered['D'] / df_filtered['N']
# summarize R again
print("Token to parameter ratio R summary after K adjustment:")
print(df_filtered['R'].describe())
# display R for the first 10 rows, only display N,B,K,R
print("First 10 rows of N, B, K, R after K adjustment:")
print(df_filtered[['N', 'B', 'K', 'R']].head(10))

# display R for the last 10 rows, only display N,B,K,R
print("Last 10 rows of N, B, K, R after K adjustment:")
print(df_filtered[['N', 'B', 'K', 'R']].tail(10))

# drop D
df_filtered = df_filtered.drop(columns=['D'])

rows_filtered = df_filtered.to_dict(orient='records')



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
plt.savefig('h_samples_resource_B_vs_K_scatter_plot_genomes.png')
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
plt.savefig('h_samples_resource_B_vs_N_scatter_plot_genomes.png')
plt.close()


# check if out_file exists
if os.path.exists(out_file):
    print(f"Output file {out_file} already exists. Exiting to avoid overwrite.")
    exit(1)

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
