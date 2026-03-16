import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# OWT data
# --------------------------------
# nqs fitted on all isoflops: "nqs_loss" column, from
#     scale_new/outputs/runs/Paper/2026-01-23-19-07_openwebtext2_pythia_adam_cosine_all_isoflops_nqs/5_resource_allocation/h_samples_with_nqs_loss.csv
# nn on all isoflops: "NN_loss" column, from
#     /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-07_openwebtext2_pythia_adam_cosine_all_isoflops_nqs/5_resource_allocation/to_get_nn_samples.csv
# nqs/nn fitted on all isotokens: "NN_loss" column and "nqs_loss" column, from
#     /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-14_openwebtext2_pythia_adam_cosine_all_isotokens_nqs/6_critical_batch_size/h_samples_with_nn_loss.csv
# chin fitted on all isoflops: "chin_loss" column, from
#    /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-23_openwebtext2_pythia_adam_cosine_chin_eval/4_loss_estimation/chinchilla_2/eval_df.csv
# chin fitted on all isotokens: "chin_loss" column, from
#    /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-20-36_openwebtext2_pythia_adam_cosine_chin_on_isotoken/4_loss_estimation/chinchilla_2/eval_df.csv
# chin fitted on isoflops and isotokens
# 

# LM1B data
# --------------------------------
# nqs fitted on all isoflops: "nqs_loss" column, from
#     /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-17-13_lm1b_llama_adam_cosine_exhibits/5_resource_allocation/h_samples_with_nqs_loss.csv
# nn on all isoflops: "NN_loss" column, from
#     /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-17-13_lm1b_llama_adam_cosine_exhibits/5_resource_allocation/to_get_nn_samples.csv
# nqs/nn fitted on all isotokens: "NN_loss" column and "nqs_loss" column, from
#     /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-17-13_lm1b_llama_adam_cosine_exhibits/6_critical_batch_size/h_samples_with_nn_loss.csv
# chin fitted on all isoflops: "chin_loss" column, from
#    /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-20-56_lm1b_llama_adam_cosine_chin_eval/4_loss_estimation/chinchilla_2/eval_df.csv
# chin fitted on all isotokens: "chin_loss" column, from
#    /mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-20-58_lm1b_llama_adam_cosine_chin_on_isotoken/4_loss_estimation/chinchilla_2/eval_df.csv
#
# NOTE: for LM1B isoflop, define train/test/mid splits as:  train: C <= 100PF, test: C > 100PF, no mid split


# ============ HELPER FUNCTIONS ============

def row_huber_loss(y_true, y_pred, delta=1e-3):
    """Compute huber loss for a single row: huber_loss(log(y_true), log(y_pred))"""
    if pd.isna(y_true) or pd.isna(y_pred):
        return np.nan
    diff = np.log(y_true) - np.log(y_pred)
    if np.abs(diff) <= delta:
        return 0.5 * diff**2
    else:
        return delta * (np.abs(diff) - 0.5 * delta)

def custom_huber_loss(y_true, y_pred, delta=1e-3):
    diff = y_true - y_pred
    cond = np.abs(diff) <= delta
    loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    return np.sum(loss)

def avg_huber_loss(y_true, y_pred, delta=1e-3):
    """Compute average Huber loss: 1/n * sum(huber_loss(log(y_true), log(y_pred)))"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    log_true = np.log(y_true)
    log_pred = np.log(y_pred)
    return custom_huber_loss(log_true, log_pred, delta) / len(y_true)

def safe_avg_huber_loss(df_subset, true_col, pred_col, delta=1e-3):
    """Compute average Huber loss with safeguard for empty or missing data."""
    df_valid = df_subset.dropna(subset=[true_col, pred_col])
    if len(df_valid) == 0:
        return np.nan
    return avg_huber_loss(df_valid[true_col].values, df_valid[pred_col].values, delta)

def safe_avg_mse(df_subset, true_col, pred_col):
    """Compute average MSE on log scale with safeguard for empty or missing data."""
    df_valid = df_subset.dropna(subset=[true_col, pred_col])
    if len(df_valid) == 0:
        return np.nan
    log_true = np.log(df_valid[true_col].values)
    log_pred = np.log(df_valid[pred_col].values)
    return np.mean((log_true - log_pred) ** 2)

def format_C_label(c):
    """Format C value as PetaFLOPs (PF)."""
    c_pf = c / 1e6  # C is in GFLOPs, convert to PF
    if c_pf >= 1:
        return f"  {int(round(c_pf))} PF"
    else:
        return f"  {c_pf:.2f} PF"


def process_dataset(dataset_name, paths, split_config, output_prefix):
    """
    Process a dataset and generate plots and tables.

    Args:
        dataset_name: Name of the dataset (e.g., "OWT", "LM1B")
        paths: Dict with keys: nqs_isoflops, nn_isoflops, isotokens, chin_isoflops, chin_isotokens
        split_config: Dict with keys: train_threshold, test_threshold, has_mid
                      For OWT: train <= 60PF, mid 60-1000PF, test > 1000PF
                      For LM1B: train <= 100PF, test > 100PF, no mid
        output_prefix: Prefix for output files
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*60}")

    # Load data
    df_nqs_isoflops = pd.read_csv(paths['nqs_isoflops'])
    df_nn_isoflops = pd.read_csv(paths['nn_isoflops'])
    df_isotokens = pd.read_csv(paths['isotokens'])
    df_chin_isoflops = pd.read_csv(paths['chin_isoflops'])
    df_chin_isotokens = pd.read_csv(paths['chin_isotokens'])

    # Build isoflop dataframe
    # Handle case where 'actual_N' may not exist - use 'N' if not present
    isoflop_cols = ['N', 'B', 'K', 'C', 'D', 'nqs_loss']
    if 'actual_N' in df_nqs_isoflops.columns:
        isoflop_cols.insert(4, 'actual_N')
    df_isoflops = df_nqs_isoflops[isoflop_cols].merge(
        df_nn_isoflops[['N', 'B', 'K', 'NN_loss']], on=['N', 'B', 'K'], how='outer'
    )
    df_isoflops = df_isoflops.rename(columns={'NN_loss': 'nn_loss'})
    if 'actual_N' not in df_isoflops.columns:
        df_isoflops['actual_N'] = df_isoflops['N']
    df_isoflops['type'] = 'isoflop'

    # Build isotoken dataframe
    isotoken_cols = ['N', 'B', 'K', 'C', 'D', 'nqs_loss', 'NN_loss']
    if 'actual_N' in df_isotokens.columns:
        isotoken_cols.insert(4, 'actual_N')
    df_isotokens = df_isotokens[isotoken_cols]
    df_isotokens = df_isotokens.rename(columns={'NN_loss': 'nn_loss'})
    if 'actual_N' not in df_isotokens.columns:
        df_isotokens['actual_N'] = df_isotokens['N']
    df_isotokens['type'] = 'isotoken'

    # Merge chin_loss for isoflops
    df_chin_isoflops = df_chin_isoflops[['N', 'B', 'K', 'chin_loss']]
    df_isoflops = df_isoflops.merge(df_chin_isoflops, on=['N', 'B', 'K'], how='left')

    # Merge chin_loss for isotokens
    df_chin_isotokens = df_chin_isotokens[['N', 'B', 'K', 'chin_loss']]
    df_isotokens = df_isotokens.merge(df_chin_isotokens, on=['N', 'B', 'K'], how='left')

    # Concatenate isoflop and isotoken rows
    df = pd.concat([df_isoflops, df_isotokens], ignore_index=True)

    # Assign split based on config
    train_threshold = split_config['train_threshold']  # in GFLOPs
    test_threshold = split_config['test_threshold']    # in GFLOPs
    has_mid = split_config['has_mid']

    def assign_split(row):
        c = row['C']
        typ = row['type']
        if typ == 'isoflop':
            if c > test_threshold:
                return 'test'
            elif has_mid and c > train_threshold:
                return 'mid'
            elif c > train_threshold:
                return 'test'
            else:
                return 'train'
        else:  # isotoken - use train_threshold for isotoken split
            if c > train_threshold:
                return 'test'
            else:
                return 'train'

    df['split'] = df.apply(assign_split, axis=1)

    # Compute per-row huber loss
    df['nqs_huber_loss'] = df.apply(lambda row: row_huber_loss(row['nn_loss'], row['nqs_loss']), axis=1)
    df['chin_huber_loss'] = df.apply(lambda row: row_huber_loss(row['nn_loss'], row['chin_loss']), axis=1)

    # Save the processed csv
    output_path = f"/mfs1/u/chuning/scale_new/{output_prefix}_raw_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved processed table to {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(df[df['split'] == 'train'])}")
    if has_mid:
        print(f"Mid rows: {len(df[df['split'] == 'mid'])}")
    print(f"Test rows: {len(df[df['split'] == 'test'])}")
    print(f"\nColumns: {list(df.columns)}")

    # ============ PLOTTING ============
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))

    # Colors: blue for train, orange for mid, red for test
    color_map = {'train': 'blue', 'mid': 'orange', 'test': 'red'}

    # LEFT PANEL: IsoFLOP plot
    df_isoflop = df[df['type'] == 'isoflop'].copy()
    unique_C_isoflop = sorted(df_isoflop['C'].unique())

    for c_val in unique_C_isoflop:
        df_c = df_isoflop[df_isoflop['C'] == c_val].copy()
        df_c = df_c.sort_values('actual_N')

        split = df_c['split'].iloc[0]
        color = color_map[split]

        ax_left.scatter(df_c['actual_N'], df_c['nn_loss'], color=color, marker='o', s=40, alpha=0.8, ec='none')
        ax_left.plot(df_c['actual_N'], df_c['nqs_loss'], color=color, linestyle='-', linewidth=2, alpha=0.7)

        df_c_chin = df_c.dropna(subset=['chin_loss'])
        if len(df_c_chin) > 0:
            ax_left.plot(df_c_chin['actual_N'], df_c_chin['chin_loss'], color=color, linestyle='--', linewidth=2, alpha=0.7)

        if len(df_c) > 0:
            label_C = format_C_label(c_val)
            ax_left.text(df_c['actual_N'].values[-1] * 1.05, df_c['nqs_loss'].values[-1],
                         label_C, color=color, fontsize=9, fontweight='bold', va='center')

    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_xlabel("N (Parameters)", fontweight='bold')
    ax_left.set_ylabel("Loss", fontweight='bold')
    ax_left.set_title(f"(a) {dataset_name} IsoFLOP Curves", fontweight='bold')

    x_min, x_max = ax_left.get_xlim()
    y_min, y_max = ax_left.get_ylim()
    ax_left.set_xlim(x_min * 0.5, x_max * 2.5)
    ax_left.set_ylim(y_min * 0.98, y_max * 1.05)

    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['left'].set_linewidth(2)
    ax_left.spines['bottom'].set_linewidth(2)

    # RIGHT PANEL: Isotoken plot
    df_isotoken = df[df['type'] == 'isotoken'].copy()
    unique_C_isotoken = sorted(df_isotoken['C'].unique())

    for c_val in unique_C_isotoken:
        df_c = df_isotoken[df_isotoken['C'] == c_val].copy()
        df_c = df_c.sort_values('B')

        split = df_c['split'].iloc[0]
        color = color_map[split]

        ax_right.scatter(df_c['B'], df_c['nn_loss'], color=color, marker='o', s=40, alpha=0.8, ec='none')
        ax_right.plot(df_c['B'], df_c['nqs_loss'], color=color, linestyle='-', linewidth=2, alpha=0.7)

        df_c_chin = df_c.dropna(subset=['chin_loss'])
        if len(df_c_chin) > 0:
            ax_right.plot(df_c_chin['B'], df_c_chin['chin_loss'], color=color, linestyle='--', linewidth=2, alpha=0.7)

        if len(df_c) > 0:
            label_C = format_C_label(c_val)
            ax_right.text(df_c['B'].values[-1] * 1.05, df_c['nqs_loss'].values[-1],
                          label_C, color=color, fontsize=9, fontweight='bold', va='center')

    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.set_xlabel("Batch Size (B)", fontweight='bold')
    ax_right.set_ylabel("Loss", fontweight='bold')
    ax_right.set_title(f"(b) {dataset_name} Isotoken Curves", fontweight='bold')

    x_min, x_max = ax_right.get_xlim()
    y_min, y_max = ax_right.get_ylim()
    ax_right.set_xlim(x_min * 0.5, x_max * 2.5)
    ax_right.set_ylim(y_min * 0.98, y_max * 1.05)

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_linewidth(2)
    ax_right.spines['bottom'].set_linewidth(2)

    # LEGEND
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label='NN (actual)'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2, alpha=0.7, label='NQS'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Chinchilla'),
        Line2D([0], [0], color='blue', linestyle='-', linewidth=4, label='Train'),
    ]
    if has_mid:
        legend_elements.append(Line2D([0], [0], color='orange', linestyle='-', linewidth=4, label='Mid'))
    legend_elements.append(Line2D([0], [0], color='red', linestyle='-', linewidth=4, label='Test'))

    ncol = 6 if has_mid else 5
    fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=ncol, prop={'weight': 'bold'})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    output_plot = f"/mfs1/u/chuning/scale_new/{output_prefix}_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.savefig(output_plot.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"\nSaved plot to {output_plot}")

    # ============ TABLE MAKING ============
    df_train = df[df['split'] == 'train']
    df_test_isoflop = df[(df['split'] == 'test') & (df['type'] == 'isoflop')]
    df_test_isotoken = df[(df['split'] == 'test') & (df['type'] == 'isotoken')]
    df_train_isoflop = df_train[df_train['type'] == 'isoflop']

    if has_mid:
        df_mid_isoflop = df[(df['split'] == 'mid') & (df['type'] == 'isoflop')]

    # NQS metrics
    nqs_train = safe_avg_huber_loss(df_train, 'nn_loss', 'nqs_loss')
    nqs_test_isoflop = safe_avg_huber_loss(df_test_isoflop, 'nn_loss', 'nqs_loss')
    nqs_test_isotoken = safe_avg_huber_loss(df_test_isotoken, 'nn_loss', 'nqs_loss')

    # Chinchilla metrics
    chin_train = safe_avg_huber_loss(df_train_isoflop, 'nn_loss', 'chin_loss')
    chin_test_isoflop = safe_avg_huber_loss(df_test_isoflop, 'nn_loss', 'chin_loss')
    chin_test_isotoken = safe_avg_huber_loss(df_test_isotoken, 'nn_loss', 'chin_loss')

    # Create the summary table
    if has_mid:
        nqs_mid_isoflop = safe_avg_huber_loss(df_mid_isoflop, 'nn_loss', 'nqs_loss')
        chin_mid_isoflop = safe_avg_huber_loss(df_mid_isoflop, 'nn_loss', 'chin_loss')
        table_data = {
            'Model': ['NQS', 'Chinchilla'],
            'Train': [nqs_train, chin_train],
            'Mid IsoFLOP': [nqs_mid_isoflop, chin_mid_isoflop],
            'Test IsoFLOP': [nqs_test_isoflop, chin_test_isoflop],
            'Test IsoToken': [nqs_test_isotoken, chin_test_isotoken]
        }
    else:
        table_data = {
            'Model': ['NQS', 'Chinchilla'],
            'Train': [nqs_train, chin_train],
            'Test IsoFLOP': [nqs_test_isoflop, chin_test_isoflop],
            'Test IsoToken': [nqs_test_isotoken, chin_test_isotoken]
        }

    df_table = pd.DataFrame(table_data)

    print(f"\n{'='*60}")
    print(f"{dataset_name} Table 1: Average Huber Loss (log scale, delta=1e-3)")
    print(f"{'='*60}")
    print(df_table.to_string(index=False, float_format=lambda x: f"{x:.6f}" if not np.isnan(x) else "N/A"))
    print(f"{'='*60}")

    table_output = f"/mfs1/u/chuning/scale_new/{output_prefix}_summary.csv"
    df_table.to_csv(table_output, index=False)
    print(f"\nSaved summary table to {table_output}")

    # MSE Summary Table
    nqs_train_mse = safe_avg_mse(df_train, 'nn_loss', 'nqs_loss')
    nqs_test_isoflop_mse = safe_avg_mse(df_test_isoflop, 'nn_loss', 'nqs_loss')
    nqs_test_isotoken_mse = safe_avg_mse(df_test_isotoken, 'nn_loss', 'nqs_loss')

    chin_train_mse = safe_avg_mse(df_train_isoflop, 'nn_loss', 'chin_loss')
    chin_test_isoflop_mse = safe_avg_mse(df_test_isoflop, 'nn_loss', 'chin_loss')
    chin_test_isotoken_mse = safe_avg_mse(df_test_isotoken, 'nn_loss', 'chin_loss')

    if has_mid:
        nqs_mid_isoflop_mse = safe_avg_mse(df_mid_isoflop, 'nn_loss', 'nqs_loss')
        chin_mid_isoflop_mse = safe_avg_mse(df_mid_isoflop, 'nn_loss', 'chin_loss')
        table_data_mse = {
            'Model': ['NQS', 'Chinchilla'],
            'Train': [nqs_train_mse, chin_train_mse],
            'Mid IsoFLOP': [nqs_mid_isoflop_mse, chin_mid_isoflop_mse],
            'Test IsoFLOP': [nqs_test_isoflop_mse, chin_test_isoflop_mse],
            'Test IsoToken': [nqs_test_isotoken_mse, chin_test_isotoken_mse]
        }
    else:
        table_data_mse = {
            'Model': ['NQS', 'Chinchilla'],
            'Train': [nqs_train_mse, chin_train_mse],
            'Test IsoFLOP': [nqs_test_isoflop_mse, chin_test_isoflop_mse],
            'Test IsoToken': [nqs_test_isotoken_mse, chin_test_isotoken_mse]
        }

    df_table_mse = pd.DataFrame(table_data_mse)

    print(f"\n{'='*60}")
    print(f"{dataset_name} Table 2: Average MSE (log scale)")
    print(f"{'='*60}")
    print(df_table_mse.to_string(index=False, float_format=lambda x: f"{x:.6f}" if not np.isnan(x) else "N/A"))
    print(f"{'='*60}")

    table_output_mse = f"/mfs1/u/chuning/scale_new/{output_prefix}_summary_mse.csv"
    df_table_mse.to_csv(table_output_mse, index=False)
    print(f"\nSaved MSE summary table to {table_output_mse}")

    return df


# ============ OWT Dataset ============
owt_paths = {
    'nqs_isoflops': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-07_openwebtext2_pythia_adam_cosine_all_isoflops_nqs/5_resource_allocation/h_samples_with_nqs_loss.csv",
    'nn_isoflops': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-07_openwebtext2_pythia_adam_cosine_all_isoflops_nqs/5_resource_allocation/to_get_nn_samples.csv",
    'isotokens': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-14_openwebtext2_pythia_adam_cosine_all_isotokens_nqs/6_critical_batch_size/h_samples_with_nn_loss.csv",
    'chin_isoflops': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-19-23_openwebtext2_pythia_adam_cosine_chin_eval/4_loss_estimation/chinchilla_2/eval_df.csv",
    'chin_isotokens': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-20-36_openwebtext2_pythia_adam_cosine_chin_on_isotoken/4_loss_estimation/chinchilla_2/eval_df.csv",
}

# OWT split: train <= 60PF, mid 60-1000PF, test > 1000PF for isoflop; train <= 60PF, test > 60PF for isotoken
owt_split_config = {
    'train_threshold': 60000000,      # 60 PF in GFLOPs
    'test_threshold': 1000000000,     # 1000 PF in GFLOPs
    'has_mid': True
}

df_owt = process_dataset("OWT", owt_paths, owt_split_config, "table1_owt")


# ============ LM1B Dataset ============
lm1b_paths = {
    'nqs_isoflops': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-17-13_lm1b_llama_adam_cosine_exhibits/5_resource_allocation/h_samples_with_nqs_loss.csv",
    'nn_isoflops': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-17-13_lm1b_llama_adam_cosine_exhibits/5_resource_allocation/to_get_nn_samples.csv",
    'isotokens': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-17-13_lm1b_llama_adam_cosine_exhibits/6_critical_batch_size/h_samples_with_nn_loss.csv",
    'chin_isoflops': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-20-56_lm1b_llama_adam_cosine_chin_eval/4_loss_estimation/chinchilla_2/eval_df.csv",
    'chin_isotokens': "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-23-20-58_lm1b_llama_adam_cosine_chin_on_isotoken/4_loss_estimation/chinchilla_2/eval_df.csv",
}

# LM1B split: train <= 100PF, test > 100PF, no mid
lm1b_split_config = {
    'train_threshold': 100000000,     # 100 PF in GFLOPs
    'test_threshold': 100000000,      # same as train threshold (no mid)
    'has_mid': False
}

df_lm1b = process_dataset("LM1B", lm1b_paths, lm1b_split_config, "table1_lm1b")
