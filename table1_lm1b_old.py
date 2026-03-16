import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# OWT data
# --------------------------------
# Base folder contains all the data files
# nqs fitted on all isoflops: "nqs_loss" column, from {base_folder}/5_resource_allocation/h_samples_with_nqs_loss.csv
# nn on all isoflops: "NN_loss" column, from {base_folder}/5_resource_allocation/to_get_nn_samples.csv
# nqs/nn fitted on all isotokens: "NN_loss" column and "nqs_loss" column, from {base_folder}/6_critical_batch_size/h_samples_with_nn_loss.csv
# chin fitted on all (isoflops and isotokens): "chin_loss" column, from chin_folder/4_loss_estimation/chinchilla_2/eval_df.csv


# ============ CONFIGURATION ============
# Base folder containing all data (NQS isoflop, isotoken, and chin)
BASE_FOLDER = "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-15-29_lm1b_llama_adam_cosine_BKHeldout"
#"/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-03-49_openwebtext2_pythia_adam_cosine_81920PF"
#"/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-03-45_openwebtext2_pythia_adam_cosine_20480PF"
#"/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-03-06_openwebtext2_pythia_adam_cosine_5120PF"
#"/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-02-21_openwebtext2_pythia_adam_cosine_320PF"
#"/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-02-08_openwebtext2_pythia_adam_cosine_20PF"
#"/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-01-52_openwebtext2_pythia_adam_cosine_complete"

# Output prefix for files
OUTPUT_PREFIX = "table1_lm1b_new"

# Train/test split threshold (in GFLOPs)
# - IsoFLOP train: C <= TRAIN_THRESHOLD
# - IsoFLOP mid: TRAIN_THRESHOLD < C <= 1000PF
# - IsoFLOP test: C > 1000PF (fixed)
# - Isotoken train: C <= TRAIN_THRESHOLD
# - Isotoken test: C > TRAIN_THRESHOLD
TRAIN_THRESHOLD = 60000000  # 60 PF in GFLOPs

# Internal split configuration (test_threshold for isoflop is fixed at 1000PF)
SPLIT_CONFIG = {
    'train_threshold': TRAIN_THRESHOLD,
    'test_threshold': TRAIN_THRESHOLD,     # 1000 PF in GFLOPs (fixed for isoflop test)
    'has_mid': False
}


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


def get_paths(base_folder):
    """Generate file paths from base folder."""
    return {
        'nqs_isoflops': f"{base_folder}/5_resource_allocation/h_samples_with_nqs_loss.csv",
        'nn_isoflops': f"{base_folder}/5_resource_allocation/to_get_nn_samples.csv",
        'isotokens': f"{base_folder}/6_critical_batch_size/h_samples_with_nn_loss.csv",
        'chin_all': f"{base_folder}/4_loss_estimation/chinchilla_2/eval_df.csv",
    }


def process_owt_dataset(base_folder, split_config, output_prefix):
    """
    Process OWT dataset and generate plots and tables.

    Args:
        base_folder: Base folder containing all data (NQS isoflop, isotoken, chin)
        split_config: Dict with keys: train_threshold, test_threshold, has_mid
        output_prefix: Prefix for output files
    """
    dataset_name = "OWT"
    paths = get_paths(base_folder)

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*60}")
    print(f"Base folder: {base_folder}")

    # Load data
    df_nqs_isoflops = pd.read_csv(paths['nqs_isoflops'])
    df_nn_isoflops = pd.read_csv(paths['nn_isoflops'])
    df_isotokens = pd.read_csv(paths['isotokens'])
    df_chin_all = pd.read_csv(paths['chin_all'])

    # Build isoflop dataframe
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

    # Concatenate isoflop and isotoken rows first
    df = pd.concat([df_isoflops, df_isotokens], ignore_index=True)

    # Merge chin_loss from single file (for both isoflop and isotoken)
    df_chin_all = df_chin_all[['N', 'B', 'K', 'chin_loss']]
    df = df.merge(df_chin_all, on=['N', 'B', 'K'], how='left')

    # Assign split indicators based on config (allows overlapping categories)
    train_threshold = split_config['train_threshold']
    test_threshold = split_config['test_threshold']
    has_mid = split_config['has_mid']

    def assign_train_ind(row):
        """Train: C <= train_threshold (for both isoflop and isotoken)"""
        return row['C'] <= train_threshold

    def assign_test_ind(row):
        """Test: C > test_threshold for isoflop, C > train_threshold for isotoken"""
        if row['type'] == 'isoflop':
            return row['C'] > test_threshold
        else:  # isotoken
            return row['C'] > train_threshold

    def assign_mid_ind(row):
        """Mid: train_threshold < C <= test_threshold for isoflop only"""
        if row['type'] == 'isoflop':
            return (row['C'] > train_threshold) & (row['C'] <= test_threshold)
        else:
            return False

    df['train_ind'] = df.apply(assign_train_ind, axis=1)
    df['test_ind'] = df.apply(assign_test_ind, axis=1)
    df['mid_ind'] = df.apply(assign_mid_ind, axis=1)

    # For backwards compatibility, also create a 'split' column (primary category for plotting)
    def assign_split(row):
        # Priority: test > mid > train for coloring
        if row['test_ind']:
            return 'test'
        elif row['mid_ind']:
            return 'mid'
        else:
            return 'train'

    df['split'] = df.apply(assign_split, axis=1)

    # Compute per-row huber loss
    df['nqs_huber_loss'] = df.apply(lambda row: row_huber_loss(row['nn_loss'], row['nqs_loss']), axis=1)
    df['chin_huber_loss'] = df.apply(lambda row: row_huber_loss(row['nn_loss'], row['chin_loss']), axis=1)

    # Save the processed csv
    output_path = BASE_FOLDER + f"/{output_prefix}_raw_data.csv"
    #f"/mfs1/u/chuning/scale_new/{output_prefix}_raw_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved processed table to {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Train rows (train_ind=True): {len(df[df['train_ind']])}")
    if has_mid:
        print(f"Mid rows (mid_ind=True): {len(df[df['mid_ind']])}")
    print(f"Test rows (test_ind=True): {len(df[df['test_ind']])}")
    print(f"Rows in both train and test: {len(df[df['train_ind'] & df['test_ind']])}")
    print(f"\nColumns: {list(df.columns)}")

    # ============ PLOTTING ============
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))

    # Define which indicators to plot and their colors (order matters for layering)
    ind_color_map = [('train_ind', 'blue'), ('mid_ind', 'orange'), ('test_ind', 'red')]

    # LEFT PANEL: IsoFLOP plot
    df_isoflop = df[df['type'] == 'isoflop'].copy()
    unique_C_isoflop = sorted(df_isoflop['C'].unique())

    for c_val in unique_C_isoflop:
        df_c = df_isoflop[df_isoflop['C'] == c_val].copy()
        df_c = df_c.sort_values('actual_N')

        # Plot for each indicator that is True (allows overlapping colors)
        for ind_col, color in ind_color_map:
            if df_c[ind_col].iloc[0]:
                ax_left.scatter(df_c['actual_N'], df_c['nn_loss'], color=color, marker='o', s=40, alpha=0.7, ec='none')
                ax_left.plot(df_c['actual_N'], df_c['nqs_loss'], color=color, linestyle='-', linewidth=3, alpha=0.7)

                df_c_chin = df_c.dropna(subset=['chin_loss'])
                if len(df_c_chin) > 0:
                    ax_left.plot(df_c_chin['actual_N'], df_c_chin['chin_loss'], color=color, linestyle='--', linewidth=3, alpha=0.7)

        # Add label using primary color (test > mid > train)
        if len(df_c) > 0:
            label_C = format_C_label(c_val)
            if df_c['test_ind'].iloc[0]:
                label_color = 'red'
            elif df_c['mid_ind'].iloc[0]:
                label_color = 'orange'
            else:
                label_color = 'blue'
            ax_left.text(df_c['actual_N'].values[-1] * 1.05, df_c['nqs_loss'].values[-1],
                         label_C, color=label_color, fontsize=9, fontweight='bold', va='center')

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

        # Plot for each indicator that is True (allows overlapping colors)
        for ind_col, color in ind_color_map:
            if df_c[ind_col].iloc[0]:
                ax_right.scatter(df_c['B'], df_c['nn_loss'], color=color, marker='o', s=40, alpha=0.7, ec='none')
                ax_right.plot(df_c['B'], df_c['nqs_loss'], color=color, linestyle='-', linewidth=3, alpha=0.7)

                df_c_chin = df_c.dropna(subset=['chin_loss'])
                if len(df_c_chin) > 0:
                    ax_right.plot(df_c_chin['B'], df_c_chin['chin_loss'], color=color, linestyle='--', linewidth=3, alpha=0.7)

        # Add label using primary color (test > mid > train)
        if len(df_c) > 0:
            label_C = format_C_label(c_val)
            if df_c['test_ind'].iloc[0]:
                label_color = 'red'
            elif df_c['mid_ind'].iloc[0]:
                label_color = 'orange'
            else:
                label_color = 'blue'
            ax_right.text(df_c['B'].values[-1] * 1.05, df_c['nqs_loss'].values[-1],
                          label_C, color=label_color, fontsize=9, fontweight='bold', va='center')

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
        Line2D([0], [0], color='gray', linestyle='-', linewidth=3, alpha=0.7, label='NQS'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=3, alpha=0.7, label='Chinchilla'),
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
    output_plot = BASE_FOLDER + f"/{output_prefix}_plot.png"
    #f"/mfs1/u/chuning/scale_new/{output_prefix}_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.savefig(output_plot.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"\nSaved plot to {output_plot}")

    # ============ TABLE MAKING ============
    # Use indicator columns to allow overlapping categories
    df_train = df[df['train_ind']]
    df_test_isoflop = df[df['test_ind'] & (df['type'] == 'isoflop')]
    df_test_isotoken = df[df['test_ind'] & (df['type'] == 'isotoken')]
    df_train_isoflop = df[df['train_ind'] & (df['type'] == 'isoflop')]

    if has_mid:
        df_mid_isoflop = df[df['mid_ind'] & (df['type'] == 'isoflop')]

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

    table_output = BASE_FOLDER + f"/{output_prefix}_summary.csv"
    #f"/mfs1/u/chuning/scale_new/{output_prefix}_summary.csv"
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

    table_output_mse = BASE_FOLDER + f"/{output_prefix}_summary_mse.csv"
    #f"/mfs1/u/chuning/scale_new/{output_prefix}_summary_mse.csv"
    df_table_mse.to_csv(table_output_mse, index=False)
    print(f"\nSaved MSE summary table to {table_output_mse}")

    return df


# ============ RUN ============
if __name__ == "__main__":
    df = process_owt_dataset(
        base_folder=BASE_FOLDER,
        split_config=SPLIT_CONFIG,
        output_prefix=OUTPUT_PREFIX
    )
