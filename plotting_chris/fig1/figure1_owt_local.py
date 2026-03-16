import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============ STYLE CONFIGURATION ============
# Modern, clean style for ML papers
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Colors
COLOR_TRAIN = 'blue'
COLOR_TEST = 'red'

# OWT data
# --------------------------------
# Base folder contains all the data files
# nqs fitted on all isoflops: "nqs_loss" column, from {base_folder}/5_resource_allocation/h_samples_with_nqs_loss.csv
# nn on all isoflops: "NN_loss" column, from {base_folder}/5_resource_allocation/to_get_nn_samples.csv
# nqs/nn fitted on all isotokens: "NN_loss" column and "nqs_loss" column, from {base_folder}/6_critical_batch_size/h_samples_with_nn_loss.csv
# chin fitted on all (isoflops and isotokens): "chin_loss" column, from chin_folder/4_loss_estimation/chinchilla_2/eval_df.csv


# ============ CONFIGURATION ============
# Base folder containing all data (NQS isoflop, isotoken, and chin)
BASE_FOLDER = "2026-01-28-15-45_openwebtext2_pythia_BFGS_x1024"
#"/mfs1/u/chuning/scale_new/outputs/runs/Fig4/2026-01-27-15-40_openwebtext2_pythia_BFGS or scipy_80PF"
#"/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-24-01-52_openwebtext2_pythia_adam_cosine_complete"

# Output prefix for files
OUTPUT_PREFIX = "figure1_lm1b"

# Train/test split threshold (in GFLOPs)
# - IsoFLOP train: C <= TRAIN_THRESHOLD
# - IsoFLOP test: C > 1000PF (fixed)
# - Isotoken train: C <= TRAIN_THRESHOLD
# - Isotoken test: C > TRAIN_THRESHOLD
TRAIN_THRESHOLD = 80000000   # in GFLOPs

# Internal split configuration (test_threshold for isoflop is fixed at 1000PF)
SPLIT_CONFIG = {
    'train_threshold': TRAIN_THRESHOLD,
    'test_threshold': 1000000000,     # 1000 PF in GFLOPs (fixed for isoflop test)
}


# ============ HELPER FUNCTIONS ============

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
    Process OWT dataset and generate 2x2 panel plot.

    Layout:
        Top row: IsoFLOP curves (Chinchilla left, NQS right)
        Bottom row: IsoToken curves (Chinchilla left, NQS right)
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

    # Assign split indicators based on config (no mid, just train and test)
    train_threshold = split_config['train_threshold']
    test_threshold = split_config['test_threshold']

    def assign_train_ind(row):
        """Train: C <= train_threshold (for both isoflop and isotoken)"""
        return row['C'] <= train_threshold

    def assign_test_ind(row):
        """Test: C > test_threshold for isoflop, C > train_threshold for isotoken"""
        if row['type'] == 'isoflop':
            return row['C'] > test_threshold
        else:  # isotoken
            return row['C'] > train_threshold

    df['train_ind'] = df.apply(assign_train_ind, axis=1)
    df['test_ind'] = df.apply(assign_test_ind, axis=1)

    print(f"Total rows: {len(df)}")
    print(f"Train rows (train_ind=True): {len(df[df['train_ind']])}")
    print(f"Test rows (test_ind=True): {len(df[df['test_ind']])}")
    print(f"Rows in both train and test: {len(df[df['train_ind'] & df['test_ind']])}")

    # ============ PLOTTING (2x2 grid) ============
    # Font sizes
    FONTSIZE_LABEL = 14
    FONTSIZE_TITLE = 14
    FONTSIZE_TICK = 12
    FONTSIZE_LEGEND = 12
    FONTSIZE_ANNOTATION = 11

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    ax_chin_isoflop = axes[0, 0]  # Top-left: Chinchilla IsoFLOP
    ax_nqs_isoflop = axes[0, 1]   # Top-right: NQS IsoFLOP
    ax_chin_isotoken = axes[1, 0] # Bottom-left: Chinchilla IsoToken
    ax_nqs_isotoken = axes[1, 1]  # Bottom-right: NQS IsoToken

    # Colors for train/test (no mid)
    ind_color_map = [('train_ind', COLOR_TRAIN), ('test_ind', COLOR_TEST)]

    # ========== TOP-LEFT: Chinchilla IsoFLOP ==========
    df_isoflop = df[df['type'] == 'isoflop'].copy()
    unique_C_isoflop = sorted(df_isoflop['C'].unique())

    # Find max C for train-only and test curves (for labeling)
    train_only_C_isoflop = [c for c in unique_C_isoflop if df_isoflop[df_isoflop['C'] == c]['train_ind'].iloc[0] and not df_isoflop[df_isoflop['C'] == c]['test_ind'].iloc[0]]
    test_C_isoflop = [c for c in unique_C_isoflop if df_isoflop[df_isoflop['C'] == c]['test_ind'].iloc[0]]
    max_train_C_isoflop = max(train_only_C_isoflop) if train_only_C_isoflop else None
    max_test_C_isoflop = max(test_C_isoflop) if test_C_isoflop else None

    for c_val in unique_C_isoflop:
        df_c = df_isoflop[df_isoflop['C'] == c_val].copy()
        df_c = df_c.sort_values('actual_N')

        for ind_col, color in ind_color_map:
            if df_c[ind_col].iloc[0]:
                ax_chin_isoflop.scatter(df_c['actual_N'], df_c['nn_loss'], color=color, marker='o', s=40, alpha=0.7, ec='none')
                df_c_chin = df_c.dropna(subset=['chin_loss'])
                if len(df_c_chin) > 0:
                    ax_chin_isoflop.plot(df_c_chin['actual_N'], df_c_chin['chin_loss'], color=color, linestyle='-', linewidth=3, alpha=0.7)

        # Label only highest train and highest test
        if len(df_c) > 0 and (c_val == max_train_C_isoflop or c_val == max_test_C_isoflop):
            label_C = format_C_label(c_val)
            label_color = COLOR_TEST if df_c['test_ind'].iloc[0] else COLOR_TRAIN
            df_c_chin = df_c.dropna(subset=['chin_loss'])
            if len(df_c_chin) > 0:
                ax_chin_isoflop.text(df_c_chin['actual_N'].values[-1] * 1.05, df_c_chin['chin_loss'].values[-1],
                                     label_C, color=label_color, fontsize=FONTSIZE_ANNOTATION, fontweight='bold', va='center')

    ax_chin_isoflop.set_xscale("log")
    ax_chin_isoflop.set_yscale("log")
    ax_chin_isoflop.set_xlabel("Model Size (N)", fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax_chin_isoflop.set_ylabel("Loss", fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax_chin_isoflop.set_title("(a) Chinchilla IsoFLOP\n       (Model Size)", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_chin_isoflop.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    x_min, x_max = ax_chin_isoflop.get_xlim()
    ax_chin_isoflop.set_xlim(x_min * 0.5, x_max * 2.5)
    ax_chin_isoflop.set_xticks([1e5, 1e7,1e9])
    ax_chin_isoflop.set_xticklabels(['$10^5$', '$10^7$',  '$10^9$'])
    ax_chin_isoflop.spines['top'].set_visible(False)
    ax_chin_isoflop.spines['right'].set_visible(False)
    ax_chin_isoflop.spines['left'].set_linewidth(2)
    ax_chin_isoflop.spines['bottom'].set_linewidth(2)

    # ========== TOP-RIGHT: NQS IsoFLOP ==========
    for c_val in unique_C_isoflop:
        df_c = df_isoflop[df_isoflop['C'] == c_val].copy()
        df_c = df_c.sort_values('actual_N')

        for ind_col, color in ind_color_map:
            if df_c[ind_col].iloc[0]:
                ax_nqs_isoflop.scatter(df_c['actual_N'], df_c['nn_loss'], color=color, marker='o', s=40, alpha=0.7, ec='none')
                ax_nqs_isoflop.plot(df_c['actual_N'], df_c['nqs_loss'], color=color, linestyle='-', linewidth=3, alpha=0.7)

        # Label only highest train and highest test
        if len(df_c) > 0 and (c_val == max_train_C_isoflop or c_val == max_test_C_isoflop):
            label_C = format_C_label(c_val)
            label_color = COLOR_TEST if df_c['test_ind'].iloc[0] else COLOR_TRAIN
            ax_nqs_isoflop.text(df_c['actual_N'].values[-1] * 1.05, df_c['nqs_loss'].values[-1],
                                label_C, color=label_color, fontsize=FONTSIZE_ANNOTATION, fontweight='bold', va='center')

    ax_nqs_isoflop.set_xscale("log")
    ax_nqs_isoflop.set_yscale("log")
    ax_nqs_isoflop.set_xlabel("Model Size (N)", fontsize=FONTSIZE_LABEL, fontweight='bold')
    # ax_nqs_isoflop.set_ylabel("Loss", fontweight='bold')  # Only label y-axis on left panels
    ax_nqs_isoflop.set_title("(b) NQS IsoFLOP\n               (Model Size)", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_nqs_isoflop.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    x_min, x_max = ax_nqs_isoflop.get_xlim()
    ax_nqs_isoflop.set_xlim(x_min * 0.5, x_max * 2.5)
    ax_nqs_isoflop.set_xticks([1e5, 1e7, 1e9])
    ax_nqs_isoflop.set_xticklabels(['$10^5$', '$10^7$', '$10^9$'])
    ax_nqs_isoflop.spines['top'].set_visible(False)
    ax_nqs_isoflop.spines['right'].set_visible(False)
    ax_nqs_isoflop.spines['left'].set_linewidth(2)
    ax_nqs_isoflop.spines['bottom'].set_linewidth(2)

    # Shared y-limits for top row (isoflop panels)
    y_min_chin, y_max_chin = ax_chin_isoflop.get_ylim()
    y_min_nqs, y_max_nqs = ax_nqs_isoflop.get_ylim()
    shared_y_min_isoflop = min(y_min_chin, y_min_nqs) * 0.98
    shared_y_max_isoflop = max(y_max_chin, y_max_nqs) * 1.05
    ax_chin_isoflop.set_ylim(shared_y_min_isoflop, shared_y_max_isoflop)
    ax_nqs_isoflop.set_ylim(shared_y_min_isoflop, shared_y_max_isoflop)

    # ========== BOTTOM-LEFT: Chinchilla IsoToken ==========
    df_isotoken = df[df['type'] == 'isotoken'].copy()
    unique_C_isotoken = sorted(df_isotoken['C'].unique())

    # Find max C for train-only and test curves (for labeling)
    train_only_C_isotoken = [c for c in unique_C_isotoken if df_isotoken[df_isotoken['C'] == c]['train_ind'].iloc[0] and not df_isotoken[df_isotoken['C'] == c]['test_ind'].iloc[0]]
    test_C_isotoken = [c for c in unique_C_isotoken if df_isotoken[df_isotoken['C'] == c]['test_ind'].iloc[0]]
    max_train_C_isotoken = max(train_only_C_isotoken) if train_only_C_isotoken else None
    max_test_C_isotoken = max(test_C_isotoken) if test_C_isotoken else None

    for c_val in unique_C_isotoken:
        df_c = df_isotoken[df_isotoken['C'] == c_val].copy()
        df_c = df_c.sort_values('B')

        for ind_col, color in ind_color_map:
            if df_c[ind_col].iloc[0]:
                ax_chin_isotoken.scatter(df_c['B'], df_c['nn_loss'], color=color, marker='o', s=40, alpha=0.7, ec='none')
                df_c_chin = df_c.dropna(subset=['chin_loss'])
                if len(df_c_chin) > 0:
                    ax_chin_isotoken.plot(df_c_chin['B'], df_c_chin['chin_loss'], color=color, linestyle='-', linewidth=3, alpha=0.7)

        # Label only highest train and highest test
        if len(df_c) > 0 and (c_val == max_train_C_isotoken or c_val == max_test_C_isotoken):
            label_C = format_C_label(c_val)
            label_color = COLOR_TEST if df_c['test_ind'].iloc[0] else COLOR_TRAIN
            df_c_chin = df_c.dropna(subset=['chin_loss'])
            if len(df_c_chin) > 0:
                ax_chin_isotoken.text(df_c_chin['B'].values[-1] * 1.05, df_c_chin['chin_loss'].values[-1],
                                      label_C, color=label_color, fontsize=FONTSIZE_ANNOTATION, fontweight='bold', va='center')

    ax_chin_isotoken.set_xscale("log")
    ax_chin_isotoken.set_yscale("log")
    ax_chin_isotoken.set_xlabel("Batch Size (B)", fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax_chin_isotoken.set_ylabel("Loss", fontsize=FONTSIZE_LABEL, fontweight='bold')
    ax_chin_isotoken.set_title("(c) Chinchilla IsoFLOP\n       (Batch Size)", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_chin_isotoken.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    x_min, x_max = ax_chin_isotoken.get_xlim()
    ax_chin_isotoken.set_xlim(x_min * 0.5, x_max * 2.5)
    ax_chin_isotoken.spines['top'].set_visible(False)
    ax_chin_isotoken.spines['right'].set_visible(False)
    ax_chin_isotoken.spines['left'].set_linewidth(2)
    ax_chin_isotoken.spines['bottom'].set_linewidth(2)

    # ========== BOTTOM-RIGHT: NQS IsoToken ==========
    for c_val in unique_C_isotoken:
        df_c = df_isotoken[df_isotoken['C'] == c_val].copy()
        df_c = df_c.sort_values('B')

        for ind_col, color in ind_color_map:
            if df_c[ind_col].iloc[0]:
                ax_nqs_isotoken.scatter(df_c['B'], df_c['nn_loss'], color=color, marker='o', s=40, alpha=0.7, ec='none')
                ax_nqs_isotoken.plot(df_c['B'], df_c['nqs_loss'], color=color, linestyle='-', linewidth=3, alpha=0.7)

        # Label only highest train and highest test
        if len(df_c) > 0 and (c_val == max_train_C_isotoken or c_val == max_test_C_isotoken):
            label_C = format_C_label(c_val)
            label_color = COLOR_TEST if df_c['test_ind'].iloc[0] else COLOR_TRAIN
            ax_nqs_isotoken.text(df_c['B'].values[-1] * 1.05, df_c['nqs_loss'].values[-1],
                                 label_C, color=label_color, fontsize=FONTSIZE_ANNOTATION, fontweight='bold', va='center')

    ax_nqs_isotoken.set_xscale("log")
    ax_nqs_isotoken.set_yscale("log")
    ax_nqs_isotoken.set_xlabel("Batch Size (B)", fontsize=FONTSIZE_LABEL, fontweight='bold')
    # ax_nqs_isotoken.set_ylabel("Loss", fontweight='bold')  # Only label y-axis on left panels
    ax_nqs_isotoken.set_title("(d) NQS IsoFLOP\n               (Batch Size)", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_nqs_isotoken.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    x_min, x_max = ax_nqs_isotoken.get_xlim()
    ax_nqs_isotoken.set_xlim(x_min * 0.5, x_max * 2.5)
    ax_nqs_isotoken.spines['top'].set_visible(False)
    ax_nqs_isotoken.spines['right'].set_visible(False)
    ax_nqs_isotoken.spines['left'].set_linewidth(2)
    ax_nqs_isotoken.spines['bottom'].set_linewidth(2)

    # Shared y-limits for bottom row (isotoken panels)
    y_min_chin, y_max_chin = ax_chin_isotoken.get_ylim()
    y_min_nqs, y_max_nqs = ax_nqs_isotoken.get_ylim()
    shared_y_min_isotoken = min(y_min_chin, y_min_nqs) * 0.98
    shared_y_max_isotoken = max(y_max_chin, y_max_nqs) * 1.05
    ax_chin_isotoken.set_ylim(shared_y_min_isotoken, shared_y_max_isotoken)
    ax_nqs_isotoken.set_ylim(shared_y_min_isotoken, shared_y_max_isotoken)

    # ========== LEGEND ==========
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label='LLM (Ground Truth)'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=3, alpha=0.7, label='Model Prediction'),
        Line2D([0], [0], color=COLOR_TRAIN, linestyle='-', linewidth=4, label='Train'),
        Line2D([0], [0], color=COLOR_TEST, linestyle='-', linewidth=4, label='Holdout'),
    ]

    fig.legend(handles=legend_elements, fontsize=FONTSIZE_LEGEND, loc='lower center',
               bbox_to_anchor=(0.5, -0.06), ncol=4, prop={'weight': 'bold'})

    plt.tight_layout(h_pad=3.0)
    plt.subplots_adjust(bottom=0.12)
    output_plot = BASE_FOLDER + f"/{output_prefix}_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.savefig(output_plot.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"\nSaved plot to {output_plot}")

    return df


# ============ RUN ============
if __name__ == "__main__":
    df = process_owt_dataset(
        base_folder=BASE_FOLDER,
        split_config=SPLIT_CONFIG,
        output_prefix=OUTPUT_PREFIX
    )
