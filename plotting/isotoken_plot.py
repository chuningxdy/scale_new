"""
Isotoken plotting function styled similarly to isoflop plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def isotoken_plot(csv_path, seq_len=128, color_palette="magma", output_file=None):
    """
    Create a styled isotoken plot from a CSV file containing loss data.

    The plot shows Loss vs N (model size) for different D (token) values,
    styled similarly to the isoflop plots in plotting_functions.py.

    Args:
        csv_path (str): Path to the CSV file with columns including
                        N, B, K, C, nqs_loss, actual_N, NN_loss, D
        seq_len (int): Sequence length used in the experiments (default: 128)
        color_palette (str): Seaborn color palette name (default: "magma")
        output_file (str): Output file path. If None, saves to "isotoken_plot.png"

    Returns:
        None
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Ensure we have required columns
    required_cols = ['N', 'B', 'K', 'NN_loss']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute D (total tokens) if not present
    if 'D' not in df.columns:
        df['D'] = seq_len * df['B'] * df['K']

    # Use actual_N if available, otherwise use N
    if 'actual_N' in df.columns:
        df['actual_N'] = df['actual_N'].fillna(df['N'])
    else:
        df['actual_N'] = df['N']

    # Convert to millions for display
    df['N_millions'] = df['actual_N'] / 1e6
    df['D_millions'] = df['D'] / 1e6

    # Get unique D values and sort them
    unique_D = sorted(df['D'].unique())

    # Use reversed color palette (style from plotting_functions.py)
    colors = sns.color_palette(color_palette, n_colors=len(unique_D))[::-1]
    D_to_color = {d: colors[i] for i, d in enumerate(unique_D)}

    # Helper to format D labels
    def format_D_label(d):
        if d >= 1e9:
            return f"  {d/1e9:.1f}B"
        elif d >= 1e6:
            return f"  {d/1e6:.0f}M"
        elif d >= 1e3:
            return f"  {d/1e3:.0f}K"
        else:
            return f"  {d:.0f}"

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each isotoken curve
    for d in unique_D:
        df_d = df[df['D'] == d].copy()

        # Sort by N for proper line plotting
        df_d = df_d.sort_values('actual_N')

        # Remove duplicates by taking mean of NN_loss for same N values
        df_d = df_d.groupby('actual_N').agg({
            'NN_loss': 'mean',
            'N_millions': 'first',
            'D_millions': 'first'
        }).reset_index()
        df_d = df_d.sort_values('actual_N')

        color = D_to_color[d]

        # Scatter for actual LLM loss
        ax.scatter(df_d['actual_N'], df_d['NN_loss'],
                   color=color, marker='o', s=40, alpha=0.8, ec='none')

        # Line connecting points
        ax.plot(df_d['actual_N'], df_d['NN_loss'],
                color=color, linestyle='-', linewidth=2.5, alpha=0.7)

        # Add label at the end of curve with D value
        if len(df_d) > 0:
            label_D = format_D_label(d)
            ax.text(df_d['actual_N'].values[-1] * 1.05, df_d['NN_loss'].values[-1],
                    label_D, color=color, fontsize=10, fontweight='bold', va='center')

    # Style the axes (from plotting_functions.py)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (Parameters)", fontweight='bold')
    ax.set_ylabel("Loss", fontweight='bold')
    ax.set_title("Isotoken Curves", fontweight='bold')

    # Expand axis range for more blank space
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min * 0.5, x_max * 2)
    ax.set_ylim(y_min * 0.95, y_max * 1.1)

    # Remove top/right spines, thicken left/bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='-',
               markersize=8, linewidth=2.5, alpha=0.7, label='LLM Loss'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right',
              prop={'weight': 'bold'})

    # Save figure
    plt.tight_layout()
    if output_file is None:
        output_file = "isotoken_plot.png"
    plt.savefig(output_file, dpi=300)
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"Saved isotoken plot to {output_file}")
    return None


def isotoken_plot_with_nqs(csv_path, seq_len=128, color_palette="magma", output_file=None):
    """
    Create a styled isotoken plot with both NN_loss and nqs_loss.

    Similar to isotoken_plot but shows both actual (NN) and predicted (NQS) losses.

    Args:
        csv_path (str): Path to the CSV file
        seq_len (int): Sequence length (default: 128)
        color_palette (str): Seaborn color palette name (default: "magma")
        output_file (str): Output file path. If None, saves to "isotoken_plot_with_nqs.png"

    Returns:
        None
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Ensure we have required columns
    required_cols = ['N', 'B', 'K', 'NN_loss', 'nqs_loss']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute D (total tokens) if not present
    if 'D' not in df.columns:
        df['D'] = seq_len * df['B'] * df['K']

    # Use actual_N if available
    if 'actual_N' in df.columns:
        df['actual_N'] = df['actual_N'].fillna(df['N'])
    else:
        df['actual_N'] = df['N']

    # Get unique D values and sort them
    unique_D = sorted(df['D'].unique())

    # Use reversed color palette
    colors = sns.color_palette(color_palette, n_colors=len(unique_D))[::-1]
    D_to_color = {d: colors[i] for i, d in enumerate(unique_D)}

    # Helper to format D labels
    def format_D_label(d):
        if d >= 1e9:
            return f"  {d/1e9:.1f}B"
        elif d >= 1e6:
            return f"  {d/1e6:.0f}M"
        elif d >= 1e3:
            return f"  {d/1e3:.0f}K"
        else:
            return f"  {d:.0f}"

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each isotoken curve
    for d in unique_D:
        df_d = df[df['D'] == d].copy()
        df_d = df_d.sort_values('actual_N')

        # Remove duplicates
        df_d = df_d.groupby('actual_N').agg({
            'NN_loss': 'mean',
            'nqs_loss': 'mean',
        }).reset_index()
        df_d = df_d.sort_values('actual_N')

        color = D_to_color[d]

        # Scatter for actual LLM loss
        ax.scatter(df_d['actual_N'], df_d['NN_loss'],
                   color=color, marker='o', s=40, alpha=0.8, ec='none')

        # Line for NQS predictions
        ax.plot(df_d['actual_N'], df_d['nqs_loss'],
                color=color, linestyle='-', linewidth=2.5, alpha=0.7)

        # Add label
        if len(df_d) > 0:
            label_D = format_D_label(d)
            ax.text(df_d['actual_N'].values[-1] * 1.05, df_d['nqs_loss'].values[-1],
                    label_D, color=color, fontsize=10, fontweight='bold', va='center')

    # Style the axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (Parameters)", fontweight='bold')
    ax.set_ylabel("Loss", fontweight='bold')
    ax.set_title("Isotoken Curves (NQS)", fontweight='bold')

    # Expand axis range
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min * 0.5, x_max * 2)
    ax.set_ylim(y_min * 0.95, y_max * 1.1)

    # Remove top/right spines, thicken left/bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=8, label='LLM (Groundtruth)'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5,
               alpha=0.7, label='NQS'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right',
              prop={'weight': 'bold'})

    # Save figure
    plt.tight_layout()
    if output_file is None:
        output_file = "isotoken_plot_with_nqs.png"
    plt.savefig(output_file, dpi=300)
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"Saved isotoken plot to {output_file}")
    return None


def isotoken_plot_batch_size(csv_path, seq_len=128, color_palette="magma", output_file=None):
    """
    Create a styled isotoken plot with Loss vs Batch Size (B).

    Shows how loss varies with batch size for different compute budgets (C).
    Uses NN_loss (dots) for LLM groundtruth and nqs_loss (line) for NQS predictions.
    Groups C values that are within +-30% of each other.

    Args:
        csv_path (str): Path to the CSV file
        seq_len (int): Sequence length (default: 128)
        color_palette (str): Seaborn color palette name (default: "magma")
        output_file (str): Output file path. If None, saves to "isotoken_batch_size_plot.png"

    Returns:
        None
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Filter data: only keep rows with NN_loss < 4.0 and B < 8000
    df = df[(df['NN_loss'] < 6.0) & (df['B'] < 8e3)]

    # Get unique C values and sort them
    # C is in units of 1e9 FLOPs (GigaFLOPs)
    unique_C = sorted(df['C'].unique())

    # Group C values that are within +-30% of each other
    def group_similar_Cs(C_list, tolerance=0.3):
        """Group C values that are within tolerance (e.g., 0.3 = 70%-130%) of each other."""
        if not C_list:
            return []
        groups = []
        current_group = [C_list[0]]
        for C in C_list[1:]:
            # Check if C is within tolerance of the first element in current group
            if C <= current_group[0] * (1 + tolerance) and C >= current_group[0] * (1 - tolerance):
                current_group.append(C)
            else:
                groups.append(current_group)
                current_group = [C]
        groups.append(current_group)
        return groups

    C_groups = group_similar_Cs(unique_C, tolerance=0.3)

    # Create a mapping from original C to group representative (mean of group)
    C_to_group = {}
    group_representatives = []
    for group in C_groups:
        rep = np.mean(group)
        group_representatives.append(rep)
        for C in group:
            C_to_group[C] = rep

    # Add grouped C column to dataframe
    df['C_group'] = df['C'].map(C_to_group)

    # Use reversed color palette
    colors = sns.color_palette(color_palette, n_colors=len(group_representatives))[::-1]
    group_to_color = {rep: colors[i] for i, rep in enumerate(group_representatives)}

    # Helper to format C labels as PetaFLOPs
    # C is in 1e9 FLOPs, so divide by 1e6 to get PF (1 PF = 1e15 FLOPs)
    def format_C_label(c_gflops):
        c_pf = c_gflops / 1e6  # Convert from GFLOPs to PF
        if c_pf >= 1:
            return f"  {int(round(c_pf))} PF"
        else:
            return f"  {c_pf:.2f} PF"

    # Create figure with smaller size (comparable to isoflop panels)
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot each isotoken curve (grouped by C_group)
    for group_rep in group_representatives:
        df_group = df[df['C_group'] == group_rep].copy()
        df_group = df_group.sort_values('B')

        # Remove duplicates by taking mean
        df_group = df_group.groupby('B').agg({
            'NN_loss': 'mean',
            'nqs_loss': 'mean' if 'nqs_loss' in df.columns else 'first',
        }).reset_index()
        df_group = df_group.sort_values('B')

        color = group_to_color[group_rep]

        # Scatter for actual LLM loss (NN_loss) - dots
        ax.scatter(df_group['B'], df_group['NN_loss'],
                   color=color, marker='o', s=40, alpha=0.8, ec='none')

        # Line for NQS predictions (nqs_loss)
        ax.plot(df_group['B'], df_group['nqs_loss'],
                color=color, linestyle='-', linewidth=2.5, alpha=0.7)

        # Add label at the end of NQS curve (in PetaFLOPs)
        if len(df_group) > 0:
            label_C = format_C_label(group_rep)
            ax.text(df_group['B'].values[-1] * 1.05, df_group['nqs_loss'].values[-1],
                    label_C, color=color, fontsize=10, fontweight='bold', va='center')

    # Style the axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch Size (B)", fontweight='bold')
    ax.set_ylabel("Loss", fontweight='bold')
    ax.set_title("Isotoken Curves", fontweight='bold')

    # Expand axis range
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min * 0.5, x_max * 2)
    ax.set_ylim(y_min * 0.95, y_max * 1.1)

    # Remove top/right spines, thicken left/bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=8, label='LLM (Groundtruth)'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5,
               alpha=0.7, label='NQS'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left',
              prop={'weight': 'bold'})

    # Save figure
    plt.tight_layout()
    if output_file is None:
        output_file = "isotoken_batch_size_plot.png"
    plt.savefig(output_file, dpi=300)
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"Saved isotoken batch size plot to {output_file}")
    return None


def isotoken_plot_batch_size_two_panel(csv_path_left, csv_path_right, seq_len=128,
                                        color_palette="magma", output_file=None,
                                        title_left=r"(a) Vanilla NQS",
                                        title_right=r"(b) NQS with lr $\propto 1/\|w\|^2$"):
    """
    Create a two-panel isotoken plot comparing two datasets.

    Left panel uses csv_path_left, right panel uses csv_path_right.
    Both panels show Loss vs Batch Size (B) with NN_loss (dots) and nqs_loss (line).

    Args:
        csv_path_left (str): Path to CSV file for left panel
        csv_path_right (str): Path to CSV file for right panel
        seq_len (int): Sequence length (default: 128)
        color_palette (str): Seaborn color palette name (default: "magma")
        output_file (str): Output file path. If None, saves to "isotoken_two_panel.png"
        title_left (str): Title for left panel
        title_right (str): Title for right panel

    Returns:
        None
    """

    def process_data(csv_path):
        """Load and process data for one panel."""
        df = pd.read_csv(csv_path)

        # Filter data: only keep rows with NN_loss < 6.0 and B < 8000
        df = df[(df['NN_loss'] < 6.0) & (df['B'] < 8e3)]

        # Get unique C values and sort them
        unique_C = sorted(df['C'].unique())

        # Group C values that are within +-30% of each other
        def group_similar_Cs(C_list, tolerance=0.3):
            if not C_list:
                return []
            groups = []
            current_group = [C_list[0]]
            for C in C_list[1:]:
                if C <= current_group[0] * (1 + tolerance) and C >= current_group[0] * (1 - tolerance):
                    current_group.append(C)
                else:
                    groups.append(current_group)
                    current_group = [C]
            groups.append(current_group)
            return groups

        C_groups = group_similar_Cs(unique_C, tolerance=0.3)

        # Create mapping from original C to group representative
        C_to_group = {}
        group_representatives = []
        for group in C_groups:
            rep = np.mean(group)
            group_representatives.append(rep)
            for C in group:
                C_to_group[C] = rep

        df['C_group'] = df['C'].map(C_to_group)

        return df, group_representatives

    def format_C_label(c_gflops):
        """Format C label as PetaFLOPs."""
        c_pf = c_gflops / 1e6  # Convert from GFLOPs to PF
        if c_pf >= 1:
            return f"  {int(round(c_pf))} PF"
        else:
            return f"  {c_pf:.2f} PF"

    def plot_panel(ax, df, group_representatives, colors, title):
        """Plot one panel."""
        group_to_color = {rep: colors[i] for i, rep in enumerate(group_representatives)}

        for group_rep in group_representatives:
            df_group = df[df['C_group'] == group_rep].copy()
            df_group = df_group.sort_values('B')

            # Remove duplicates by taking mean
            df_group = df_group.groupby('B').agg({
                'NN_loss': 'mean',
                'nqs_loss': 'mean' if 'nqs_loss' in df.columns else 'first',
            }).reset_index()
            df_group = df_group.sort_values('B')

            color = group_to_color[group_rep]

            # Scatter for actual LLM loss (NN_loss) - dots
            ax.scatter(df_group['B'], df_group['NN_loss'],
                       color=color, marker='o', s=40, alpha=0.8, ec='none')

            # Line for NQS predictions (nqs_loss)
            ax.plot(df_group['B'], df_group['nqs_loss'],
                    color=color, linestyle='-', linewidth=2.5, alpha=0.7)

            # Add label at the end of NQS curve (in PetaFLOPs)
            if len(df_group) > 0:
                label_C = format_C_label(group_rep)
                ax.text(df_group['B'].values[-1] * 1.05, df_group['nqs_loss'].values[-1],
                        label_C, color=color, fontsize=10, fontweight='bold', va='center')

        # Style the axes
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch Size (B)", fontweight='bold')
        ax.set_ylabel("Loss", fontweight='bold')
        ax.set_title(title, fontweight='bold')

        # Expand axis range
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(x_min * 0.5, x_max * 2)
        ax.set_ylim(y_min * 0.95, y_max * 1.1)

        # Remove top/right spines, thicken left/bottom
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

    # Process both datasets
    df_left, groups_left = process_data(csv_path_left)
    df_right, groups_right = process_data(csv_path_right)

    # Use colors based on max number of groups
    max_groups = max(len(groups_left), len(groups_right))
    colors = sns.color_palette(color_palette, n_colors=max_groups)[::-1]

    # Create two-panel figure
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6, 3))

    # Plot left panel
    colors_left = sns.color_palette(color_palette, n_colors=len(groups_left))[::-1]
    plot_panel(ax_left, df_left, groups_left, colors_left, title_left)

    # Plot right panel
    colors_right = sns.color_palette(color_palette, n_colors=len(groups_right))[::-1]
    plot_panel(ax_right, df_right, groups_right, colors_right, title_right)

    # Add shared legend at bottom (moved further down)
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=8, label='LLM (Groundtruth)'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5,
               alpha=0.7, label='NQS'),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
               ncol=2, prop={'weight': 'bold'}, bbox_to_anchor=(0.5, -0.08))

    # Save figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # Make room for legend
    if output_file is None:
        output_file = "isotoken_two_panel.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"Saved two-panel isotoken plot to {output_file}")
    return None


if __name__ == "__main__":
    # Example usage
    csv_path = "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-22-23-39_openwebtext2_pythia_adam_cosine_train_exhibits/6_critical_batch_size/h_samples_with_nn_loss.csv"
    csv_path_no_lra = "/mfs1/u/chuning/scale_new/outputs/runs/Paper/2026-01-22-23-40_openwebtext2_pythia_adam_cosine_train_ablation_no_lra/6_critical_batch_size/h_samples_with_nn_loss.csv"

    # Create the basic isotoken plot (Loss vs Batch Size since D is constant per curve)
    isotoken_plot_batch_size(csv_path, seq_len=128, color_palette="magma",
                             output_file="/mfs1/u/chuning/scale_new/plotting/isotoken_batch_size.png")

    # Create isotoken plot with NQS predictions
    isotoken_plot_with_nqs(csv_path, seq_len=128, color_palette="magma",
                           output_file="/mfs1/u/chuning/scale_new/plotting/isotoken_with_nqs.png")

    # Create two-panel isotoken plot comparing with and without lr scaling
    isotoken_plot_batch_size_two_panel(
        csv_path_left=csv_path_no_lra,
        csv_path_right=csv_path,
        seq_len=128,
        color_palette="magma",
        output_file="/mfs1/u/chuning/scale_new/plotting/isotoken_two_panel.png"
    )
