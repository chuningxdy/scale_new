"""
Sweep over different train/test splits for Chinchilla Method 3 on Hoffman data.
- Test slices: always last two (1e21, 3e21)
- Train slices: 2,3,4,5,6,7,8,9 slices starting from smallest FLOPs
- When train and test overlap, use alpha=0.5 to show overlapping lines
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.optimize import minimize

# ============ HELPER FUNCTIONS FOR LOSS METRICS ============

def custom_huber_loss(y_true, y_pred, delta=1e-3):
    """Compute sum of Huber loss."""
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

def avg_mse(y_true, y_pred):
    """Compute average MSE on log scale."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    log_true = np.log(y_true)
    log_pred = np.log(y_pred)
    return np.mean((log_true - log_pred) ** 2)

# ============ CHINCHILLA MODEL FUNCTIONS ============

def log_sum_exp(a, b, e, alpha, beta, N, D):
    """Chinchilla loss function: L = A/N^alpha + B/D^beta + E"""
    return np.log(np.exp(a - alpha * np.log(N)) + np.exp(b - beta * np.log(D)) + np.exp(e))

def huber_loss_objective(params, N, D, losses):
    """Objective function for fitting Chinchilla model."""
    a, b, e, alpha, beta = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    diff = np.log(losses) - predictions
    delta = 1e-3
    cond = np.abs(diff) <= delta
    loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    return np.sum(loss)

def train_chin(N, D, losses):
    """Fit Chinchilla model using grid search + L-BFGS-B optimization."""
    from itertools import product

    alpha_vals = np.arange(0, 2.5, 0.5)
    beta_vals = np.arange(0, 2.5, 0.5)
    e_vals = np.arange(-1, 1.5, 0.5)
    a_vals = np.arange(0, 30, 5)
    b_vals = np.arange(0, 30, 5)

    best_loss = np.inf
    best_params = None

    for alpha, beta, e, a, b in product(alpha_vals, beta_vals, e_vals, a_vals, b_vals):
        init_params = [a, b, e, alpha, beta]
        result = minimize(huber_loss_objective, init_params,
                         args=(N, D, losses), method='L-BFGS-B')
        if result.success and result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x

    if best_params is not None:
        A = np.exp(best_params[0])
        B = np.exp(best_params[1])
        E = np.exp(best_params[2])
        alpha = best_params[3]
        beta = best_params[4]
        fitted_params = {'A': A, 'B': B, 'E': E, 'N_power': alpha, 'D_power': beta}
    else:
        raise ValueError("Optimization failed to converge.")

    return fitted_params

def predict_chin(N, D, fitted_params):
    """Predict loss using fitted Chinchilla model."""
    A = fitted_params['A']
    B = fitted_params['B']
    E = fitted_params['E']
    alpha = fitted_params['N_power']
    beta = fitted_params['D_power']
    preds = log_sum_exp(np.log(A), np.log(B), np.log(E), alpha, beta, N, D)
    return np.exp(preds)

def format_flop_label(fl):
    if fl >= 1e21:
        return f"  {fl/1e21:.0f}e21"
    elif fl >= 1e20:
        return f"  {fl/1e20:.0f}e20"
    elif fl >= 1e19:
        return f"  {fl/1e19:.1f}e19"
    else:
        return f"  {fl/1e18:.1f}e18"

# ============ MAIN FUNCTION FOR SINGLE CONFIGURATION ============

def run_single_config(train_slice_count, output_dir, data_on_isoflops, numeric_isoflops, color_palette="magma"):
    """
    Run Chinchilla fitting for a specific train slice count.

    Args:
        train_slice_count: Number of training slices (2-9)
        output_dir: Directory to save outputs
        data_on_isoflops: Dict mapping flop value to (N, D, losses)
        numeric_isoflops: List of numeric isoflop values
        color_palette: Color palette for plotting
    """
    os.makedirs(output_dir, exist_ok=True)

    # Test slices: always last two
    test_flops = [1e21, 3e21]

    # Train slices: first train_slice_count from smallest
    train_flops = numeric_isoflops[:train_slice_count]

    # Check for overlap
    overlap_flops = set(train_flops) & set(test_flops)
    has_overlap = len(overlap_flops) > 0

    print(f"\n{'='*60}")
    print(f"Train slice count: {train_slice_count}")
    print(f"Train FLOPs: {[f'{f:.1e}' for f in train_flops]}")
    print(f"Test FLOPs: {[f'{f:.1e}' for f in test_flops]}")
    print(f"Overlap: {[f'{f:.1e}' for f in overlap_flops] if has_overlap else 'None'}")
    print(f"{'='*60}")

    # Collect training data
    isoflop_N = []
    isoflop_D = []
    isoflop_losses = []
    for fl in train_flops:
        N_fl, D_fl, losses_fl = data_on_isoflops[fl]
        isoflop_N.extend(N_fl)
        isoflop_D.extend(D_fl)
        isoflop_losses.extend(losses_fl)

    N_train = np.array(isoflop_N)
    D_train = np.array(isoflop_D)
    losses_train = np.array(isoflop_losses)

    # Fit Chinchilla model
    print(f"Fitting Chinchilla model on {len(N_train)} training points...")
    fitted_params = train_chin(N_train, D_train, losses_train)
    print(f"Fitted parameters: {fitted_params}")

    # Compute train predictions and errors
    train_preds = predict_chin(N_train, D_train, fitted_params)
    train_mse = avg_mse(losses_train, train_preds)
    train_huber = avg_huber_loss(losses_train, train_preds)

    # Collect test data (excluding overlap for pure test metrics)
    test_only_flops = [f for f in test_flops if f not in train_flops]
    test_N = []
    test_D = []
    test_losses_arr = []
    for fl in test_only_flops:
        N_fl, D_fl, losses_fl = data_on_isoflops[fl]
        test_N.extend(N_fl)
        test_D.extend(D_fl)
        test_losses_arr.extend(losses_fl)

    test_N = np.array(test_N)
    test_D = np.array(test_D)
    test_losses_arr = np.array(test_losses_arr)

    # Compute test predictions and errors
    if len(test_N) > 0:
        test_preds = predict_chin(test_N, test_D, fitted_params)
        test_mse = avg_mse(test_losses_arr, test_preds)
        test_huber = avg_huber_loss(test_losses_arr, test_preds)
    else:
        test_mse = np.nan
        test_huber = np.nan

    # Also compute metrics on ALL test flops (including overlap)
    all_test_N = []
    all_test_D = []
    all_test_losses = []
    for fl in test_flops:
        N_fl, D_fl, losses_fl = data_on_isoflops[fl]
        all_test_N.extend(N_fl)
        all_test_D.extend(D_fl)
        all_test_losses.extend(losses_fl)

    all_test_N = np.array(all_test_N)
    all_test_D = np.array(all_test_D)
    all_test_losses = np.array(all_test_losses)

    if len(all_test_N) > 0:
        all_test_preds = predict_chin(all_test_N, all_test_D, fitted_params)
        all_test_mse = avg_mse(all_test_losses, all_test_preds)
        all_test_huber = avg_huber_loss(all_test_losses, all_test_preds)
    else:
        all_test_mse = np.nan
        all_test_huber = np.nan

    # ============ PLOTTING ============
    colors = sns.color_palette(color_palette, n_colors=len(numeric_isoflops))[::-1]

    COLOR_TRAIN = 'blue'
    COLOR_TEST = 'red'
    ALPHA_NORMAL = 0.8
    ALPHA_OVERLAP = 0.5

    fig, ax = plt.subplots(figsize=(6, 5))

    max_train_flop = max(train_flops)
    max_test_flop = max(test_flops)

    # Plot training curves (blue)
    for fl in train_flops:
        N_fl, D_fl, losses_fl = data_on_isoflops[fl]
        if len(N_fl) == 0:
            continue
        sorted_indices = np.argsort(N_fl)
        N_fl = N_fl[sorted_indices]
        D_fl = D_fl[sorted_indices]
        losses_fl = losses_fl[sorted_indices]
        preds_fl = predict_chin(N_fl, D_fl, fitted_params)

        # Use lower alpha if this flop is in overlap
        alpha = ALPHA_OVERLAP if fl in overlap_flops else ALPHA_NORMAL

        ax.scatter(N_fl, losses_fl, s=40, ec='none', facecolor=COLOR_TRAIN, alpha=alpha)
        ax.plot(N_fl, preds_fl, color=COLOR_TRAIN, linestyle='-', linewidth=2.5, alpha=alpha)

        # Label the largest training slice
        if fl == max_train_flop and fl not in test_flops:
            label_fl = format_flop_label(fl)
            ax.text(N_fl[-1] * 1.05, preds_fl[-1], label_fl,
                   color=COLOR_TRAIN, fontsize=10, fontweight='bold', va='center')

    # Plot test curves (red)
    for fl in test_flops:
        N_fl, D_fl, losses_fl = data_on_isoflops[fl]
        if len(N_fl) == 0:
            continue
        sorted_indices = np.argsort(N_fl)
        N_fl = N_fl[sorted_indices]
        D_fl = D_fl[sorted_indices]
        losses_fl = losses_fl[sorted_indices]
        preds_fl = predict_chin(N_fl, D_fl, fitted_params)

        # Use lower alpha if this flop is in overlap
        alpha = ALPHA_OVERLAP if fl in overlap_flops else ALPHA_NORMAL

        ax.scatter(N_fl, losses_fl, s=40, ec='none', facecolor=COLOR_TEST, alpha=alpha)
        ax.plot(N_fl, preds_fl, color=COLOR_TEST, linestyle='-', linewidth=2.5, alpha=alpha)

        # Label the largest test slice
        if fl == max_test_flop:
            label_fl = format_flop_label(fl)
            ax.text(N_fl[-1] * 1.05, preds_fl[-1], label_fl,
                   color=COLOR_TEST, fontsize=10, fontweight='bold', va='center')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Size (N)", fontweight='bold')
    ax.set_ylabel("Loss", fontweight='bold')

    overlap_str = " (with overlap)" if has_overlap else ""
    ax.set_title(f"Chinchilla Method 3\n{train_slice_count} Train Slices{overlap_str}", fontweight='bold')

    ax.set_xlim(3e7, 2e10)
    ax.set_ylim(2.0, 3.5)
    ax.set_yticks([2.0, 2.5, 3.0, 3.5])
    ax.set_yticklabels(['2.0', '2.5', '3.0', '3.5'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color=COLOR_TRAIN, linestyle='-', linewidth=2.5, markersize=8, alpha=ALPHA_NORMAL, label='Train'),
        Line2D([0], [0], marker='o', color=COLOR_TEST, linestyle='-', linewidth=2.5, markersize=8, alpha=ALPHA_NORMAL, label='Test'),
    ]
    if has_overlap:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='purple', linestyle='-', linewidth=2.5, markersize=8, alpha=ALPHA_OVERLAP, label='Overlap')
        )
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/plot.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # ============ SAVE RESULTS ============
    results = {
        'train_slice_count': train_slice_count,
        'train_flops': str([f'{f:.1e}' for f in train_flops]),
        'test_flops': str([f'{f:.1e}' for f in test_flops]),
        'overlap_flops': str([f'{f:.1e}' for f in overlap_flops]) if has_overlap else 'None',
        'train_n': len(N_train),
        'test_n_pure': len(test_N),
        'test_n_all': len(all_test_N),
        'train_mse': train_mse,
        'test_mse_pure': test_mse,
        'test_mse_all': all_test_mse,
        'train_huber': train_huber,
        'test_huber_pure': test_huber,
        'test_huber_all': all_test_huber,
        'fitted_A': fitted_params['A'],
        'fitted_B': fitted_params['B'],
        'fitted_E': fitted_params['E'],
        'fitted_N_power': fitted_params['N_power'],
        'fitted_D_power': fitted_params['D_power']
    }

    df_results = pd.DataFrame([results])
    df_results.to_csv(f"{output_dir}/results.csv", index=False)

    print(f"\nResults for {train_slice_count} train slices:")
    print(f"  Train MSE: {train_mse:.6f}, Huber: {train_huber:.6f}")
    print(f"  Test MSE (pure): {test_mse:.6f}, Huber: {test_huber:.6f}")
    print(f"  Test MSE (all): {all_test_mse:.6f}, Huber: {all_test_huber:.6f}")
    print(f"Saved to {output_dir}/")

    return results

# ============ MAIN SWEEP ============

def main():
    # Output directory
    base_output_dir = "/mfs1/u/chuning/scale_new/chin_rebuttal_hoffman"
    os.makedirs(base_output_dir, exist_ok=True)

    # Load Hoffman data
    # Need to run from the directory containing chin_data.csv
    os.chdir("/mfs1/u/chuning/scale")
    training_df = pd.read_csv('chin_data.csv')

    N = training_df['Model Size'].values
    D = training_df['Training Tokens'].values
    flops = training_df['Training FLOP'].values
    losses = training_df['loss'].values

    # Define isoflops
    isoflops = [5.5e18, 9.7e18, 2.8e19, 5.6e19, 8.7e19, 2.8e20, 5.6e20, 1e21, 3e21]
    numeric_isoflops = isoflops  # All are numeric

    # Find data points on each isoflops line
    data_on_isoflops = {}
    for fl in isoflops:
        indices = np.where((flops > fl * 0.9) & (flops < fl * 1.15))[0]
        data_on_isoflops[fl] = (N[indices], D[indices], losses[indices])
        print(f"IsoFLOP {fl:.1e}: {len(indices)} data points")

    # Run sweep over train slice counts 2-9
    all_results = []
    for train_slice_count in range(2, 10):
        output_dir = f"{base_output_dir}/train_{train_slice_count}_slices"
        results = run_single_config(
            train_slice_count=train_slice_count,
            output_dir=output_dir,
            data_on_isoflops=data_on_isoflops,
            numeric_isoflops=numeric_isoflops
        )
        all_results.append(results)

    # Save combined results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(f"{base_output_dir}/all_results.csv", index=False)

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE: All Configurations")
    print("="*80)
    summary_cols = ['train_slice_count', 'train_n', 'test_n_pure', 'train_mse', 'test_mse_pure', 'train_huber', 'test_huber_pure']
    print(df_all[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))
    print("="*80)

    # Create summary plot: MSE vs train slice count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(df_all['train_slice_count'], df_all['train_mse'], 'b-o', label='Train MSE', linewidth=2)
    ax1.plot(df_all['train_slice_count'], df_all['test_mse_pure'], 'r-o', label='Test MSE (pure)', linewidth=2)
    ax1.set_xlabel('Number of Train Slices', fontweight='bold')
    ax1.set_ylabel('MSE (log scale)', fontweight='bold')
    ax1.set_title('MSE vs Train Slice Count', fontweight='bold')
    ax1.legend()
    ax1.set_xticks(range(2, 10))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(df_all['train_slice_count'], df_all['train_huber'], 'b-o', label='Train Huber', linewidth=2)
    ax2.plot(df_all['train_slice_count'], df_all['test_huber_pure'], 'r-o', label='Test Huber (pure)', linewidth=2)
    ax2.set_xlabel('Number of Train Slices', fontweight='bold')
    ax2.set_ylabel('Huber Loss (log scale)', fontweight='bold')
    ax2.set_title('Huber Loss vs Train Slice Count', fontweight='bold')
    ax2.legend()
    ax2.set_xticks(range(2, 10))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{base_output_dir}/summary_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_output_dir}/summary_plot.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"\nSaved all results to {base_output_dir}/")
    print(f"- Subfolders: train_2_slices through train_9_slices")
    print(f"- Combined results: all_results.csv")
    print(f"- Summary plot: summary_plot.png/pdf")


if __name__ == "__main__":
    main()
