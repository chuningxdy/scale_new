import numpy as np
import pandas as pd
import json
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

    # Set up the grid for initial parameter values
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

# ============ MAIN FUNCTION ============

def chin_plot_with_table(color_palette="magma"):
    """
    Styled Chinchilla plot with 2 panels + summary table of MSE and Huber loss.
    """

    # Define isoflops lines
    isoflops = [5.5e18, 9.7e18, 2.8e19, 5.6e19, 8.7e19, 2.8e20, 5.6e20, 1e21, 3e21, "all_else_small", "all_else_large"]

    training_df = pd.read_csv('chin_data.csv')

    N = training_df['Model Size'].values
    D = training_df['Training Tokens'].values
    flops = training_df['Training FLOP'].values
    losses = training_df['loss'].values

    # Find data points on each isoflops line
    data_on_isoflops = {}
    all_isoflops_indices = []
    for fl in isoflops:
        if fl not in ["all_else_large", "all_else_small"]:
            indices = np.where((flops > fl * 0.9) & (flops < fl * 1.15))[0]
            data_on_isoflops[fl] = (N[indices], D[indices], losses[indices])
            all_isoflops_indices = np.concatenate((all_isoflops_indices, indices))

    all_else_indices = np.setdiff1d(np.arange(len(N)), all_isoflops_indices)
    all_else_indices_leq_3e21 = all_else_indices[flops[all_else_indices] <= 3e21]
    all_else_indices_gt_3e21 = all_else_indices[flops[all_else_indices] > 3e21]
    data_on_isoflops["all_else_large"] = (N[all_else_indices_gt_3e21], D[all_else_indices_gt_3e21], losses[all_else_indices_gt_3e21])
    data_on_isoflops["all_else_small"] = (N[all_else_indices_leq_3e21], D[all_else_indices_leq_3e21], losses[all_else_indices_leq_3e21])

    # Colors
    colors = sns.color_palette(color_palette, n_colors=len(isoflops))[::-1]

    def format_flop_label(fl):
        if fl >= 1e21:
            return f"  {fl/1e21:.0f}e21"
        elif fl >= 1e20:
            return f"  {fl/1e20:.0f}e20"
        elif fl >= 1e19:
            return f"  {fl/1e19:.1f}e19"
        else:
            return f"  {fl/1e18:.1f}e18"

    # ============ RESULTS STORAGE ============
    results = []

    # ============ PANEL CONFIGURATIONS ============
    panel_configs = [
        {
            'name': 'All IsoFLOPs',
            'train_flops': isoflops[:-2],  # max 3e21
            'test_flops': [],
            'title': "(a) Chinchilla Method 3\nFitted on All IsoFLOPs"
        },
        {
            'name': 'Train max 6e19',
            'train_flops': isoflops[:-7],  # max 6e19
            'test_flops': [1e21, 3e21],
            'title': "(b) Chinchilla Method 3\nTrain vs Heldout IsoFLOPs"
        }
    ]

    # Create figure with 2 panels
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))

    COLOR_TRAIN = 'blue'
    COLOR_TEST = 'red'

    for panel_idx, config in enumerate(panel_configs):
        ax = axs[panel_idx]
        train_flops = config['train_flops']
        test_flops = config['test_flops']

        # Collect training data
        isoflop_N = []
        isoflop_D = []
        isoflop_losses = []
        for fl in train_flops:
            if fl not in ["all_else_large", "all_else_small"]:
                N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                isoflop_N.extend(N_fl)
                isoflop_D.extend(D_fl)
                isoflop_losses.extend(losses_fl)

        N_train = np.array(isoflop_N)
        D_train = np.array(isoflop_D)
        losses_train = np.array(isoflop_losses)

        # Fit Chinchilla model
        print(f"\nFitting {config['name']}...")
        fitted_params = train_chin(N_train, D_train, losses_train)
        print(f"Fitted parameters: {fitted_params}")

        # Compute train predictions and errors
        train_preds = predict_chin(N_train, D_train, fitted_params)
        train_mse = avg_mse(losses_train, train_preds)
        train_huber = avg_huber_loss(losses_train, train_preds)

        # Collect test data
        test_N = []
        test_D = []
        test_losses_arr = []
        for fl in test_flops:
            if fl not in ["all_else_large", "all_else_small"]:
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

        # Store results
        results.append({
            'Panel': config['name'],
            'Train_N': len(N_train),
            'Test_N': len(test_N),
            'Train_MSE': train_mse,
            'Test_MSE': test_mse,
            'Train_Huber': train_huber,
            'Test_Huber': test_huber
        })

        # Get max flops for labeling
        numeric_train_flops = [fl for fl in train_flops if fl not in ["all_else_large", "all_else_small"]]
        max_train_flop = max(numeric_train_flops) if numeric_train_flops else None

        # ============ PLOTTING ============
        if panel_idx == 0:
            # Panel (a): All training, use color palette
            for i, fl in enumerate(isoflops):
                if fl not in ["all_else_large", "all_else_small"]:
                    N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                    if len(N_fl) == 0:
                        continue
                    sorted_indices = np.argsort(N_fl)
                    N_fl = N_fl[sorted_indices]
                    D_fl = D_fl[sorted_indices]
                    losses_fl = losses_fl[sorted_indices]
                    preds_fl = predict_chin(N_fl, D_fl, fitted_params)

                    ax.scatter(N_fl, losses_fl, s=40, ec='none', facecolor=colors[i], alpha=0.8)
                    ax.plot(N_fl, preds_fl, color=colors[i], linestyle='-', linewidth=2.5, alpha=0.7)

                    if fl == max_train_flop:
                        label_fl = format_flop_label(fl)
                        ax.text(N_fl[-1] * 1.05, preds_fl[-1], label_fl,
                               color=colors[i], fontsize=10, fontweight='bold', va='center')
        else:
            # Panel (b): Train (blue) and Test (red)
            # Plot training curves
            for fl in train_flops:
                if fl not in ["all_else_large", "all_else_small"]:
                    N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                    if len(N_fl) == 0:
                        continue
                    sorted_indices = np.argsort(N_fl)
                    N_fl = N_fl[sorted_indices]
                    D_fl = D_fl[sorted_indices]
                    losses_fl = losses_fl[sorted_indices]
                    preds_fl = predict_chin(N_fl, D_fl, fitted_params)

                    ax.scatter(N_fl, losses_fl, s=40, ec='none', facecolor=COLOR_TRAIN, alpha=0.8)
                    ax.plot(N_fl, preds_fl, color=COLOR_TRAIN, linestyle='-', linewidth=2.5, alpha=0.7)

                    if fl == max_train_flop:
                        label_fl = format_flop_label(fl)
                        ax.text(N_fl[-1] * 1.05, preds_fl[-1], label_fl,
                               color=COLOR_TRAIN, fontsize=10, fontweight='bold', va='center')

            # Plot test curves
            max_test_flop = max(test_flops) if test_flops else None
            for fl in test_flops:
                N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                if len(N_fl) == 0:
                    continue
                sorted_indices = np.argsort(N_fl)
                N_fl = N_fl[sorted_indices]
                D_fl = D_fl[sorted_indices]
                losses_fl = losses_fl[sorted_indices]
                preds_fl = predict_chin(N_fl, D_fl, fitted_params)

                ax.scatter(N_fl, losses_fl, s=40, ec='none', facecolor=COLOR_TEST, alpha=0.8)
                ax.plot(N_fl, preds_fl, color=COLOR_TEST, linestyle='-', linewidth=2.5, alpha=0.7)

                if fl == max_test_flop:
                    label_fl = format_flop_label(fl)
                    ax.text(N_fl[-1] * 1.05, preds_fl[-1], label_fl,
                           color=COLOR_TEST, fontsize=10, fontweight='bold', va='center')

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model Size (N)", fontweight='bold')
        ax.set_ylabel("Loss", fontweight='bold')
        ax.set_title(config['title'], fontweight='bold')

        ax.set_xlim(3e7, 2e10)
        ax.set_ylim(2.1, 3.3)
        ax.set_yticks([2.5, 3.0])
        ax.set_yticklabels(['2.5', '3.0'])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='blue', linestyle='-', linewidth=2.5, markersize=8, alpha=0.7, label='Train'),
        Line2D([0], [0], marker='o', color='red', linestyle='-', linewidth=2.5, markersize=8, alpha=0.7, label='Heldout'),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=2, prop={'weight': 'bold'})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("chin_rebuttal_with_table_plot.png", dpi=300)
    plt.savefig("chin_rebuttal_with_table_plot.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print("\nSaved plot to chin_rebuttal_with_table_plot.png/pdf")

    # ============ CREATE SUMMARY TABLE ============
    df_results = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY TABLE: Error Metrics for Chinchilla Method 3")
    print("="*80)
    print("\nMSE (Mean Squared Error on log scale):")
    print("-"*60)
    for _, row in df_results.iterrows():
        print(f"{row['Panel']}:")
        print(f"  Train (n={row['Train_N']}): MSE = {row['Train_MSE']:.6f}")
        if not np.isnan(row['Test_MSE']):
            print(f"  Test  (n={row['Test_N']}): MSE = {row['Test_MSE']:.6f}")

    print("\nHuber Loss (delta=1e-3, on log scale):")
    print("-"*60)
    for _, row in df_results.iterrows():
        print(f"{row['Panel']}:")
        print(f"  Train (n={row['Train_N']}): Huber = {row['Train_Huber']:.6f}")
        if not np.isnan(row['Test_Huber']):
            print(f"  Test  (n={row['Test_N']}): Huber = {row['Test_Huber']:.6f}")

    print("="*80)

    # Save to CSV
    df_results.to_csv("chin_rebuttal_with_table_summary.csv", index=False)
    print("\nSaved summary table to chin_rebuttal_with_table_summary.csv")

    # Create a formatted table for display/paper
    table_formatted = pd.DataFrame({
        'Configuration': ['All IsoFLOPs (Train)', 'Train max 6e19 (Train)', 'Train max 6e19 (Test)'],
        'N': [df_results.iloc[0]['Train_N'], df_results.iloc[1]['Train_N'], df_results.iloc[1]['Test_N']],
        'MSE': [df_results.iloc[0]['Train_MSE'], df_results.iloc[1]['Train_MSE'], df_results.iloc[1]['Test_MSE']],
        'Huber': [df_results.iloc[0]['Train_Huber'], df_results.iloc[1]['Train_Huber'], df_results.iloc[1]['Test_Huber']]
    })

    print("\n" + "="*80)
    print("FORMATTED TABLE:")
    print("="*80)
    print(table_formatted.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(int(x))))
    print("="*80)

    table_formatted.to_csv("chin_rebuttal_with_table_formatted.csv", index=False)
    print("\nSaved formatted table to chin_rebuttal_with_table_formatted.csv")

    return df_results


if __name__ == "__main__":
    chin_plot_with_table(color_palette="magma")
