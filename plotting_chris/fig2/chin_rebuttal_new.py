
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
if not (jnp.array(1.).dtype is jnp.dtype("float64")):
    raise RuntimeError("jax.config.jax_enable_x64 must be set for numerical stability. "
                      "Please set this environment variable before using NQS.")
from collections import namedtuple
from functools import partial
import os
import time
import datetime
import json
from jax import lax

import numpy as np
from pyDOE import lhs  # for latin hypercube sampling



import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import erf
import scipy.stats as stats
from scipy.stats import norm

#from a_scale.nqs.nqs_sgd import _fit_nqs, latin_hypercube_initializations, _risk_no_sch
import seaborn as sns


def log_sum_exp(a, b, e, alpha, beta, N, D):
    return np.log(np.exp(a - alpha * np.log(N)) + np.exp(b - beta * np.log(D)) + np.exp(e))
#
# Define training data
# use some random data for now
#np.random.seed(0)
##N = np.random.randint(100, 1000, 100)
#D = np.random.randint(100, 1000, 100)
#losses = np.exp(log_sum_exp(*true_params, N, D) + np.random.normal(0, 0.1, 100))
#training_df = pd.DataFrame({'Model Size': N, 'Training Tokens': D, 'loss': losses})

#np.random.seed(42)
#random_indices = [np.random.choice(indices, size=len(indices), replace=True) for _ in range(bootstraps)]

# Define the log-sum-exp function
def log_sum_exp(a, b, e, alpha, beta, N, D):
    return np.log(np.exp(a - alpha * np.log(N)) + np.exp(b - beta * np.log(D)) + np.exp(e))

# Define the Huber loss function
def custom_huber_loss(y_true, y_pred, delta=1e-3):
    # Calculate the difference
    diff = y_true - y_pred
    # Calculate the condition for Huber loss
    cond = np.abs(diff) <= delta
    # Apply Huber loss formula
    loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    return np.sum(loss)

def huber_normalizing_factor(delta=1e-3):
    return np.sqrt(2*np.pi) * (1 - 2*norm.sf(delta)) + 2 * np.exp(-0.5*delta**2)/delta

def huber_logpdf(x, delta=1e-3, loc=0, scale=1):
    x = (x-loc)/scale

    cond = np.abs(x) <= delta
    loss = np.where(cond, 0.5 * x**2, delta * (np.abs(x) - 0.5 * delta))
    return -loss - np.log(huber_normalizing_factor(delta=delta)) - np.log(scale)

def huber_pdf(x, delta=1e-3, loc=0, scale=1):
    return np.exp(huber_logpdf(x, delta=delta, loc=loc, scale=scale))

# Define the objective function to be minimized
def objective(params, N, D, losses):
    a, b, e, alpha, beta, sigma = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return -np.sum(huber_logpdf(np.log(losses), loc=predictions, scale=np.exp(sigma), delta=1e-3))
    # return custom_huber_loss(np.log(losses), predictions, delta=1e-3)

def scale_objective(sigma, params, N, D, losses):
    a, b, e, alpha, beta = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return -np.sum(huber_logpdf(np.log(losses), loc=predictions, scale=np.exp(sigma), delta=1e-3))
    # return custom_huber_loss(np.log(losses), predictions, delta=1e-3)

def constant_term_objective(params, a, b, alpha, beta, N, D, losses):
    e, sigma = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return -np.sum(huber_logpdf(np.log(losses), loc=predictions, scale=np.exp(sigma), delta=1e-3))

def huber_loss_objective(params, N, D, losses):
    a, b, e, alpha, beta = params
    predictions = log_sum_exp(a, b, e, alpha, beta, N, D)
    return custom_huber_loss(np.log(losses), predictions, delta=1e-3)

# Define the parameter untransform
def untransform_params(param_array):
    if len(np.shape(param_array)) == 2:
      return np.hstack((np.exp(param_array[:, :3]), param_array[:, 3:]))
    else:
      return np.hstack((np.exp(param_array[:3]), param_array[3:]))

# Define the Huber loss function on residuals
def huber_loss(residuals, delta=1e-3):
    # Calculate the difference
    diff = residuals
    # Calculate the condition for Huber loss
    cond = np.abs(diff) <= delta
    # Apply Huber loss formula
    loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    return loss



def train_chin(N, D, losses): #, nr_of_models_excluded=1):

    #nr_of_models_excluded = 1
    #training_df = pd.read_csv('chin_data.csv')

   # N = training_df['Model Size'].values
   # D = training_df['Training Tokens'].values
   # losses = training_df['loss'].values
    #bootstraps = 4000

   # sorted_losses = sorted(losses)
   # if nr_of_models_excluded == 0:
    #indices = list(range(len(N)))
   # else:
   #     sorted_losses = sorted(losses)
   #     indices = [i for i in range(len(N)) if losses[i] <= sorted_losses[-nr_of_models_excluded]]

    #N = training_df['Model Size'].values
    #D = training_df['Training Tokens'].values
    #losses = training_df['loss'].values

    # Set up the grid for initial parameter values
    alpha_vals = np.arange(0, 2.5, 0.5)
    beta_vals = np.arange(0, 2.5, 0.5)
    e_vals = np.arange(-1, 1.5, 0.5)
    a_vals = np.arange(0, 30, 5)
    b_vals = np.arange(0, 30, 5)

    # Perform the optimization using L-BFGS over the grid of initial values
    best_loss = np.inf
    best_params = None

    from itertools import product
    results_dict = {}
    for alpha, beta, e, a, b in product(alpha_vals, beta_vals, e_vals, a_vals, b_vals):
        init_params = [a, b, e, alpha, beta]
        result = minimize(huber_loss_objective, init_params,
                        args=(N,
                                D,
                                losses), method='L-BFGS-B')
        results_dict[tuple(init_params)] = {'params': result.x, 'loss': result.fun}
        if result.success and result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x
            best_init_params = init_params
            print(f"New best loss: {best_loss}")
            print(f"Best params: {best_params}")
            print(f"Initial guess: {init_params}")

    # Transform the fitted parameters a, b, e to A, B, E
    if best_params is not None:
        A = np.exp(best_params[0])
        B = np.exp(best_params[1])
        E = np.exp(best_params[2])
        alpha = best_params[3]
        beta = best_params[4]
        print(f"Best fit parameters: A={A}, B={B}, E={E}, alpha={alpha}, beta={beta}")
    else:
        print("Optimization failed to converge.")

    # return a dictionary with the fitted parameters
    fitted_params = {'A': A, 'B': B, 'E': E, 'N_power': alpha, 'D_power': beta}
    best_init_params = {'a': best_init_params[0], 'b': best_init_params[1], 'e': best_init_params[2], 'alpha': best_init_params[3], 'beta': best_init_params[4]}
    return fitted_params, best_init_params, results_dict





def chin_plot():



    # Define isoflops lines
    isoflops = [5.5e18, 9.7e18, 2.8e19, 5.6e19, 8.7e19, 2.8e20, 5.6e20, 1e21, 3e21, "all_else_small", "all_else_large"]
    # the last 3 isoflops lines are for testing x10
    #.  ... 4 ... x 20

    training_df = pd.read_csv('chin_data.csv')

    N = training_df['Model Size'].values
    D = training_df['Training Tokens'].values
    flops = training_df['Training FLOP'].values
    losses = training_df['loss'].values


    # find data points reasonably close to each isoflops line
    data_on_isoflops = {}
    all_isoflops_indices = []
    for fl in isoflops:
        if fl not in ["all_else_large", "all_else_small"]:
            indices = np.where((flops > fl * 0.9) & (flops < fl * 1.15))[0]
            data_on_isoflops[fl] = (N[indices], D[indices], losses[indices])
            all_isoflops_indices = np.concatenate((all_isoflops_indices, indices))
    # add all else data points
    # these are data points that are not on any isoflops line
    all_else_indices = np.setdiff1d(np.arange(len(N)), all_isoflops_indices)
    all_else_indices_leq_3e21 = all_else_indices[flops[all_else_indices] <= 3e21]
    all_else_indices_gt_3e21 = all_else_indices[flops[all_else_indices] > 3e21]
    # add these to the data_on_isoflops dictionary
    data_on_isoflops["all_else_large"] = (N[all_else_indices_gt_3e21], D[all_else_indices_gt_3e21], losses[all_else_indices_gt_3e21])
    data_on_isoflops["all_else_small"] = (N[all_else_indices_leq_3e21], D[all_else_indices_leq_3e21], losses[all_else_indices_leq_3e21])


    # print number of data points on each isoflops line
    for fl in isoflops:
        if not fl in ["all_else_large", "all_else_small"]:
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            print(f"Isoflops {fl:.1e}: {len(N_fl)} data points")
        elif fl == "all_else_large":
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            print(f"All else large (>3e21): {len(N_fl)} data points")
        elif fl == "all_else_small":
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            print(f"All else small (<=3e21): {len(N_fl)} data points")

    # make a plot of the training data,
    # scatter plot
    # x is model size N, y is training tokens D
    import matplotlib.pyplot as plt

    plt.scatter(N, D, color='gray', alpha=0.5, label='All Data')

    # plot the data points on each isoflops line with different colors
    colors = sns.color_palette("mako", n_colors=len(isoflops))
    # reverse colors so that higher isoflops lines are darker
    colors = list(reversed(colors))

    for i, fl in enumerate(isoflops):
        if not fl in ["all_else_large", "all_else_small"]:
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            plt.scatter(N_fl, D_fl, color=colors[i], label=f"Isoflops {fl:.1e}")
        elif fl == "all_else_large":
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            plt.scatter(N_fl, D_fl, color='yellow', alpha=0.5, label='All Else Data')
        elif fl == "all_else_small":
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            plt.scatter(N_fl, D_fl, color='gray', alpha=0.5, label='All Else Small Data')

    #plt.legend()
    plt.xlabel("Model Size (N)")
    plt.ylabel("Training Tokens (D)")
    # log scale both axes
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Hoffman Data")
    # save the plot
    plt.savefig("chin_training_data.png")
    plt.close()


    train_slices = [isoflops[:-2],  isoflops[:-5], isoflops[:-7]]
    test_slices = [[],[5.6e20, 1e21, 3e21],[8.7e19, 2.8e20, 5.6e20, 1e21, 3e21]]
    # make a figure with the number of subplots equal to the number of test slices
    f1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 2.2), #7.5, 1.8),
                                   constrained_layout=True)
    #f2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 2.2),
    #                               constrained_layout=True)
    # use 2x2 instead of 1x4
    f2, axes = plt.subplots(1, 3, figsize=(6.9, 2.5),
                                   constrained_layout=True)


    fit_chin = True
    if fit_chin:
      for (j, test_flops) in enumerate(test_slices):
        train_flops = train_slices[j]
        if j == 0:
            title = "(a) Chinchilla Method 3 Fitted on All IsoFLOPs"
        elif j == 1:
            title = "(b) Fit on IsoFLOPs max 3e20"
        elif j == 2:
            title = "(c) Fit on IsoFLOPs max 6e19"
        plt.sca(f1.axes[j])

        # define isoflop data to be the subset of data points on the isoflops lines
        # and that the flop is between 1e19 and 6e20
        isoflop_N = []
        isoflop_D = []
        isoflop_losses = []
        flop_values = []
        for fl in train_flops:
                N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                isoflop_N.extend(N_fl)
                isoflop_D.extend(D_fl)
                isoflop_losses.extend(losses_fl)
                flop_values.extend([fl] * len(N_fl))

        N = np.array(isoflop_N)
        D = np.array(isoflop_D)
        losses = np.array(isoflop_losses)
        flop_values = np.array(flop_values)

        # define the test data as the data points on flops outside 1e19 to 6e20 but on the isoflops lines
        test_N = []
        test_D = []
        test_losses = []
        test_flop_values = []
        for fl in test_flops:
                N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                test_N.extend(N_fl)
                test_D.extend(D_fl)
                test_losses.extend(losses_fl)
                test_flop_values.extend([fl] * len(N_fl))

        test_N = np.array(test_N)
        test_D = np.array(test_D)
        test_losses = np.array(test_losses)
        test_flop_values = np.array(test_flop_values)
        fitted_params, best_init_params, results_dict = train_chin(N, D, losses)
        print(f"Fitted parameters: {fitted_params}")
        print(f"Best initial guess: {best_init_params}")

        testt_N = []
        testt_D = []
        testt_losses = []
        testt_flop_values = []
        for fl in [3e21]:
                N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                testt_N.extend(N_fl)
                testt_D.extend(D_fl)
                testt_losses.extend(losses_fl)
                testt_flop_values.extend([fl] * len(N_fl))

        testt_N = np.array(testt_N)
        testt_D = np.array(testt_D)
        testt_losses = np.array(testt_losses)
        testt_flop_values = np.array(testt_flop_values)


        # compute the variance explained metric on both train and test data
        def variance_explained(y_true, y_pred, flop):
            y_true = np.log(y_true)
            y_pred = np.log(y_pred)
            ss_res = np.sum((y_true - y_pred) ** 2)
            #ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            # instead of the standard ss_tot, group the data by slice, and
            # compute the mean per slice, then compute ss_tot based on that
            unique_flops = np.unique(flop)
            ss_tot = 0
            for uf in unique_flops:
                indices = np.where(flop == uf)[0]
                y_true_slice = y_true[indices]
                mean_slice = np.mean(y_true_slice)
                ss_tot += np.sum((y_true_slice - mean_slice) ** 2)
            return 1 - ss_res / ss_tot


        def predict_chin(N, D, fitted_params):
            A = fitted_params['A']
            B = fitted_params['B']
            E = fitted_params['E']
            alpha = fitted_params['N_power']
            beta = fitted_params['D_power']
            preds = log_sum_exp(np.log(A), np.log(B), np.log(E), alpha, beta, N, D)
            return np.exp(preds)

        # make predictions on both train and test data
        train_preds = predict_chin(N, D, fitted_params)
        test_preds = predict_chin(test_N, test_D, fitted_params)
        testt_preds = predict_chin(testt_N, testt_D, fitted_params)


        # make a plot of predicted vs actual losses for both train and test data
        # the x-axis is actual losses, y-axis is predicted losses
        plt.scatter(losses, train_preds, color='blue', alpha=0.5) #, label='Train Data')
        plt.scatter(test_losses, test_preds, color='red', alpha=0.5, label='Test Data')
        # plot y=x line
        # if test_losses is empty, only plot train data
        if len(test_losses) == 0:
            max_loss = max(losses)
        else:
            max_loss = max(max(losses), max(test_losses))

        # compute variance explained for train and test data
        train_ve = variance_explained(losses, train_preds, flop_values)

        test_ve = variance_explained(testt_losses, testt_preds, testt_flop_values)
        print(f"Variance explained on train data: {train_ve:.4f}")
        print(f"Variance explained on test data: {test_ve:.4f}")
        # save the ve results to a json file
        ve_results = {"job": j,
                      "training_dataset_size": len(train_preds),
                        "test_dataset_size": len(test_preds),
            "train_variance_explained": train_ve,
            "test_variance_explained": test_ve
        }
        with open(f"chin_variance_explained_job_{j}.json", "w") as f:
            json.dump(ve_results, f, indent=4)

        plt.plot([0, max_loss], [0, max_loss], color='black', linestyle='-')
        plt.xlabel("Actual Loss")
        plt.ylabel("Predicted Loss")
        plt.title(title, fontsize=8)  # Reduce space between title and plot
        plt.tick_params(axis='x', which='both', labelsize=7, pad=1)  # Reduce space between ticks and labels
        plt.tick_params(axis='y', which='both', labelsize=7, pad=1)  # Reduce space between ticks and labels
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(2.1, 2.6)
        plt.legend()
        # save the plot
        plt.savefig("chin_predicted_vs_actual_losses.png")

        plt.close()

        # make isoflop plots with actual and fitted
        # x-axis is model size N, y-axis is loss
        # put all slices in the same plot
        plt.sca(f2.axes[j])
        for i, fl in enumerate(isoflops):
            if not fl in ["all_else_large", "all_else_small"]:
                N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                if len(N_fl) == 0:
                    continue
                # sort by N
                sorted_indices = np.argsort(N_fl)
                N_fl = N_fl[sorted_indices]
                D_fl = D_fl[sorted_indices]
                losses_fl = losses_fl[sorted_indices]
                preds_fl = predict_chin(N_fl, D_fl, fitted_params)
                # no shape outline
                plt.scatter(N_fl, losses_fl, s= 15,
                                ec ='none', facecolor=colors[i],
                                alpha=0.6)

                             #no shape outline,
                             #label=f"Actual Isoflops {fl:.1e}")
                plt.plot(N_fl, preds_fl, color=colors[i], linestyle='-') #, label=f"Fitted Isoflops {fl:.1e}")
            if fl == "all_else_large" and j == 10:
                N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                if len(N_fl) == 0:
                    continue
                # sort by N
                sorted_indices = np.argsort(N_fl)
                N_fl = N_fl[sorted_indices]
                D_fl = D_fl[sorted_indices]
                losses_fl = losses_fl[sorted_indices]
                preds_fl = predict_chin(N_fl, D_fl, fitted_params)
                plt.scatter(N_fl, losses_fl, fc='grey', ec = 'none', alpha=0.3, s=10, label='All Else Data')
                #plt.plot(N_fl, preds_fl, color='grey', linestyle='-', label='Fitted All Else Data')
            if fl == "all_else_small" and j == 10:
                N_fl, D_fl, losses_fl = data_on_isoflops[fl]
                if len(N_fl) == 0:
                    continue
                # sort by N
                sorted_indices = np.argsort(N_fl)
                N_fl = N_fl[sorted_indices]
                D_fl = D_fl[sorted_indices]
                losses_fl = losses_fl[sorted_indices]
                preds_fl = predict_chin(N_fl, D_fl, fitted_params)
                plt.scatter(N_fl, losses_fl, fc='gray', ec = 'none', alpha=0.3, s=10, label='All Else Small Data')
                #plt.plot(N_fl, preds_fl, color='gray', linestyle='-', label='Fitted All Else Small Data')



        plt.title(title, fontsize=9)  # Reduce space between title and plot
        plt.tick_params(axis='x', which='both', labelsize=7, pad=1)  # Reduce space between ticks and labels
        plt.tick_params(axis='y', which='both', labelsize=7, pad=1)  # Reduce space between ticks and labels
        if j in [0,1,2,3]:
            plt.xlabel("Model Size (N)", fontsize=9)
        if j in [0]:
            plt.ylabel("Loss", fontsize=9)
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(2.0, 3.5)
        #plt.title("Isoflop Slices: Actual vs Fitted Losses")
        plt.savefig("chin_on_hoffman_isoflop_slices.png")
        plt.savefig('chin_on_hoffman_isoflop_slices.pdf', bbox_inches='tight', pad_inches=0.0)
        plt.close()

    return None


def chin_plot_styled(color_palette="magma"):
    """
    Styled version of chin_plot with only 2 panels (omitting the middle one).
    Applies the same style as the isoflop figures in plotting_functions.py:
    - Reversed color palette
    - Bold axis labels
    - No top/right spines, thicker left/bottom spines
    - Bold, larger font for labels
    - Left panel: only top and bottom C labels
    - Right panel: inset for training, main plot for test (heldout)
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Define isoflops lines
    isoflops = [5.5e18, 9.7e18, 2.8e19, 5.6e19, 8.7e19, 2.8e20, 5.6e20, 1e21, 3e21, "all_else_small", "all_else_large"]
    # Numeric isoflops only (for indexing)
    numeric_isoflops = [fl for fl in isoflops if fl not in ["all_else_large", "all_else_small"]]

    training_df = pd.read_csv('chin_data.csv')

    N = training_df['Model Size'].values
    D = training_df['Training Tokens'].values
    flops = training_df['Training FLOP'].values
    losses = training_df['loss'].values

    # find data points reasonably close to each isoflops line
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

    # Use reversed color palette (style from plotting_functions.py)
    colors = sns.color_palette(color_palette, n_colors=len(isoflops))[::-1]

    # Helper to format FLOP labels
    def format_flop_label(fl):
        if fl >= 1e21:
            return f"  {fl/1e21:.0f}e21"
        elif fl >= 1e20:
            return f"  {fl/1e20:.0f}e20"
        elif fl >= 1e19:
            return f"  {fl/1e19:.1f}e19"
        else:
            return f"  {fl/1e18:.1f}e18"

    def predict_chin(N, D, fitted_params):
        A = fitted_params['A']
        B = fitted_params['B']
        E = fitted_params['E']
        alpha = fitted_params['N_power']
        beta = fitted_params['D_power']
        preds = log_sum_exp(np.log(A), np.log(B), np.log(E), alpha, beta, N, D)
        return np.exp(preds)

    # Create figure with 2 panels (stacked vertically)
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))

    # ============ LEFT PANEL (panel_idx=0): All IsoFLOPs, only label top and bottom ============
    ax = axs[0]
    train_flops_left = isoflops[:-2]  # max 3e21

    # Collect training data
    isoflop_N = []
    isoflop_D = []
    isoflop_losses = []
    for fl in train_flops_left:
        if fl not in ["all_else_large", "all_else_small"]:
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            isoflop_N.extend(N_fl)
            isoflop_D.extend(D_fl)
            isoflop_losses.extend(losses_fl)

    N_train = np.array(isoflop_N)
    D_train = np.array(isoflop_D)
    losses_train = np.array(isoflop_losses)

    # Fit Chinchilla model
    fitted_params_left, _, _ = train_chin(N_train, D_train, losses_train)
    print(f"Left Panel: Fitted parameters: {fitted_params_left}")

    # Get min and max flops for labeling (only top and bottom)
    numeric_train_flops = [fl for fl in train_flops_left if fl not in ["all_else_large", "all_else_small"]]
    min_flop = min(numeric_train_flops)
    max_flop = max(numeric_train_flops)

    # Plot each isoflop curve
    for i, fl in enumerate(isoflops):
        if fl not in ["all_else_large", "all_else_small"]:
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            if len(N_fl) == 0:
                continue
            sorted_indices = np.argsort(N_fl)
            N_fl = N_fl[sorted_indices]
            D_fl = D_fl[sorted_indices]
            losses_fl = losses_fl[sorted_indices]
            preds_fl = predict_chin(N_fl, D_fl, fitted_params_left)

            ax.scatter(N_fl, losses_fl, s=40, ec='none', facecolor=colors[i], 
                       alpha=0.8)
            ax.plot(N_fl, preds_fl, color=colors[i], 
                    linestyle='-', linewidth=2.5, alpha=0.7)

            # Only label the largest compute slice
            if fl == max_flop:
                label_fl = format_flop_label(fl)
                ax.text(N_fl[-1] * 1.05, preds_fl[-1], label_fl,
                       color=colors[i], 
                       fontsize=10, fontweight='bold', va='center')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Size (N)", fontweight='bold')
    ax.set_ylabel("Loss", fontweight='bold')
    ax.set_title("(a) Chinchilla Method 3\nFitted on All IsoFLOPs", fontweight='bold')

    # Use same axis limits for both panels
    ax.set_xlim(3e7, 2e10)
    ax.set_ylim(2.1, 3.3)
    ax.set_yticks([2.5, 3.0])
    ax.set_yticklabels(['2.5', '3.0'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # ============ BOTTOM PANEL: Train (blue) and Test (red) on same plot ============
    ax = axs[1]
    train_flops_right = isoflops[:-7]  # max 6e19
    test_flops_right = [1e21, 3e21]  # Only the 2 highest

    # Colors for train and test
    COLOR_TRAIN = 'blue'
    COLOR_TEST = 'red'

    # Collect training data
    isoflop_N = []
    isoflop_D = []
    isoflop_losses = []
    for fl in train_flops_right:
        if fl not in ["all_else_large", "all_else_small"]:
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            isoflop_N.extend(N_fl)
            isoflop_D.extend(D_fl)
            isoflop_losses.extend(losses_fl)

    N_train = np.array(isoflop_N)
    D_train = np.array(isoflop_D)
    losses_train = np.array(isoflop_losses)

    # Fit Chinchilla model
    fitted_params_right, _, _ = train_chin(N_train, D_train, losses_train)
    print(f"Bottom Panel: Fitted parameters: {fitted_params_right}")

    # Plot TRAINING isoflop curves (blue)
    max_train_flop = max([f for f in train_flops_right if f not in ["all_else_large", "all_else_small"]])
    for fl in train_flops_right:
        if fl not in ["all_else_large", "all_else_small"]:
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            if len(N_fl) == 0:
                continue
            sorted_indices = np.argsort(N_fl)
            N_fl = N_fl[sorted_indices]
            D_fl = D_fl[sorted_indices]
            losses_fl = losses_fl[sorted_indices]
            preds_fl = predict_chin(N_fl, D_fl, fitted_params_right)

            ax.scatter(N_fl, losses_fl, s=40, ec='none', facecolor=COLOR_TRAIN, alpha=0.8)
            ax.plot(N_fl, preds_fl, color=COLOR_TRAIN, linestyle='-', linewidth=2.5, alpha=0.7)

            # Only label the largest training slice
            if fl == max_train_flop:
                label_fl = format_flop_label(fl)
                ax.text(N_fl[-1] * 1.05, preds_fl[-1], label_fl,
                       color=COLOR_TRAIN, fontsize=10, fontweight='bold', va='center')

    # Plot TEST isoflop curves (red)
    for fl in test_flops_right:
        N_fl, D_fl, losses_fl = data_on_isoflops[fl]
        if len(N_fl) == 0:
            continue
        sorted_indices = np.argsort(N_fl)
        N_fl = N_fl[sorted_indices]
        D_fl = D_fl[sorted_indices]
        losses_fl = losses_fl[sorted_indices]
        preds_fl = predict_chin(N_fl, D_fl, fitted_params_right)

        ax.scatter(N_fl, losses_fl, s=40, ec='none', facecolor=COLOR_TEST, alpha=0.8)
        ax.plot(N_fl, preds_fl, color=COLOR_TEST, linestyle='-', linewidth=2.5, alpha=0.7)

        # Only label the largest test slice
        if fl == max(test_flops_right):
            label_fl = format_flop_label(fl)
            ax.text(N_fl[-1] * 1.05, preds_fl[-1], label_fl,
                   color=COLOR_TEST, fontsize=10, fontweight='bold', va='center')

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Size (N)", fontweight='bold')
    ax.set_ylabel("Loss", fontweight='bold')
    ax.set_title("(b) Chinchilla Method 3\nTrain vs Holdout IsoFLOPs", fontweight='bold')

    ax.set_xlim(3e7, 2e10)
    ax.set_ylim(2.1, 3.3)
    ax.set_yticks([2.5, 3.0])
    ax.set_yticklabels(['2.5', '3.0'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Create custom legend entries (centered at bottom)
    legend_elements = [
        Line2D([0], [0], marker='o', color='blue', linestyle='-', linewidth=2.5, markersize=8, alpha=0.7, label='Train'),
        Line2D([0], [0], marker='o', color='red', linestyle='-', linewidth=2.5, markersize=8, alpha=0.7, label='Holdout'),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc='lower center',
               bbox_to_anchor=(0.5, -0.015), ncol=2, prop={'weight': 'bold'})

    # Save figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for legend below plot
    plt.savefig("chin_on_hoffman_isoflop_slices_styled.png", dpi=300)
    plt.savefig("chin_on_hoffman_isoflop_slices_styled.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.close()

    return None


if __name__ == "__main__":
    # Run the new styled plot with 2 panels
    chin_plot_styled(color_palette="magma")
