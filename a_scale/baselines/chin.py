import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import erf
import scipy.stats as stats
from scipy.stats import norm

from a_scale.nqs.nqs_sgd import _fit_nqs, latin_hypercube_initializations, _risk_no_sch

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


# start of main script
if __name__ == "__main__":
    

    
    training_df = pd.read_csv('chin_data.csv')

    N = training_df['Model Size'].values
    D = training_df['Training Tokens'].values
    flops = training_df['Training FLOP'].values
    losses = training_df['loss'].values

    # Define isoflops lines
    isoflops = [6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21]
    # the last 3 isoflops lines are for testing x10
    #.  ... 4 ... x 20
    test_flops = [3e20, 6e20, 1e21, 3e21]

    # find data points reasonably close to each isoflops line
    data_on_isoflops = {}
    for fl in isoflops:
        indices = np.where((flops > fl * 0.9) & (flops < fl * 1.15))[0]
        data_on_isoflops[fl] = (N[indices], D[indices], losses[indices])
    # print number of data points on each isoflops line
    for fl in isoflops:
        N_fl, D_fl, losses_fl = data_on_isoflops[fl]
        print(f"Isoflops {fl:.1e}: {len(N_fl)} data points")

    # make a plot of the training data, 
    # scatter plot
    # x is model size N, y is training tokens D
    import matplotlib.pyplot as plt

    plt.scatter(N, D, color='gray', alpha=0.5, label='All Data')

    # plot the data points on each isoflops line with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(isoflops)))
    for i, fl in enumerate(isoflops):
        N_fl, D_fl, losses_fl = data_on_isoflops[fl]
        plt.scatter(N_fl, D_fl, color=colors[i], label=f"Isoflops {fl:.1e}")
    #plt.legend()
    plt.xlabel("Model Size (N)")
    plt.ylabel("Training Tokens (D)")
    # log scale both axes
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Training Data")
    # save the plot
    plt.savefig("chin_training_data.png")
    plt.close()


    # define isoflop data to be the subset of data points on the isoflops lines
    # and that the flop is between 1e19 and 6e20
    isoflop_N = []
    isoflop_D = []
    isoflop_losses = []
    for fl in isoflops:
        if fl not in test_flops:
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            isoflop_N.extend(N_fl)
            isoflop_D.extend(D_fl)
            isoflop_losses.extend(losses_fl)

    N = np.array(isoflop_N)
    D = np.array(isoflop_D)
    losses = np.array(isoflop_losses)

    # define the test data as the data points on flops outside 1e19 to 6e20 but on the isoflops lines
    test_N = []
    test_D = []
    test_losses = []
    for fl in isoflops:
        if fl in test_flops:
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            test_N.extend(N_fl)
            test_D.extend(D_fl)
            test_losses.extend(losses_fl)

    test_N = np.array(test_N)
    test_D = np.array(test_D)
    test_losses = np.array(test_losses)

    fit_chin = False
    if fit_chin:
        fitted_params, best_init_params, results_dict = train_chin(N, D, losses)
        print(f"Fitted parameters: {fitted_params}")
        print(f"Best initial guess: {best_init_params}")

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
        plt.plot([0, max_loss], [0, max_loss], color='black', linestyle='--')
        plt.xlabel("Actual Loss")
        plt.ylabel("Predicted Loss")
        plt.xscale("log")
        plt.yscale("log")
        # set max y to be 3.5
        plt.ylim(2, 3.5)
        plt.title("Predicted vs Actual Losses")
        plt.legend()
        # save the plot
        plt.savefig("chin_predicted_vs_actual_losses.png")
        plt.close()

        # make isoflop plots with actual and fitted
        # x-axis is model size N, y-axis is loss
        # put all slices in the same plot
        plt.figure()
        for i, fl in enumerate(isoflops):
            N_fl, D_fl, losses_fl = data_on_isoflops[fl]
            if len(N_fl) == 0:
                continue
            # sort by N
            sorted_indices = np.argsort(N_fl)
            N_fl = N_fl[sorted_indices]
            D_fl = D_fl[sorted_indices]
            losses_fl = losses_fl[sorted_indices]
            preds_fl = predict_chin(N_fl, D_fl, fitted_params)
            plt.scatter(N_fl, losses_fl, color=colors[i], alpha=0.5) #, label=f"Actual Isoflops {fl:.1e}")
            plt.plot(N_fl, preds_fl, color=colors[i], linestyle='--') #, label=f"Fitted Isoflops {fl:.1e}")
        plt.xlabel("Model Size (N)")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(2, 3.5)
        plt.title("Isoflop Slices: Actual vs Fitted Losses")
        #plt.legend()
        # save the plot
        plt.savefig("chin_isoflop_slices.png")
        plt.close()



    list_of_nqs_inits = latin_hypercube_initializations(
        seed = 42,
        num_inits = 1000,
        param_names = ['p', 'q', 'P', 'Q', 'e_irr', 'R', 'r'],
        param_ranges = {
            'p': (0.8, 2.0),
            'q': (0.6, 1.2),
            'P': (0.1, 5.0),
            'Q': (0.1, 5.0),
            'e_irr': (0.01, 1.0),
            'R': (0.5, 10.0),
            'r': (0.6, 1.2)
        } ,
        r_equals_q = True  
    )


    def BK_from_D(D):
        # approximate B = 0.06 * D^0.5
        B = 0.06 * (D ** 0.5)
        B = int(B)
        K = D/6/B
        K = int(K)
        return B, K
    
    default_lr = 2.0
    BKs = [BK_from_D(D_i) for D_i in D]
    cfgs = [[N_i, K_i, B_i, default_lr] for N_i, (B_i, K_i) in zip(N, BKs)]
    ys = losses
    num_itrs = 5
    best_nqs, best_loss, best_idx, trajectories = _fit_nqs(list_of_nqs_inits, 
                                                    cfgs, 
                                                    ys, 
                                                    itrs=num_itrs, 
                                                    return_trajectories=True)
    
    print(f"Best NQS params: {best_nqs}")
    print(f"Best NQS loss: {best_loss}")

    