# read the csv_file h_samples_with_nn_loss.csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import numpy as np
from a_scale.run_nn import build_hf_jax
import os

seq_len = 2048


def get_N_color(N):
    """
    Get a color based on the model size N.
    """
    cmap = plt.get_cmap('viridis', len(N_unique_sorted))
    index = np.where(N_unique_sorted == N)[0][0]
    return cmap(index)


# raise error with the unique model sizes
#if dat['N'].nunique() > 1:
 #   raise ValueError(f"The  contains more than one unique model size N: {dat['N'].unique()}. Please filter the dataset to a single model size before plotting.")
# multiply K by 100
#dat['K'] = dat['K'] * 100  # Convert K to thousands for better scaling
#dat['B'] = dat['B']/100
# print the first 5 rows
#dat = pd.read_csv('h_samples_with_nn_loss.csv')

# Keep the original power law fit function unchanged
def fit_curve_power(D, loss, return_params=False):
    def model(D, E_N, D_const, beta):
        return E_N + D_const * D**(-beta)

    # Initial guess for parameters
    initial_guess = [1.0, 30, 0.5]
    
    # Fit the model to the data
    params, _ = curve_fit(model, D, loss, p0=initial_guess, maxfev=10000)

    # define a function that makes predictions
    def predict(D):
        return model(D, *params)
    
    # return the fitted values on the input range, but with finer granularity
    D_min = D.min()
    D_max = D.max()
    D_fine = pd.Series([D_min * (D_max / D_min)**(i / 100) for i in range(101)])
    if return_params:
        return (D_fine, predict(D_fine), params)
    else:
        return (D_fine, predict(D_fine))

# 
def fit_curve(D, loss, return_params=False):
    def model(D, m, c):
        return np.exp(m * np.log(D) + c)
    # Initial guess for parameters
    initial_guess = [ -0.5, 1.0]
    params, _ = curve_fit(model, D, loss, p0=initial_guess, maxfev=10000)
    def predict(D):
        return model(D, *params)
    # return the fitted values on the input range, but with finer granularity
    D_min = D.min()
    D_max = D.max()
    D_fine = pd.Series([D_min * (D_max / D_min)**(i / 100) for i in range(101)])
    if return_params:
        return (D_fine, predict(D_fine), params)
    else:
        return (D_fine, predict(D_fine))

# Add hyperbola fitting functions using least squares method
def residuals_hyperbola(p, x, y):
    """
    Residual function for hyperbola fitting
    Hyperbola equation: (x - h)*(y - k - m*(x - h)) = a^2
    """
    h, k, a = p #m, a = p
    return (x - h)*(y - k - 0*(x - h)) - a**2

def empirical_law(p, x, y):
    S_min, E_min = p
    return (y/E_min - 1)*(x/S_min - 1) - 1

def fit_hyperbola_least_squares(x_data, y_data):
    """
    Fit hyperbola using least squares method and return fitted curve points
    """
    # Crude initial guesses
    p0 = [np.min(x_data),
          np.min(y_data)]
         # 2,    # slope m (must stay >0)
         # 1e2]    # scale a (must stay >0)
    
    try:
        # constraint h to be less than min of x_data
        S_min_max = np.min(x_data)
        E_min_max = np.min(y_data)
        sol = least_squares(empirical_law, p0,
                           args=(x_data, y_data),
                           #bounds=([-np.inf, -np.inf, 1e-6, 1e-6],
                           #        [np.inf, np.inf, np.inf, np.inf]))
                           bounds = ([-np.inf, -np.inf],
                                     [S_min_max, E_min_max]))

        if sol.success:
            # Extract fitted parameters
            S_min_fit, E_min_fit = sol.x
            #h_fit, k_fit, a_fit = sol.x #m_fit, a_fit = sol.x
            #m_fit = 0
           # print(f'Hyperbola fitted parameters: h={h_fit:.3f}, k={k_fit:.3f}, m={m_fit:.3f}, a={a_fit:.3f}')
            print(f"Emprical law fitted parameters: S_min={S_min_fit:.3f}, E_min={E_min_fit:.3f}")
            # Generate fitted curve points
            x_range = np.linspace(min(x_data) * 0.8, max(x_data) * 1.2, 200)
            
            # Remove points too close to the vertical asymptote
            #x_range = x_range[np.abs(x_range - h_fit) > 0.01]
            
            # Calculate hyperbola: y = k + m*(x - h) + a^2/(x - h)
            y_fitted = ((x_range / S_min_fit - 1)**(-1) + 1)* E_min_fit
            #k_fit + m_fit*(x_range - h_fit) + a_fit**2/(x_range - h_fit)
            
            # Filter to reasonable range
            y_min = min(y_data) * 0.5
            y_max = max(y_data) * 2.0
            mask = (y_fitted >= y_min) & (y_fitted <= y_max)
            
            print(f"Fitted curve ")            # Return fitted curve points and parameters
            print(f"S_min_fit: {S_min_fit}, E_min_fit: {E_min_fit}")
            #print(f"Hyperbola fitting successful: h={h_fit}, k={k_fit}, m={m_fit}, a={a_fit}")
            return x_range[mask], y_fitted[mask], (S_min_fit, E_min_fit)    
        else:
            raise ValueError("Hyperbola fitting failed, using fallback")
            return None, None, None
    except Exception as e:
        raise ValueError(f"Hyperbola fitting error: {e}, using fallback")
        return None, None, None


def compute_vertex_intersection(S_min, E_min):
    return 2* S_min, 2 * E_min, 0 

def compute_vertex_intersection_old(h, k, m, a):
    """
    Compute the vertex = intersection of top-right branch with transverse axis
    """
    
    s = m
    #s = m + np.sqrt(m**2 + 1)  # slope of transverse axis
    
    x_vertex = h + a
    y_vertex = k + (m+1)*(a) 
    # Vertex coordinates
    #u_vertex = a / np.sqrt(s - m)        # offset in x from centre
    #x_vertex = h + u_vertex
    #y_vertex = k + s * u_vertex

    #s = m
    # Vertex coordinates: find intersection of transverse axis with hyperbola
    # i.e. find the point  y = k + s * (x - h) and y = k + m*(x - h) + a**2/(x - h)
    # Rearranging gives us the quadratic equation:
    # a^2 - (s - m) * x + (k - h * m + h * s) = 0
    #A = 1
    #B = -(s - m)
    #C = k - h * m + h * s - a**2
    #discriminant = B**2 - 4 * A * C
    #if discriminant < 0:
    #    raise ValueError("No intersection found, hyperbola may not be valid")
    #sqrt_discriminant = np.sqrt(discriminant)
    #x_vertex = (-B + sqrt_discriminant) / (2 * A)
    #y_vertex = k + s * (x_vertex - h)
    
    return x_vertex, y_vertex, s


def compute_vertex_intersection_old(h, k, m, a):
    """
    Compute the vertex = intersection of top-right branch with transverse axis
    """
    s = m + np.sqrt(m**2 + 1)  # slope of transverse axis
    
    # Vertex coordinates
    u_vertex = a / np.sqrt(s - m)        # offset in x from centre
    x_vertex = h + u_vertex
    y_vertex = k + s * u_vertex

    #s = m
    # Vertex coordinates: find intersection of transverse axis with hyperbola
    # i.e. find the point  y = k + s * (x - h) and y = k + m*(x - h) + a**2/(x - h)
    # Rearranging gives us the quadratic equation:
    # a^2 - (s - m) * x + (k - h * m + h * s) = 0
    #A = 1
    #B = -(s - m)
    #C = k - h * m + h * s - a**2
    #discriminant = B**2 - 4 * A * C
    #if discriminant < 0:
    #    raise ValueError("No intersection found, hyperbola may not be valid")
    #sqrt_discriminant = np.sqrt(discriminant)
    #x_vertex = (-B + sqrt_discriminant) / (2 * A)
    #y_vertex = k + s * (x_vertex - h)
    
    return x_vertex, y_vertex, s

# Keep the original fit_curve_K function unchanged
def fit_curve_K(K, D, return_params=False):
    def model(K, D_min, K_min):
        return D_min * ((K / K_min - 1)**(-1) + 1)

    # Initial guess for parameters
    initial_guess = [1e6, 100]
    
    # Fit the model to the data
    params, _ = curve_fit(model, K, D, p0=initial_guess, maxfev=10000)
    print(f'Fitted parameters: {params}')

    # define a function that makes predictions
    def predict(K):
        return model(K, *params)
    
    # return the fitted values on the input range, but with finer granularity
    K_min = K.min()
    K_max = K.max()
    K_fine = pd.Series([K_min * (K_max / K_min)**(i / 100) for i in range(101)])
    if return_params:
        return (K_fine, predict(K_fine), params)
    else:
        return (K_fine, predict(K_fine))

loss_values = [1.26, 1.27, 1.28, 1.29, 1.30, 1.31, 1.32, 1.33]

#[3.0,3.1,3.2,3.3,3.4, 3.5] #, 4.2, 4.4, 4.6]#, 4.8, 5.0]

def plot_loss_vs_D_subplot(dat, to_plot_col, ax1, ax2):
    """
    Modular function to plot loss vs D for a given column on specified axes
    Returns iso_loss_df for potential reuse
    """
    # check if dat contains more than one unique model size N
    if dat['N'].nunique() > 1:
        raise ValueError("The dataset contains more than one unique model size N. Please filter the dataset to a single model size before plotting.")
    else:
        N = dat['N'].unique()[0]
        print(f"Plotting for model size N={N}, column={to_plot_col}")
    
    # Get unique batch sizes
    batch_sizes = dat['B'].unique()
    batch_sizes.sort()
    
    # Create a colormap
    cmap = plt.get_cmap('viridis', len(batch_sizes))
    iso_loss_df = pd.DataFrame(columns=['B', 'D', to_plot_col])
    
    # First subplot: loss vs D (unchanged - still uses power law fit)
    for i, B in enumerate(batch_sizes):
        if len(dat[dat['B'] == B]) < 3:
            continue
        subset = dat[dat['B'] == B]
        D = subset['D']
        loss = subset[to_plot_col]
        
        # Fit the curve using original power law

        D_fine, loss_fit = fit_curve(D, loss)
        min_loss_fit = loss_fit.min()
        max_loss_fit = loss_fit.max()
        
        # Store iso-loss data
        for loss_value in loss_values:
            print(f"loss_value: {loss_value}, min_loss_fit: {min_loss_fit}, max_loss_fit: {max_loss_fit}")
            if min_loss_fit <= loss_value <= max_loss_fit:
                index = (loss_fit <= loss_value).argmax()
                c_fine_value = D_fine.iloc[index]
                row_df = pd.DataFrame({'B': [B], 'D': [c_fine_value], to_plot_col: [loss_value], 'N': [N]})
                iso_loss_df = pd.concat([iso_loss_df, row_df], ignore_index=True)
        #raise ValueError(f"iso_loss_df: {tst}")
        # Plot on first axis
        ax1.scatter(D, loss, label=f'B={B}', color=cmap(i), alpha=0.6)
        ax1.plot(D_fine, loss_fit, color=cmap(i), linestyle='--')
        #if B == 384:
         #   raise ValueError(f"D:{D}, loss: {loss}, D_fine: {D_fine}, loss_fit: {loss_fit}")
        
        # add a dot where the loss is 1.3
        if (loss_fit <= 1.3).any():
            index = (loss_fit <= 1.3).argmax()
            c_fine_value = D_fine.iloc[index]
            ax1.scatter(c_fine_value, 1.3, color="grey", alpha=0.5, marker='o', s=100, edgecolor='black')

    # Format first subplot
    # get model size
    model_size = dat['N'].unique()[0]
    ax1.axhline(y=1.3, alpha=0.5, color='grey', linestyle='--', label='Loss = 1.3'+ ' N is ' + str(model_size), linewidth=5)
    ax1.set_xlabel('Total Data Budget (D)')
    ax1.set_ylabel(f'{to_plot_col}')
    ax1.set_title(f'{to_plot_col} vs Total Data Budget by Batch Size')
    ax1.legend(title='Batch Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # Second subplot: iso-loss curves
    iso_loss_df = iso_loss_df.sort_values(by=[to_plot_col, 'B'])
    iso_loss_df['K'] = iso_loss_df['D'] / iso_loss_df['B'] /seq_len
    iso_loss_df['K'] = iso_loss_df['K'].round().astype(int)
    iso_loss_df['D'] = iso_loss_df['D'] /token_unit
    
    cmapp = plt.get_cmap('plasma', len(loss_values))
    
    
    for loss_value in loss_values:
        subset = iso_loss_df[iso_loss_df[to_plot_col] == loss_value]
        if len(subset) == 0:
            continue
            
        color = cmapp(loss_values.index(loss_value) / len(loss_values))
        
        # Plot data points
        ax2.scatter(subset['K'], subset['D'], label=f'Loss = {loss_value}', alpha=0.6, color=color)
        
        # Use different fitting methods based on loss type
        if len(subset) >= 3: # and to_plot_col == 'nqs_loss':
            # Use hyperbola fit for NN_loss
            K_fitted, D_fitted, params = fit_hyperbola_least_squares(subset['K'].values, subset['D'].values)
            
            if K_fitted is not None and D_fitted is not None and params is not None:
                ax2.plot(K_fitted, D_fitted, linestyle='-', color=color, alpha=0.8, linewidth=2)
                
                #h_fit, k_fit, m_fit, a_fit = params
                S_min, E_min = params
                # Compute and plot vertex (intersection of transverse axis with hyperbola)
                try:
                    x_vertex, y_vertex, s = compute_vertex_intersection(S_min, E_min)
                    ax2.scatter(x_vertex, y_vertex, color=color, marker='*', s=200, 
                               edgecolor='black', alpha=0.9, linewidth=2,
                               label=f'Vertex' if loss_value == loss_values[0] else "")
                    print(f'Loss {loss_value}: Vertex at ({x_vertex:.1f}, {y_vertex:.1f})')

                    #x_axis = np.array([min(iso_loss_df['K']), max(iso_loss_df['K'])])  # Extend line across full range
                    # use linspace to ensure full range
                    x_axis = np.linspace(min(iso_loss_df['K']), max(iso_loss_df['K']), 200)
                    #s = m_fit + np.sqrt(m_fit**2 + 1)  # slope of transverse axis
                    y_axis = ((x_axis / S_min - 1)**(-1) + 1) * E_min
                    # Plot the straight line
                    ax2.plot(x_axis, y_axis, color=color, linestyle='-.', 
                            alpha=0.5, linewidth=3, label=f'Transverse Axis' if loss_value == loss_values[0] else "")

                    # put a + at the center of the hyperbola
                    #ax2.scatter(h_fit, k_fit, color=color, marker='+', s=150,
                    #         edgecolor='black', alpha=0.9, linewidth=2)

                
                    y_min = subset['D'].min()
                    ax2.scatter(S_min, y_min, color=color, marker='+', s=150,
                             edgecolor='black', alpha=0.9, linewidth=2)

                    y_axis2 = E_min #+ m_fit * (x_axis - h_fit) 
                  
                  #  ax2.plot(x_axis, y_axis2, color=color, linestyle='--',
                  #          alpha=0.5, linewidth=1.5, label=f'Asymptote' if loss_value == loss_values[0] else "")
                    # the other asymptote is a vertical line at h_fit
                    ax2.axvline(x=S_min, color=color, linestyle='--',
                            alpha=0.5, linewidth=1.5)
                    
                except Exception as e:
                    print(f'Could not compute vertex for loss {loss_value}: {e}')
                    # Fallback to center
                  #  ax2.scatter(h_fit, k_fit, color=color, marker='+', s=150, 
                     #          edgecolor='black', alpha=0.9, linewidth=2)
                
                # Store parameters for transverse axis

                
            else:
                # Fallback to simple line connection
                sorted_indices = np.argsort(subset['K'])
                ax2.plot(subset['K'].iloc[sorted_indices], subset['D'].iloc[sorted_indices], 
                        linestyle='--', color=color, alpha=0.7)
                
        elif False and len(subset) >= 3 and to_plot_col != 'NN_loss':
            # Keep original power law fit for non-NN loss
            try:
                K_fine, D_fitted, K_D_params = fit_curve_K(subset['K'], subset['D'], return_params=True)
                ax2.plot(K_fine, D_fitted, linestyle='--', color=color)
                ax2.scatter(2*K_D_params[1], 2*K_D_params[0], 
                           color=color, marker='+', s=100, edgecolor='black')
            except:
                ax2.plot(subset['K'], subset['D'], linestyle='--', color=color)
        else:
            # Connect points with simple line for insufficient data
            sorted_indices = np.argsort(subset['K'])
            ax2.plot(subset['K'].iloc[sorted_indices], subset['D'].iloc[sorted_indices], 
                    linestyle='--', color=color)
    

    
    # Format second subplot
    ax2.set_xlabel('Total Optimization Steps (K)')
    ax2.set_ylabel('Total Data Budget (D) in '+ str(token_unit) + ' tokens')
    ax2.set_xlim(1e2, 1e5)
    ax2.set_ylim(1e4, 1e7)
    ax2.set_title('Iso-Loss Curves')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True)


    
    return iso_loss_df

def plot_combined_iso_loss(iso_loss_df_nn, iso_loss_df_nqs, ax):
    """
    Plot combined iso-loss curves from both NN_loss and nqs_loss on a single axis
    """
    cmapp = plt.get_cmap('plasma', len(loss_values))
    
    # Store parameters for transverse axis plotting
    transverse_axes = []
    plotting_df = []
    # Plot NN_loss curves - circles with hyperbola fitted curves and vertex marks
    list_of_vertices = []  # Store vertex points for transverse axis fitting
    for loss_value in loss_values:
        subset = iso_loss_df_nn[iso_loss_df_nn['NN_loss'] == loss_value]
        if len(subset) > 0:
            color = cmapp(loss_values.index(loss_value) / len(loss_values))
            
            # Plot circles only
           # ax.scatter(subset['K'], subset['D'], label=f'NN Loss = {loss_value}', 
           #           alpha=0.8, marker='o', s=50, color=color)
            
            # Add hyperbola fitted curve
            if len(subset) >= 3:
                # add subset K, D to plotting_df
                for ka, da in zip(subset['K'], subset['D']):
                    plotting_df.append({'x': ka, 'y': da, 'loss_level': loss_value, 'type': 'nn_data_point'})
                K_fitted, D_fitted, params = fit_hyperbola_least_squares(subset['K'].values, subset['D'].values)
                # add K_fitted, D_fitted to plotting_df
                for kf, df in zip(K_fitted, D_fitted):
                    plotting_df.append({'x': kf, 'y': df, 'loss_level': loss_value, 'type': 'fitted_curve'})
                
                if K_fitted is not None and D_fitted is not None and params is not None:
                   # ax.plot(K_fitted, D_fitted, linestyle='-', alpha=0.7, color=color, linewidth=2)
                    
                    S_min, E_min = params
                    
                    # Compute and plot vertex (intersection of transverse axis with hyperbola)
                    if True:
                    #try:
                        x_vertex, y_vertex, s = compute_vertex_intersection(S_min, E_min)
                        list_of_vertices.append((x_vertex, y_vertex))
                        #raise ValueError(f"vertices: {vertices}")
                        ax.scatter(x_vertex, y_vertex, color=color, marker='*', s=150, 
                                 edgecolor='black', alpha=0.3, linewidth=2)
                        # add vertex to the plotting_df
                        # add a dictionary, keys are x_coord, y_coord, loss_value, type
                        plotting_df.append({'x': x_vertex, 'y': y_vertex, 'loss_level': loss_value, 'type': 'fitted_critical_point'})
                    #except:
                    #    # Fallback to center
                    #    ax.scatter(h_fit, k_fit, color=color, marker='+', s=120, 
                    #             edgecolor='black', alpha=0.9, linewidth=2)
                    
                    # Store parameters for transverse axis
                        #x_axis = np.array([min(iso_loss_df_nn['K']), max(iso_loss_df_nn['K'])])  # Extend line across full range
                        # use linspace to ensure full range
                       # x_axis = np.linspace(min(iso_loss_df_nn['K']), max(iso_loss_df_nn['K']), 200)
                       # y_axis = k_fit + s * (x_axis - h_fit)
                        # Plot the straight line
                       # ax.plot(x_axis, y_axis, color=color, linestyle='-.', 
                       #        alpha=0.5, linewidth=1.5, label=f'Transverse Axis' if loss_value == loss_values[0] else "")
                else:
                    # Fallback
                    sorted_indices = np.argsort(subset['K'])
                    ax.plot(subset['K'].iloc[sorted_indices], subset['D'].iloc[sorted_indices], 
                           linestyle='-', alpha=0.7, color=color)
            #raise ValueError(f"vertices: {vertices}")
        # fit a straight line to the vertices
    # fit a line to the vertices
    

    #convert plotting_df to a dataframe, and save to csv
    plotting_df = pd.DataFrame(plotting_df)
    # rename x to K, y to D
    plotting_df = plotting_df.rename(columns={'x': 'K', 'y': 'D'})
    # multiply D by token_unit
    plotting_df['D'] = plotting_df['D'] * token_unit
    # calculate B = D/seq_len/K
    plotting_df['B'] = plotting_df['D'] / seq_len / plotting_df['K']
    # convert K, B, D to integers
    plotting_df['K'] = plotting_df['K'].round().astype(int)
    plotting_df['B'] = plotting_df['B'].round().astype(int)
    plotting_df['D'] = plotting_df['D'].round().astype(int)
    plotting_df.to_csv('plotting_data_isoloss_curves_for_critical_batch_size_adamw_cosine.csv', index=False)
    #raise ValueError(f"plotting_df: {plotting_df.head(20)}")

    if len(list_of_vertices) >= 2:
        vertices = np.array(list_of_vertices)
        # Fit a line to the vertices (log-log scale)
        p = np.polyfit(np.log10(vertices[:, 0]), np.log10(vertices[:, 1]), 1)
        x_fit = np.linspace(min(iso_loss_df_nn['K']), max(iso_loss_df_nn['K']), 200)
        y_fit = 10**(p[0] * np.log10(x_fit) + p[1])
        ax.plot(x_fit, y_fit, 
                #color='black', 
                color=get_N_color(iso_loss_df_nn['N'].unique()[0]),
                linestyle='--', alpha=0.5, linewidth=1.5,
                # label with slope and intercept
                label='Fitted, NN Loss Vertices, N=' + str(iso_loss_df_nn['N'].unique()[0]) + 
                f', slope={p[0]:.2f}, intercept={p[1]:.2f}')
    
        p_fitted_line_NN = p.copy()  # Store parameters for NN loss fitted line
        raise ValueError(f"Fitted line parameters for NN loss vertices: slope={p_fitted_line_NN[0]}, intercept={p_fitted_line_NN[1]}")
        # create a function that takes a D value and 
        # returns the K, B value on the fitted line
        def fitted_line(D):

            D = D/token_unit  # Convert D to tokens
            """
            Given a D value, return the K value on the fitted line
            """
            # Use the fitted line equation: y = mx + b
            # where y is D, x is K, m is slope, b is intercept
            # Rearranging gives us: K = (D - b) / m
            K_raw = 10**((np.log10(D) - p_fitted_line_NN[1]) / p_fitted_line_NN[0])
            return K_raw
        
        # use the fitted line to compute K values for a range of D values
        D_values = np.linspace(min(dat['D']), max(dat['D']), 100)
        K_values = fitted_line(D_values)
        tst_D = 1e9
        tst_K = fitted_line(tst_D)
        #raise ValueError(f"Test D: {tst_D}, Test K: {tst_K}. Please check the fitted line function.")
        # Plot the fitted line for D values
        ax.scatter(K_values, D_values/token_unit, color='black', marker='x', s=50,
                 edgecolor='black', alpha=0.5, linewidth=1.5, label='Fitted Line for D Values')

    list_of_vertices = []  # Reset for NQS loss vertices
    # Plot nqs_loss curves (use triangle markers and dotted lines)
    for loss_value in loss_values:
        subset = iso_loss_df_nqs[iso_loss_df_nqs['nqs_loss'] == loss_value]
        if len(subset) > 0:
            color = cmapp(loss_values.index(loss_value) / len(loss_values))
            
            # Plot dotted line only (no markers)
           # ax.scatter(subset['K'], subset['D'], label=f'NQS Loss = {loss_value}',
           #             alpha=0.8, marker='^', s=50, color=color)
            
            if len(subset) >= 3:
                K_fitted, D_fitted, params = fit_hyperbola_least_squares(subset['K'].values, subset['D'].values)
                if K_fitted is not None and D_fitted is not None and params is not None:
                  #  ax.plot(K_fitted, D_fitted, linestyle=':', alpha=0.7, 
                  #          color=color, linewidth=2)
                    
                    S_min, E_min = params #h_fit, k_fit, m_fit, a_fit = params
                    
                    # Compute and plot vertex (intersection of transverse axis with hyperbola)
                    if True:
                        x_vertex, y_vertex, s = compute_vertex_intersection(S_min, E_min)
                        list_of_vertices.append((x_vertex, y_vertex))
                        ax.scatter(x_vertex, y_vertex, color=color, marker='^', s=150, 
                                 edgecolor='black', alpha=0.3, linewidth=2)
                        
                        # Store parameters for transverse axis
                        x_axis = np.array([min(iso_loss_df_nqs['K']), max(iso_loss_df_nqs['K'])])
                        y_axis = ((x_axis/S_min - 1)**(-1) +1)* E_min 
                        # Plot the straight line
                       # ax.plot(x_axis, y_axis, color=color, linestyle='-.', 
                       #        alpha=0.5, linewidth=1.5, label=f'Transverse Axis' if loss_value == loss_values[0] else "")
                    # Fallback to center
                    #    ax.scatter(h_fit, k_fit, color=color, marker='+', s=120,
                    #             edgecolor='black', alpha=0.9, linewidth=2)
                    # Store parameters for transverse axis
                    # transverse_axes.append((h_fit, k_fit, m_fit, a_fit))
            else:
                # Connect points with simple line for insufficient data
                sorted_indices = np.argsort(subset['K'])
                ax.plot(subset['K'].iloc[sorted_indices], subset['D'].iloc[sorted_indices], 
                        linestyle=':', alpha=0.7, color=color)
    # Fit a line to the vertices for NQS loss
    if len(list_of_vertices) >= 2:
        vertices = np.array(list_of_vertices)
        # Fit a line to the vertices, log-log scale
        p = np.polyfit(np.log10(vertices[:, 0]), np.log10(vertices[:, 1]), 1)
        x_fit = np.linspace(min(iso_loss_df_nqs['K']), max(iso_loss_df_nqs['K']), 200)
        y_fit = 10**(p[0] * np.log10(x_fit) + p[1])
        # define color based on model size

        ax.plot(x_fit, y_fit, 
                # for color use model size cmap
                
                color=get_N_color(iso_loss_df_nqs['N'].unique()[0]), 
                linestyle=':', alpha=0.5, linewidth=1.5,   
                label='Fitted, NQS Loss Vertices, N=' + str(iso_loss_df_nqs['N'].unique()[0]))
    # Add legend for loss values
    # Add legend for transverse axes
    #ax.legend(title='Transverse Axis', loc='upper right', fontsize='small')
        # save x_fit, y_fit to a dataframe
        df_save_xy = pd.DataFrame({'K': x_fit, 'D': y_fit})


    
    # Format the axis
    ax.set_xlabel('Total Optimization Steps (K)')
    ax.set_ylabel('Total Data Budget (D)')
    ax.set_xlim(1e1, 1e6)
    ax.set_ylim(1e4, 1e7)
    ax.set_title('Combined Iso-Loss Curves (NN vs NQS)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # return the function that computes K from D
    tst_D = 15000000/6/1e7*1e9
    tst_K = fitted_line(tst_D)
    tst_B = tst_D / (tst_K * seq_len)
    #raise ValueError(f"Test D: {tst_D}, Test K: {tst_K}, Test B: {tst_B}. Please check the fitted line function.")

    return fitted_line

def plot_loss_vs_D_combined(dat, figname):

    dat0 = dat.copy()
    """
    Create a 3x2 figure with NN_loss in first row, nqs_loss in second row,
    and combined iso-loss curves in bottom right
    """
    # Create the 3x2 subplot layout
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # get unique model sizes in dat0
    unique_N = dat0['N'].unique()
    # raise error with unique model sizes
    #raise ValueError(f"unique model sizes in dat0: {unique_N}. Please filter the dataset to a single model size before plotting.")
    # loop through each unique model size and plot
    # start a dictionary of fitted lines
    fitted_lines = {}
    # Loop through each unique model size N
    for N in unique_N:
        dat = dat0[dat0['N'] == N]
        if len(dat) < 3:
            print(f"Skipping model size N={N} due to insufficient data")
            continue
        
        # Print the number of samples for this model size
        print(f"Plotting for model size N={N}, samples: {len(dat)}")
        # Create combined iso-loss plot on the right
        iso_loss_df_nn = plot_loss_vs_D_subplot(dat, 'NN_loss', axes[0, 0], axes[0, 1])
        iso_loss_df_nqs = plot_loss_vs_D_subplot(dat, 'nqs_loss', axes[1, 0], axes[1, 1])
        fitted_line = plot_combined_iso_loss(iso_loss_df_nn, iso_loss_df_nqs, axes[2, 1])
        #tst_D = 1e9
        #tst_K = fitted_line(tst_D)
        #raise ValueError(f"Test D: {tst_D}, Test K: {tst_K}. Please check the fitted line function.")
        # Store the fitted line function for this model size
        fitted_lines[N] = fitted_line

    
    # Third row: Combined plot (right side only)
    # Hide the left subplot in the third row
    axes[2, 0].set_visible(False)
    
    
    # Adjust layout and add overall title
    plt.tight_layout()
    plt.suptitle('Loss Analysis: NN vs NQS Loss Comparison with Vertices', fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()

    return fitted_lines


def get_actual_N(proposed_N):
    nn_dict ={"data": "opengenome2_local", "loss": "condcrossent", "model": "striped_hyena"}
    h_dict = {"N": proposed_N}
    path = './outputs/temp_iso_loss_4/'

    actual_N = build_hf_jax(nn_dict, h_dict, path)
    return actual_N

def edit_h_samples(h_samples, fitted_line):
    # edit h_samples so that B,K are on a fitted line
    # first compute D using the formula D = B * K * seq_len
    # then reallocate D so that D, K is on the fitted line
    # check if there is a saved file
    if os.path.exists('h_samples_with_actual_N.csv'):
        h_samples = pd.read_csv('h_samples_with_actual_N.csv')
    else:
        h_samples['actual_N'] = h_samples['N'].apply(get_actual_N)
    # save the file at this point
    h_samples.to_csv('h_samples_with_actual_N.csv', index=False)

    print(h_samples)
    print(h_samples['N'])
    # do in two steps to avoid overflow
    h_samples['D_preN'] = h_samples['B'] * h_samples['K']  * 128/h_samples['actual_N']
    h_samples['D_wN'] = h_samples['D_preN']*h_samples['N']
    h_samples['D'] =  h_samples['D_wN'] ##h_samples['B'] * h_samples['K']  * seq_len * h_samples['N'] / h_samples['actual_N']

    # if D is not finite, raise an error with the values of N, B, K, seq_len, actual_N
    if not np.all(np.isfinite(h_samples['D'])):
        invalid_rows = h_samples[~np.isfinite(h_samples['D'])]
        raise ValueError(f"Non-finite D values found in rows:\n{invalid_rows[['N', 'actual_N', 'B', 'K']]}")
    # drop D_preN and D_wN
    h_samples = h_samples.drop(columns=['D_preN', 'D_wN'])
    # now compute K using the fitted line
    # print D_preN, D_wN
    #print(h_samples[['D_preN', 'D_wN']])
   #raise ValueError(h_samples[['D']])
    
    h_samples['K'] = h_samples['D'].apply(fitted_line)
    # save another debug file
    h_samples.to_csv('h_samples_after_fitted_line.csv', index=False)
    if not np.all(np.isfinite(h_samples['K'])):
        invalid_rows = h_samples[~np.isfinite(h_samples['K'])]
        raise ValueError(f"Non-finite K values found in rows after applying fitted line:\n{invalid_rows[['N', 'actual_N', 'B', 'D']]}")
    #test_D = 1e9
    #test_K = fitted_line(test_D)
    #raise ValueError(f"Test D: {test_D}, Test K: {test_K}. Please check the fitted line function.")
    # now compute B using the formula B = D / (K * seq_len)
    h_samples['B'] = h_samples['D'] / (h_samples['K'] * seq_len)
    # round B to the nearest integer
    h_samples['B'] = h_samples['B'].round().astype(int)
    # recompute K using the new B
    h_samples['K'] = h_samples['D'] / (h_samples['B'] * seq_len)
    # round K to the nearest integer
    h_samples['K'] = h_samples['K'].round().astype(int)
    # return the edited h_samples
    return h_samples



# Run the combined plot
if __name__ == "__main__":
    

    #dat1 = pd.read_csv('outputs/runs/2025-06-20-19-58_lm1b_pythia_300iters/6_critical_batch_size/h_samples_with_nn_loss.csv')
    #dat2 = pd.read_csv('outputs/runs/2025-06-23-13-56_lm1b_pythia_300iters_small/6_critical_batch_size/h_samples_with_nn_loss.csv')
    #dat3 = pd.read_csv('outputs/runs/2025-06-23-14-01_lm1b_pythia_300iters_large/6_critical_batch_size/h_samples_with_nn_loss.csv')

    # concatenate the dataframes
   # dat = pd.concat([dat1, dat2, dat3], ignore_index=True)
    # save the concatenated dataframe to a new csv file
   # dat.to_csv('h_samples_with_nn_loss_combined.csv', index=False)
    dat = pd.read_csv('outputs/runs/LRA_investigation/2026-01-17-22-29_opengenome2_local_striped_hyena_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv')
    # scale_new/outputs/runs/LRA_investigation/2026-01-17-16-12_opengenome2_local_striped_hyena_adam_cosine/6_critical_batch_size/h_samples_with_nn_loss.csv
        #"outputs/runs/2025-08-06-22-57_openwebtext2_pythia/4_loss_estimation/eval_df.csv")

    # document owt sgd critical batch size data: pd.read_csv("outputs/runs/2025-08-06-22-57_openwebtext2_pythia/4_loss_estimation/eval_df.csv")
    # 'outputs/runs/2025-07-23-20-05_lm1b_pythia/4_loss_estimation/eval_df.csv')
    #pd.read_csv('outputs/runs/2025-06-20-19-58_lm1b_pythia_300iters/4_loss_estimation/eval_df.csv')
    
    N_0 = 50000000 # This is the model size we are interested in
    N_1 = 50000000
    dat = dat[dat['N'].isin([N_0, N_1])]  # Filter for the specific model size
    # filter for train split
    # check if 'h_samples_type' column exists
    if 'h_samples_type' in dat.columns:
        dat = dat[dat['h_samples_type'] == 'test']
    # dat['NN_loss'] = dat['loss']
    # check if 'loss' column exist, if not check for 'NN_loss' column
    if 'loss' not in dat.columns:
        if 'NN_loss' not in dat.columns:
            raise ValueError("The dataset does not contain 'loss' or 'NN_loss' column.")
        else:
            dat['loss'] = dat['NN_loss']
    # filter for where loss is not missing
    dat = dat.dropna(subset=['loss'])
    # filter for where loss is < 1.5
    dat = dat[dat['loss'] < 1.5]
    dat['D'] = seq_len * dat['K'] * dat['B']  # Compute D from K and B

    # remove rows with NaN values in 'NN_loss' or 'nqs_loss'

    # ONLY make plots for nn
    # set nqs_loss = NN_loss
    dat['nqs_loss'] = dat['NN_loss']
    dat = dat.dropna(subset=['NN_loss', 'nqs_loss'])
    

    print(dat.head())
    # Remove the filter that limited to single model size
    #raise ValueError("Please filter the dataset to a single model size before plotting.")


    # define a color map for the model sizes
    N_unique = dat['N'].unique()
    N_unique_sorted = np.sort(N_unique)

    token_unit = 1000

    # Call the plotting function
    figure_name = 'loss_vs_D_combined_genome_adamw_cosine.png'
    # save the data used for plotting
    icsv_name = figure_name.replace('.png', '_data.csv')
    dat.to_csv(icsv_name, index=False)
    print(f"Data used for plotting saved to {icsv_name}")
    fitted_lines = plot_loss_vs_D_combined(dat, figure_name)
    print(f"Fitted lines for model sizes: {list(fitted_lines.keys())}")

    fitted_line = fitted_lines[N_0]  # Get the fitted line function for the specific model size N_0
    print(f"Fitted line function for model size {N_0} obtained.")
    print(f"Fitted line function: {fitted_line}")

    edit_samples = True # Set to True to edit h_samples, False to skip editing
    if edit_samples:
        # load h_samples
        # inputs/h_samples_resource_allocation_nn.csv
        file_name = 'inputs/h_samples_resource_allocation_all_balanced_adamw_cosine_test_extended.csv'
        #'inputs/h_samples_test_nqs_owt_in_sample_adamw_isoflop_slice_square_single.csv'
        #'inputs/h_samples_critical_batch_size_adamw_small_w_valid_and_test_owt_extended_single.csv'
        #'inputs/h_samples_resource_allocation_owt_small_w_valid_and_double_test_adamw_refined.csv'
        #'inputs/h_samples_resource_allocation_owt_very_very_large_adamw.csv'
        #'inputs/h_samples_resource_allocation_owt_all_adamw.csv'
        #'inputs/h_samples_resource_allocation_owt_large.csv'
        #'h_samples_resource_allocation_adamw_small.csv'
        h_samples = pd.read_csv(file_name)
        # edit h_samples to be on the fitted line
        h_samples_edited = edit_h_samples(h_samples, fitted_line)
        # save the edited h_samples to a new csv file
        # new name is file_name + on_critical_line.csv
        new_file_name = file_name.replace('.csv', '_on_critical_line_tst_Jan_17.csv')
        h_samples_edited.to_csv(new_file_name, index=False)
        print(f"Edited h_samples saved to {new_file_name}")
        # print the first 5 rows of the edited h_samples
        print(h_samples_edited.head())