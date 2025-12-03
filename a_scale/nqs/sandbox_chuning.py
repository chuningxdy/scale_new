import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp
import nqs  # Import the provided NQS module
import time
from functools import partial

# Set up JAX precision
jax.config.update("jax_enable_x64", True)

def calculate_risk(N, K, B, nqs_params, folds):
    """Calculate risk for given parameters without JIT."""
    return nqs.risk(N, K, B, nqs_params, folds)

def generate_risk_contours(N_values, K_values, B, nqs_params, folds):
    """Generate risk contours over a grid of N and K values with consistent indexing."""
    # Create standard meshgrid with 'xy' indexing
    # N_mesh varies along columns (j), K_mesh varies along rows (i)
    N_mesh, K_mesh = np.meshgrid(N_values, K_values)
    
    # Create an empty list to store risks in row-major order
    risk_list = []
    
    print("Calculating risk contours...")
    start_time = time.time()
    
    # Loop through K (rows) then N (columns) for consistency with row-major ordering
    for i in range(len(K_values)):  # rows (K)
        for j in range(len(N_values)):  # columns (N)
            # With 'xy' indexing: N_mesh[i,j] = N_values[j], K_mesh[i,j] = K_values[i]
            N_val = N_values[j]
            K_val = K_values[i]
            
            # Calculate risk and append to flat list
            risk_val = calculate_risk(N_val, K_val, B, nqs_params, folds)
            risk_list.append(risk_val)
            print(f"N={N_val:.1e}, K={K_val:.1e}, B={B}, Risk={risk_val:.6f}")
    
    # Reshape the risk list to match the meshgrid dimensions
    # With 'xy' indexing: shape should be (len(K_values), len(N_values))
    risks = np.array(risk_list).reshape(len(K_values), len(N_values))
    
    print(f"Risk calculation took {time.time() - start_time:.2f} seconds")
    return N_mesh, K_mesh, risks

def create_training_data(N_train, K_train, B_values, ground_truth_nqs, folds):
    """Create training data from combinations of N, K, and B values."""
    NKBfoldsarrs = []
    lossarr = []
    
    print("Creating training data...")
    for N in N_train:
        for K in K_train:
            for B in B_values:
                NKBfoldsarrs.append((N, K, B, folds))
                loss = calculate_risk(N, K, B, ground_truth_nqs, folds)
                lossarr.append(loss)
                print(f"Training point: N={N:.1e}, K={K:.1e}, B={B}, Loss={loss:.6f}")
                
    return NKBfoldsarrs, lossarr

def calculate_param_contours_normalized(param_pairs, ground_truth_nqs, NKBfoldsarrs, lossarr, num_points=20):
    """Calculate loss contours using normalized parameters (to_x space)."""
    print("Calculating parameter contours with normalized parameters...")
    start_time = time.time()
    
    # Convert ground truth to normalized space
    gt_x = nqs.to_x(ground_truth_nqs)
    
    contour_data = []
    
    # Parameter index mapping
    param_to_index = {'a': 0, 'b': 1, 'ma': 2, 'mb': 3, 'c': 4, 'sigma': 5}
    
    # Parameter ranges as specified
    param_ranges = {
        'a': (1.3, 1.9),
        'b': (0.8, 1.3),
        'ma': (0.3, 0.7),  # log_1e5
        'mb': (0.05, 0.35),  # log_1e5
        'c': (-0.2, 0.15),  # log_1e5
        'sigma': (-0.2, 0.15)  # log_1e5
    }
    
    for param1_name, param2_name in param_pairs:
        # Get indices for these parameters in the normalized space
        param1_idx = param_to_index[param1_name]
        param2_idx = param_to_index[param2_name]
        
        # Use specified parameter ranges
        param1_range = np.linspace(param_ranges[param1_name][0], param_ranges[param1_name][1], num_points)
        param2_range = np.linspace(param_ranges[param2_name][0], param_ranges[param2_name][1], num_points)
        
        # Create standard meshgrid with 'xy' indexing
        param1_mesh, param2_mesh = np.meshgrid(param1_range, param2_range)
        
        # Initialize empty list for NQS objects
        nqs_list = []
        
        # Loop through param2 (rows) then param1 (columns) for consistency
        for i in range(len(param2_range)):  # rows (param2)
            for j in range(len(param1_range)):  # columns (param1)
                # With 'xy' indexing:
                # param1_mesh[i,j] = param1_range[j], param2_mesh[i,j] = param2_range[i]
                param1_norm_val = param1_range[j]
                param2_norm_val = param2_range[i]
                
                # Start with ground truth in normalized space
                x = np.copy(gt_x)  # Use numpy copy instead of JAX's .at
                
                # Update the two parameters we're varying
                x[param1_idx] = param1_norm_val
                x[param2_idx] = param2_norm_val
                
                # Convert back to NQS space
                nqs_obj = nqs.to_nqs(x)
                nqs_list.append(nqs_obj)
        
        # Calculate all losses at once
        loss_values = nqs.compute_loss_multiple(NKBfoldsarrs, lossarr, nqs_list)
        
        # Reshape to match the meshgrid dimensions
        loss_grid = np.array(loss_values).reshape(len(param2_range), len(param1_range))
        
        # Store data with normalized parameter meshes
        contour_data.append({
            'param1_name': param1_name,
            'param2_name': param2_name,
            'param1_idx': param1_idx,
            'param2_idx': param2_idx,
            'param1_mesh': param1_mesh,
            'param2_mesh': param2_mesh,
            'losses': loss_grid
        })
    
    print(f"Parameter contour calculation took {time.time() - start_time:.2f} seconds")
    return contour_data

def plot_risk_contour(ax, N_mesh, K_mesh, risks, B, N_lb, N_ub, K_lb, K_ub, training_data=None, title_prefix=""):
    """Plot risk contours with training region."""
   # raise ValueError("N_lb: " + str(N_lb) + " N_ub: " + str(N_ub) + " K_lb: " + str(K_lb) + " K_ub: " + str(K_ub))
    # With 'xy' indexing and proper reshaping, we can directly use these arrays
    cp = ax.contourf(np.log10(N_mesh), np.log10(K_mesh), np.log10(risks), 
                  levels=20, cmap='viridis')
    ax.set_title(f'{title_prefix}Risk Contours (log10) for B={B}')
    ax.set_xlabel('log10(N)')
    ax.set_ylabel('log10(K)')
    cbar = plt.colorbar(cp, ax=ax, label='log10(Risk)')
    
    # Mark the training region with a box
    box_x = [np.log10(N_lb), np.log10(N_ub), np.log10(N_ub), np.log10(N_lb), np.log10(N_lb)]
    box_y = [np.log10(K_lb), np.log10(K_lb), np.log10(K_ub), np.log10(K_ub), np.log10(K_lb)]
    ax.plot(box_x, box_y, 'r--', linewidth=2, label='Training Region')
    
    # Mark the training points if provided
    if training_data is not None:
        for N, K, B_val, _ in training_data:
            if B_val == B:  # Only plot points with the matching B value
                ax.plot(np.log10(N), np.log10(K), 'rx', markersize=8)
    
    ax.legend()
    return cbar

def get_trajectory_index_map(trajectories, init_nqs_list):
    """Map trajectory indices to their corresponding initialization indices."""
    # Create a mapping from initialization index to trajectory index
    index_map = {}
    
    # For each trajectory, find the matching initialization
    for traj_idx, traj_dict in enumerate(trajectories):
        init_idx = traj_dict.get('initialization_index', None)
        if init_idx is not None:
            index_map[init_idx] = traj_idx
    
    # If mapping is incomplete, try to match based on first trajectory point
    if len(index_map) < len(trajectories):
        for traj_idx, traj_dict in enumerate(trajectories):
            if traj_idx not in index_map.values():
                traj = traj_dict['trajectory']
                first_point = traj[0] if traj else None
                
                if first_point:
                    for init_idx, init_nqs in enumerate(init_nqs_list):
                        if init_idx not in index_map:
                            # Check if this initialization matches the first trajectory point
                            if (abs(init_nqs.a - first_point.a) < 1e-5 and
                                abs(init_nqs.b - first_point.b) < 1e-5):
                                index_map[init_idx] = traj_idx
                                break
    
    # Create a reverse mapping from traj_idx to init_idx
    traj_to_init = {traj_idx: init_idx for init_idx, traj_idx in index_map.items()}
    
    # For any remaining unmatched trajectories, assign them sequentially to unmatched initializations
    unmatched_trajs = [i for i in range(len(trajectories)) if i not in traj_to_init]
    unmatched_inits = [i for i in range(len(init_nqs_list)) if i not in index_map]
    
    for traj_idx, init_idx in zip(unmatched_trajs, unmatched_inits):
        traj_to_init[traj_idx] = init_idx
    
    return traj_to_init

def get_top_trajectory_indices(trajectories, n=3):
    """Get indices of the top n trajectories with lowest final loss (excluding NaN)."""
    # Extract final losses
    final_losses = []
    for traj_dict in trajectories:
        loss = traj_dict.get('loss', float('inf'))
        final_losses.append(loss)
    
    # Convert to numpy array for nanargmin
    final_losses = np.array(final_losses)
    
    # Find indices of top n non-NaN losses
    top_indices = []
    for _ in range(n):
        if np.all(np.isnan(final_losses)):
            break
        
        best_idx = np.nanargmin(final_losses)
        if np.isnan(final_losses[best_idx]):
            break
            
        top_indices.append(best_idx)
        final_losses[best_idx] = np.nan  # Mark as processed
    
    return top_indices

def plot_parameter_contour_normalized(ax, data, trajectories, init_nqs_list, fitted_nqs, ground_truth_nqs):
    """Plot parameter contour with trajectories and key points in normalized parameter space."""
    param1_name = data['param1_name']
    param2_name = data['param2_name']
    param1_idx = data['param1_idx']
    param2_idx = data['param2_idx']
    param1_mesh = data['param1_mesh']
    param2_mesh = data['param2_mesh']
    losses = data['losses']
    
    # Take the log for better visualization
    log_losses = np.log10(losses)
    
    # Create contour plot
    cp = ax.contourf(param1_mesh, param2_mesh, log_losses, levels=20, cmap='cividis')
    ax.set_title(f'Log10 Loss Contour: {param2_name} vs {param1_name} (Normalized)')
    ax.set_xlabel(f'{param1_name} (normalized)')
    ax.set_ylabel(f'{param2_name} (normalized)')
    cbar = plt.colorbar(cp, ax=ax, label='Log10 Loss')
    
    # Check if fitted_nqs contains NaN values
    has_nan = isinstance(fitted_nqs.a, np.ndarray) and np.isnan(fitted_nqs.a)
    
    if not has_nan and trajectories:
        # Generate consistent colors for initializations
        init_colors = plt.cm.tab10(np.linspace(0, 1, len(init_nqs_list)))
        
        # Get mapping from trajectory index to initialization index
        traj_to_init = get_trajectory_index_map(trajectories, init_nqs_list)
        
        # Get top 3 trajectories (lowest final loss, excluding NaN)
        top_traj_indices = get_top_trajectory_indices(trajectories, n=3)
        
        # Convert all points to normalized space for plotting
        gt_x = nqs.to_x(ground_truth_nqs)
        fitted_x = nqs.to_x(fitted_nqs)
        init_x_list = [nqs.to_x(init_nqs) for init_nqs in init_nqs_list]
        
        # Plot only top 3 trajectories
        for i, traj_dict in enumerate(trajectories):
            # Skip if not in top 3
            if i not in top_traj_indices:
                continue
                
            # Get color based on the initialization that this trajectory corresponds to
            init_idx = traj_to_init.get(i, i % len(init_colors))
            color = init_colors[init_idx]
            
            traj = traj_dict['trajectory']
            # Convert each point in trajectory to normalized space
            traj_x = [nqs.to_x(t) for t in traj]
            # Extract the parameters of interest
            param1_traj = [x[param1_idx] for x in traj_x]
            param2_traj = [x[param2_idx] for x in traj_x]
            
            ax.plot(param1_traj, param2_traj, '-', color=color, linewidth=1.5, alpha=0.8, 
                   label=f'Traj {init_idx+1}')
        
        # Mark all initialization points
        legend_added = []  # Track which top trajectories we've added to the legend
        for i, x in enumerate(init_x_list):
            param1_val = x[param1_idx]
            param2_val = x[param2_idx]
            
            # Check if this initialization corresponds to a top trajectory
            is_top = False
            for traj_idx in top_traj_indices:
                if traj_to_init.get(traj_idx) == i:
                    is_top = True
                    break
            
            # Use different styling based on whether this is a top trajectory
            if is_top:
                # This is a top trajectory - use normal marker
                marker_size = 10
                alpha = 1.0
                # Only add to legend if not already added
                label = f'Init {i+1}' if i not in legend_added else None
                if label:
                    legend_added.append(i)
            else:
                # Not a top trajectory - use smaller marker with transparency
                marker_size = 3
                alpha = 0.15
                label = None  # No legend entry
                
            ax.plot(param1_val, param2_val, 'o', color=init_colors[i], 
                   markersize=marker_size, alpha=alpha, label=label)
            
            # Add text label only for top trajectories
            if is_top:
                ax.text(param1_val, param2_val, f'Init {i+1}', fontsize=9, ha='right')
        
        # Mark fitted point in normalized space
        param1_val_fitted = fitted_x[param1_idx]
        param2_val_fitted = fitted_x[param2_idx]
        ax.plot(param1_val_fitted, param2_val_fitted, 'r*', markersize=15, label='Fitted')
        ax.text(param1_val_fitted, param2_val_fitted, 'Fitted', fontsize=9, ha='right')
    
    # Mark ground truth in normalized space
    param1_val_gt = gt_x[param1_idx]
    param2_val_gt = gt_x[param2_idx]
    ax.plot(param1_val_gt, param2_val_gt, 'kx', markersize=15, label='Ground Truth')
    ax.text(param1_val_gt, param2_val_gt, 'Ground Truth', fontsize=9, ha='right')
    
    # Create a cleaner legend with only important entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
    
    return cbar

def plot_parameter_convergence(ax, trajectories, ground_truth_nqs, param_names, init_nqs_list=None):
    """Plot parameter convergence over training steps."""
    # Check if we have valid trajectories
    if not trajectories or 'trajectory' not in trajectories[0]:
        ax.text(0.5, 0.5, "No valid trajectory data", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        return
    
    steps = np.arange(len(trajectories[0]['trajectory']))
    
    # Generate consistent colors based on init_nqs_list
    if init_nqs_list:
        init_colors = plt.cm.tab10(np.linspace(0, 1, len(init_nqs_list)))
        # Get mapping from trajectory index to initialization index
        traj_to_init = get_trajectory_index_map(trajectories, init_nqs_list)
    else:
        init_colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        traj_to_init = {i: i for i in range(len(trajectories))}
    
    # Get top 3 trajectories (lowest final loss, excluding NaN)
    top_traj_indices = get_top_trajectory_indices(trajectories, n=3)
    
    # Plot only top 3 trajectories
    for i, traj_dict in enumerate(trajectories):
        # Skip if not in top 3
        if i not in top_traj_indices:
            continue
            
        # Get color based on the initialization that this trajectory corresponds to
        init_idx = traj_to_init.get(i, i % len(init_colors))
        color = init_colors[init_idx]
        
        traj = traj_dict['trajectory']
        for j, param in enumerate(param_names):
            linestyle = '-' if j == 0 else '--'
            ax.plot(steps, [getattr(t, param) for t in traj], linestyle, color=color, 
                   label=f'{param} (Init {init_idx+1})')
    
    # Add ground truth reference lines
    for j, param in enumerate(param_names):
        linestyle = '-' if j == 0 else '--'
        ax.axhline(y=getattr(ground_truth_nqs, param), color='k', linestyle=linestyle, 
                  alpha=0.5, label=f'GT {param}')
    
    ax.set_title(f"Parameter Convergence: {', '.join(param_names)}")
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Parameter Value')
    
    # Create a cleaner legend by removing duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

def latin_hypercube_initializations(num_inits, param_ranges, ground_truth_nqs):
    """Create initialization points using Latin Hypercube Sampling."""
    # Parameter names in order
    param_names = ['a', 'b', 'ma', 'mb', 'c', 'sigma']
    
    # Generate Latin Hypercube samples in [0, 1] range
    samples = lhs(len(param_names), samples=num_inits)
    
    # Scale samples to the parameter ranges
    init_nqs_list = []
    for i in range(num_inits):
        x_normalized = np.copy(nqs.to_x(ground_truth_nqs))
        
        for j, param_name in enumerate(param_names):
            # Get parameter range
            lower, upper = param_ranges[param_name]
            # Scale from [0, 1] to [lower, upper]
            x_normalized[j] = lower + samples[i, j] * (upper - lower)
        
        # Convert to NQS space
        init_nqs = nqs.to_nqs(x_normalized)
        init_nqs_list.append(init_nqs)
        
    return init_nqs_list

def create_nqs_test_visualization():
    # Define the ground truth NQS
    ground_truth_nqs = nqs.NQS(a=1.5, b=0.95, ma=100, mb=10, c=1.5, sigma=1)
    print("Ground Truth NQS:", ground_truth_nqs)
    print("Ground Truth NQS (normalized):", nqs.to_x(ground_truth_nqs))

    # Define the SGD optimizer
    folds = nqs.SGD(lr=1.0, momentum=0.0, nesterov=False, scheduler=None)

    # Define ranges for N and K
    N_u, N_l = 1e7, 1e5
    K_u, K_l = 1e6, 1e4
    
    # Define B values
    B_values = [1, 100]

    # Generate mesh for risk contours
    N_values = np.logspace(np.log10(N_l), np.log10(N_u), 5)
    K_values = np.logspace(np.log10(K_l), np.log10(K_u), 5)
    
    # Generate risk contours for GT NQS with B=10
    N_mesh_B10, K_mesh_B10, risks_B10 = generate_risk_contours(N_values, K_values, 1, ground_truth_nqs, folds)
    
    # Generate risk contours for GT NQS with B=100
    N_mesh_B100, K_mesh_B100, risks_B100 = generate_risk_contours(N_values, K_values, 100, ground_truth_nqs, folds)

    # Define the training set (subset of the N, K values)
    N_ub, N_lb = 1e7, 1e6
    K_ub, K_lb = 1e5, 1e4

    # Create training data with multiple B values
    N_train = np.logspace(np.log10(N_lb), np.log10(N_ub), 3)
    K_train = np.logspace(np.log10(K_lb), np.log10(K_ub), 3)
    
    NKBfoldsarrs, lossarr = create_training_data(N_train, K_train, B_values, ground_truth_nqs, folds)

    # Define initializations
    num_inits = 1024  # Number of initializations (Latin Hypercube)
    
    # Compute the loss at the ground truth
    loss_gt = nqs.compute_loss_multiple(NKBfoldsarrs, lossarr, [ground_truth_nqs])
    print(f"Ground truth loss: {loss_gt}")
    #raise ValueError("Stop here to check the loss at the ground truth NQS.")
    
    # Parameter ranges
    param_ranges = {
        'a': (1.3, 1.9),
        'b': (0.8, 1.3),
        'ma': (0.3, 0.7),  # log_1e5
        'mb': (0.05, 0.35),  # log_1e5
        'c': (-0.2, 0.15),  # log_1e5
        'sigma': (-0.2, 0.15)  # log_1e5
    }
    
    # Create initializations using Latin Hypercube Sampling
    try:
        init_nqs_list = latin_hypercube_initializations(num_inits, param_ranges, ground_truth_nqs)
        for i, init_nqs in enumerate(init_nqs_list):
            print(f"Initialization {i+1}: {init_nqs}")
            print(f"Initialization {i+1} (normalized): {nqs.to_x(init_nqs)}")
    except ImportError:
        print("pyDOE not available, using regular initialization")
        init_nqs_list = []
        
        # Create varied initializations around ground truth
        gt_x = nqs.to_x(ground_truth_nqs)
        
        # Create initializations based on ranges
        for i in range(num_inits):
            # Create variations in normalized space
            x_normalized = np.array(gt_x)
            # Scale factor for initialization - spread initializations across the parameter space
            factor = i / (num_inits - 1) if num_inits > 1 else 0.5
            
            # Modify each parameter based on ranges
            for idx, param_name in enumerate(['a', 'b', 'ma', 'mb', 'c', 'sigma']):
                lower, upper = param_ranges[param_name]
                # Mix of lower and upper bounds based on factor
                x_normalized[idx] = lower + factor * (upper - lower)
            
            # Convert back to NQS space
            init_nqs = nqs.to_nqs(x_normalized)
            init_nqs_list.append(init_nqs)
            print(f"Initialization {i+1}: {init_nqs}")
            print(f"Initialization {i+1} (normalized): {x_normalized}")
    
    # Fit the model
    print("Fitting NQS model...")
    start_time = time.time()
    nqs_fitting_time = 0.0
    try:
        fitted_nqs, trajectories = nqs.fit_multiple(
            NKBfoldsarrs, lossarr, nqs0_list=init_nqs_list,
            return_traj=True, steps=500, gtol=1e-7
        )
        print(f"Fitting took {time.time() - start_time:.2f} seconds")
        nqs_fitting_time = time.time() - start_time
        print("Fitted NQS:", fitted_nqs)
        print("Fitted NQS (normalized):", nqs.to_x(fitted_nqs))
        
        # Check if fit failed with NaN values
        has_nan = isinstance(fitted_nqs.a, np.ndarray) and np.isnan(fitted_nqs.a)
        if has_nan:
            print("Warning: Fitting produced NaN values. Proceeding without fitted parameters.")
            # Create a placeholder for visualization
            trajectories = []
            
    except Exception as e:
        print(f"Error during fitting: {e}")
        print("Proceeding without fitted parameters.")
        fitted_nqs = ground_truth_nqs  # Use ground truth as a placeholder
        trajectories = []
    
    # Generate risk contours for fitted NQS with B=10, only if fitting was successful
    fitted_risks_B10 = None
    fitted_risks_B100 = None
    
    if trajectories and not (isinstance(fitted_nqs.a, np.ndarray) and np.isnan(fitted_nqs.a)):
        print("Generating risk contours for fitted NQS...")
        N_mesh_fitted_B10, K_mesh_fitted_B10, fitted_risks_B10 = generate_risk_contours(
            N_values, K_values, 1, fitted_nqs, folds)
        
        N_mesh_fitted_B100, K_mesh_fitted_B100, fitted_risks_B100 = generate_risk_contours(
            N_values, K_values, 100, fitted_nqs, folds)
    
    # Define parameter pairs for contour plots
    param_pairs = [
        ('a', 'b'), ('a', 'ma'), ('b', 'ma'), 
        ('b', 'mb'), ('ma', 'mb'), ('c', 'sigma')
    ]
    
    # Calculate parameter contours with normalized parameters
    contour_data = calculate_param_contours_normalized(param_pairs, ground_truth_nqs, NKBfoldsarrs, lossarr)
    
    # Determine figure layout - add 1 more row for fitted NQS contours
    n_rows = 3 + len(param_pairs) // 2  # 4 risk contours + parameter pairs (2 per row)
    if len(param_pairs) % 2 != 0:
        n_rows += 1  # Add an extra row if odd number of parameter pairs
    
    # Setup the figure
    fig = plt.figure(figsize=(12, 4 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig)
    
    # Plot ground truth risk contours
    ax1 = fig.add_subplot(gs[0, 0])
    
    plot_risk_contour(ax1, N_mesh_B10, K_mesh_B10, risks_B10, 1, N_lb, N_ub, K_lb, K_ub, 
                     NKBfoldsarrs, title_prefix="Ground Truth ")
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_risk_contour(ax2, N_mesh_B100, K_mesh_B100, risks_B100, 100, N_lb, N_ub, K_lb, K_ub, 
                     NKBfoldsarrs, title_prefix="Ground Truth ")
    
    # Plot fitted NQS risk contours if available
    if fitted_risks_B10 is not None and fitted_risks_B100 is not None:
        ax3 = fig.add_subplot(gs[1, 0])
        plot_risk_contour(ax3, N_mesh_fitted_B10, K_mesh_fitted_B10, fitted_risks_B10, 10, 
                         N_lb, N_ub, K_lb, K_ub, NKBfoldsarrs, title_prefix="Fitted NQS ")
        
        ax4 = fig.add_subplot(gs[1, 1])
        plot_risk_contour(ax4, N_mesh_fitted_B100, K_mesh_fitted_B100, fitted_risks_B100, 100, 
                         N_lb, N_ub, K_lb, K_ub, NKBfoldsarrs, title_prefix="Fitted NQS ")
    else:
        # If fitted contours unavailable, add placeholders
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.text(0.5, 0.5, "Fitted NQS Risk Contours (B=10)\nNot Available", 
                horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        ax3.set_xticks([])
        ax3.set_yticks([])
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.text(0.5, 0.5, "Fitted NQS Risk Contours (B=100)\nNot Available", 
                horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    # Plot parameter convergence in original space
    ax5 = fig.add_subplot(gs[2, 0])
    plot_parameter_convergence(ax5, trajectories, ground_truth_nqs, ['a', 'b'], init_nqs_list)
    
    ax6 = fig.add_subplot(gs[2, 1])
    plot_parameter_convergence(ax6, trajectories, ground_truth_nqs, ['ma', 'mb'], init_nqs_list)
    
    # Plot parameter contours in normalized space
    for i, data in enumerate(contour_data):
        row = 3 + i // 2  # Start from row 3 since we added the fitted contours
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        plot_parameter_contour_normalized(
            ax, data, trajectories, init_nqs_list, fitted_nqs, ground_truth_nqs
        )
    
    plt.tight_layout()
    plt.savefig('nqs_testing_results.png', dpi=300)
    plt.show()

    # Print a summary if fitting was successful
    if trajectories:
        print("\nSummary of Results:")
        print("===================")
        print(f"Ground Truth NQS: {ground_truth_nqs}")
        print(f"Ground Truth NQS (normalized): {nqs.to_x(ground_truth_nqs)}")
        
        # Get top 3 trajectories
        top_indices = get_top_trajectory_indices(trajectories, n=3)
        traj_to_init = get_trajectory_index_map(trajectories, init_nqs_list)
        
        print("\nTop 3 Trajectories:")
        for i, traj_idx in enumerate(top_indices):
            init_idx = traj_to_init.get(traj_idx, traj_idx)
            loss = trajectories[traj_idx].get('loss', 'N/A')
            print(f"Rank {i+1}: Init {init_idx+1}, Loss = {loss}")
            print(f"  NQS: {init_nqs_list[init_idx]}")
            print(f"  Normalized: {nqs.to_x(init_nqs_list[init_idx])}")
        
        print(f"\nFitted NQS: {fitted_nqs}")
        print(f"Fitted NQS (normalized): {nqs.to_x(fitted_nqs)}")

        # Calculate relative errors
        def rel_error(true, pred):
            return abs(true - pred) / abs(true) * 100

        rel_errors = {
            'a': rel_error(ground_truth_nqs.a, fitted_nqs.a),
            'b': rel_error(ground_truth_nqs.b, fitted_nqs.b),
            'ma': rel_error(ground_truth_nqs.ma, fitted_nqs.ma),
            'mb': rel_error(ground_truth_nqs.mb, fitted_nqs.mb),
            'c': rel_error(ground_truth_nqs.c, fitted_nqs.c),
            'sigma': rel_error(ground_truth_nqs.sigma, fitted_nqs.sigma),
        }

        print("\nRelative Errors (%):")
        for param, error in rel_errors.items():
            print(f"{param}: {error:.2f}%")

    #print("fitting took ", nqs_fitting_time, "seconds")
    # print how many minutes it took to fit and seconds
    print(f"Fitting took {nqs_fitting_time // 60:.0f} minutes and {nqs_fitting_time % 60:.0f} seconds")

if __name__ == "__main__":
    create_nqs_test_visualization()