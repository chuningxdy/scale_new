import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pandas as pd
from omegaconf import OmegaConf
from a_scale.archiving_utils import archive_wrapper, df_row_to_dict
from a_scale.run_nn import train_nn
import json
from matplotlib.colors import LogNorm

def find_max_curvature(x_values, y_values, use_log = True):
    """
    Find the maximum curvature point
    """
    # x, y are lists of numbers
    
    if use_log:
        x = np.log(x_values)
        y = np.log(y_values)
    xy = [(x[i], y[i]) for i in range(len(x))]
    # sort xy by x
    xy.sort(key=lambda p: p[0])
    x = [(p[0]) for p in xy]
    y = [(p[1]) for p in xy]
    slopes = []
    for i in range(len(x)-1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            dydx = dy/dx
            slopes.append(max(0,dydx))
    
    curvatures = []
    for i in range(len(slopes)-1):
        dx = x[i+1] - x[i]
        d_slope = slopes[i+1] - slopes[i]
        curvature = d_slope/dx
        curvatures.append(curvature)
    
    #raise ValueError("curvatures: ", curvatures)

    max_curvature_idx = np.argmax(curvatures)
    max_curvature_x = x_values[max_curvature_idx + 1]
    max_curvature_y = y_values[max_curvature_idx + 1]
    #raise ValueError('slopes are: ' + str(slopes), 'curvatures are: ' + str(curvatures), 'max_index is: ' + str(max_curvature_idx))
    
    # find the first index where curvature is at least 0.03
    for i in range(len(curvatures)):
        if curvatures[i] >= 0.03:
            # if we found such an index, we can use it
            max_curvature_x = x_values[i + 1]
            max_curvature_y = y_values[i + 1]
            break

    return max_curvature_x, max_curvature_y


def IsoX_plot(df_in, x_axis_name, seq_len = 128, models_to_plot = ['nqs'], output_file = None):
    """
    Plot the IsoX plot for the given dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        x_axis_name (str): Name of the column to use for the x-axis.
    """

    def round_significant_figures(x, sig_figs=2):
        """
        Round a number to a specified number of significant figures.
        """
        if x == 0:
            return 0
        else:
            return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)

    # prepare the data
    df = df_in.copy() 
    
    # where actual_N is nan, replace it with the value of N
    df['actual_N'] = df['actual_N'].fillna(df['N'])
    #C = 6 * seq_len * N/1000 * B/1000 * K/1000 (billion flops)
    #df['C'] = 6 * seq_len * df['actual_N'] / 1000 * df['B'] / 1000 * df['K'] / 1000
    # round the values in the C column to 2 significant figures
    #df['C'] = df['C'].apply(lambda x: round_significant_figures(x, 2))
    #D = seq_len * B/1000 * K/1000 (million tokens)
    df['D'] =seq_len * df['B'] / 1000 * df['K'] / 1000
    #N = N/1000000 (million parameters)
    df['N'] = df['actual_N'] / 1000000


    # sort by C
    df = df.sort_values(by='C')


    # look for all the unique C values
    unique_C_values = df['C'].unique()
    
    # create a colormap
    cmap = plt.get_cmap('viridis', len(unique_C_values))
    # create a color map for the C values
    color_map = {val: cmap(i) for i, val in enumerate(unique_C_values)}

    # sort the dataframe by the x_axis_name column
    df = df.sort_values(by=x_axis_name)

    # define plotting parameters
    fig, ax = plt.subplots(figsize=(8, 6))

    # colors
    color_dict = {'nqs':'green', 'nn':'blue'}
    marker_dict = {'nqs':'o', 'nn':'x'}    
    marker_size_dict = {'nqs':3, 'nn':100}
    line_style_dict = {'nqs':'-', 'nn':'--'}
    line_width_dict = {'nqs':2, 'nn':6}
    alpha_dict = {'nqs':0.8, 'nn':0.3}
    star_style_dict = {'nqs':'*', 'nn':'o'}
    triangle_style_dict = {'nqs':'^', 'nn':'v'}
    triangle_size = {'nqs':20, 'nn':20}
    star_size = {'nqs':300, 'nn':30}
    

    if x_axis_name == 'B':
        critical_batch_size_list = []
    first_C = True
    second_C = False
    nqs_critical_values = []
    nn_critical_values = []

    for C in unique_C_values:
        #if not first_C:
        #    z_old = z.copy()
        #    p_old = np.poly1d(p.coeffs)
         #   x_at_min_old = x_at_min.copy()
         #   y_at_min_old = y_at_min.copy()
         #   print(y_at_min_old)
        #raise ValueError("df shape: ", df.shape)
        # print all unique C values in df
        #print("Unique C values in df: ", df['C'].unique())
        #raise ValueError("Check the unique C values in df")
        # filter the dataframe for the current C value
        df_C = df[df['C'] == C]
        #print("df_C shape: ", df_C.shape)
        #raise ValueError("Check the df_C shape")
        # add a scatter plot for the NN_loss
        df_model = df_C.copy()
        df_model = df_C[df_C['NN_loss'].notna()]
        # sort by N, B, K and NN_loss
        df_model = df_model.sort_values(by=['N', 'B', 'K', 'NN_loss'])
        # group by N, B, K and take the first row in each group`
        df_model = df_model.groupby(['N', 'B', 'K']).first().reset_index()
        ax.scatter(df_model[x_axis_name], df_model['NN_loss'],
                   color=color_map[C], 
                   marker=marker_dict['nn'], 
                   s=marker_size_dict['nn'], 
                   alpha=alpha_dict['nn'], 
                   label= 'NN Loss (C = {:.2f})'.format(C))
        # add a line plot for the NN_loss
        ax.plot(df_model[x_axis_name], df_model['NN_loss'],
                color=color_map[C], 
                linestyle=line_style_dict['nn'], 
                linewidth=line_width_dict['nn'], 
                alpha=alpha_dict['nn'], 
                label= 'NN Loss (C = {:.2f})'.format(C))
        
        # get the max curvature point
        x_values = df_model[x_axis_name].values
        y_values = df_model['NN_loss'].values

        if len(x_values) >= 3:
            max_curvature_x, max_curvature_y = find_max_curvature(x_values, y_values)
            ax.plot(max_curvature_x, max_curvature_y, marker=triangle_style_dict['nn'], color=color_map[C], markersize=triangle_size['nn'], alpha = alpha_dict['nn'])
            nn_critical_values.append((C, max_curvature_x, max_curvature_y))


        # fit a quadratic curve to the data for NN_loss, use x-axis on the log scale
        df_model['log_x_axis'] = np.log10(df_model[x_axis_name])

        #z = np.polyfit(df_model['log_x_axis'], df_model['NN_loss'], 2)
        #p = np.poly1d(z)
        #x_fit = np.linspace(df_model['log_x_axis'].min(), df_model['log_x_axis'].max(), 100)
        #y_fit = p(x_fit)
        #if first_C:
        #    label_parab = 'NN Loss Parabolic Fit (C = {:.2f})'.format(C)
        #else:
         #   label_parab = None
        #ax.plot(10**x_fit, y_fit,
        #        color=color_map[C],
        #        linestyle=line_style_dict['nn'],
        #        linewidth=line_width_dict['nn'],
        #        alpha=alpha_dict['nn'],
        #        label=label_parab)
        # find the minimum of the Parabola, and mark it with a vertical line
        #x_at_min = -z[1] / (2 * z[0])
        # make sure the min_x is within the range of x_fit
        #if x_at_min < x_fit.min():
        #    x_at_min = x_fit.min()
        #if x_at_min > x_fit.max():
        #    x_at_min = x_fit.max()
        #x_at_min = 10**x_at_min
        #y_at_min = p(np.log10(x_at_min))
        #print(y_at_min)
        #line_label = None
        #if first_C:
         #   line_label = 'NN Loss Minima (C = {:.2f})'.format(C)
        #if True:
        #    ax.axvline(x=x_at_min, color=color_map[C], linestyle=':', 
        #            linewidth=3, alpha=0.7,
        #                label=line_label)

        if x_axis_name == 'B' and not first_C:
            # add critical batch size (in addition to optimal bs)
            # find the intersection of the current parabola with the previous 
            # y = y_min line
            # i.e. solve for x where p(x) = y_at_min_old
            # use the quadratic formula
            # p(x) = z[0] * x^2 + z[1] * x + z[2] = y_at_min_old
           # x_at_old_ymin = (-z[1] + np.sqrt(z[1]**2 - 4 * z[0] * (z[2] - y_at_min_old))) / (2 * z[0])
           # print("calced_ymin_old: ", p(x_at_old_ymin))
            #old_y_at_min_old = p_old(x_at_old_ymin)
           # x_at_old_ymin = 10**x_at_old_ymin
            # pick the larger of the two solutions
            # place vertical line at x_at_old_ymin, label it as "Critical Batch Size (C = old C)"
           # label = None
            if second_C:
                label = 'Critical Batch Size (C = {:.2f})'.format(C_old)
           # if False:
           #     ax.axvline(x=x_at_old_ymin, color=color_map[C_old], linestyle='--',
           #             linewidth=3, alpha=0.7,
           #             label=label)
            # add a star for the critical batch size, at (x_at_old_ymin, y_at_min_old)
            label = None
            if second_C:
                label = 'Critical Batch Size'
           # ax.scatter(x_at_old_ymin, y_at_min_old,
            #            color=color_map[C_old],
            #            edgecolor='black',
            #            marker=star_style_dict['nn'],
            #            s=star_size['nn'],
             #           label=label)
           # if False:
           #     ax.scatter(x_at_old_ymin, old_y_at_min_old,
           #                 color=color_map[C_old],
           #                 edgecolor='black',
            #                marker=star_style_dict['nn'],
            #                s=star_size['nn'],
            #                label=label)
            # add this tuple to the critical_batch_size_list
           # critical_batch_size_list.append((C_old, x_at_old_ymin, y_at_min_old))
            
            
        for model in models_to_plot:
            model_loss_col = model + '_loss'
            df_model = df_C.copy()
            #raise ValueError("Check the df_model: ", df_model)
            df_model = df_C[df_C[model_loss_col].notna()]
            #raise ValueError("Check the df_model: ", df_model)
            # sort by N, B, K and model_loss
            df_model = df_model.sort_values(by=['N', 'B', 'K', model_loss_col])
            # group by N, B, K and take the first row in each group`
            df_model = df_model.groupby(['N', 'B', 'K']).first().reset_index()

            model_label = None
            if first_C:
                model_label = '{} Loss (C = {:.2f})'.format(model.upper(), C)
            # add a line plot for the model_loss
            ax.plot(df_model[x_axis_name], df_model[model_loss_col],
                    color=color_map[C], 
                    linestyle=line_style_dict[model], 
                    linewidth=line_width_dict[model], 
                    alpha=alpha_dict[model], 
                    # add marker
                    marker=marker_dict[model],
                    markersize=marker_size_dict[model],
                    label=model_label)
            # add a star for the minimum of the model loss
            # reset index to get the correct index for the minimum
            df_model = df_model.reset_index(drop=True)
            #raise ValueError("Check the df_model: ", df_model)
            min_model_loss = df_model[model_loss_col].min()
            min_model_loss_index = df_model[model_loss_col].idxmin()
            min_model_loss_x = df_model[x_axis_name].iloc[min_model_loss_index]

            # find the maximum curvature point and label with a triangle
           
            x_values = df_model[x_axis_name].values
            y_values = df_model[model_loss_col].values
            if len(x_values)>=3:
                max_curvature_x, max_curvature_y = find_max_curvature(x_values, y_values)
                ax.plot(max_curvature_x, max_curvature_y, marker=triangle_style_dict[model], color=color_map[C], markersize=triangle_size[model])
                nqs_critical_values.append((C, max_curvature_x, max_curvature_y))

            model_min_loss_label = None
            if first_C:
                model_min_loss_label = '{} Loss Minima (C = {:.2f})'.format(model.upper(), C)
            print("min_model_loss: ", min_model_loss)
           # if False:
           #     ax.scatter(min_model_loss_x, min_model_loss,
            #                color=color_map[C],
            #                edgecolor='black',
            #                marker=star_style_dict[model],
             #               s=star_size[model],
             #               label=model_min_loss_label)
        
        if first_C:
            second_C = True
            first_C = False
        else:
            second_C = False
        
        C_old = C.copy()



        
    # set the x-axis to be logarithmic
    ax.set_xscale('log')
    #if x_axis_name == 'B': # log log scale for B
    #    ax.set_yscale('log')
    # add legend
    ax.legend(loc='upper right', fontsize=8)

    # add axis labels
    ax.set_xlabel(x_axis_name, fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)

    # set y lim 2.5 to 8 (0.8-1.6)
    #ax.set_ylim([1.2, 1.4])
    ax.set_ylim([2.5, 8])

    if False:
        # set ylim for max to be 3-6
        if x_axis_name == 'B':
            ax.set_ylim([3, 6])

    # save figure
    if output_file is not None:
        fig.savefig(output_file)
        # save a csv file that includes the nqs_critical values
        nqs_critical_df = pd.DataFrame(nqs_critical_values, columns=['C', 'x', 'y'])
        nqs_critical_df.to_csv(output_file.replace('.png', '_nqs_critical_values.csv'), index=False)
        nn_critical_df = pd.DataFrame(nn_critical_values, columns=['C', 'x', 'y'])
        nn_critical_df.to_csv(output_file.replace('.png', '_nn_critical_values.csv'), index=False)
    else:
        fig.savefig("test_IsoX_plot.png")


    return None



def actual_vs_fitted_plot(actual_values, fitted_values, output_file,
                          test_actual_values = None, test_fitted_values = None,
                          train_colors = None, test_colors = None):
            # round all values, if not None, to 3 significant digits
            actual_values = np.round(actual_values, 4)
            fitted_values = np.round(fitted_values, 4)
            if test_actual_values is not None and test_fitted_values is not None:
                test_actual_values = np.round(test_actual_values, 4)
                test_fitted_values = np.round(test_fitted_values, 4)
            
            # settings
            plt.rcParams.update({'font.size': 12})
            # bold font for x and y axis labels
            plt.rcParams.update({'axes.labelweight': 'bold'})
            plt.rcParams.update({'lines.linewidth': 5})
            plt.rcParams.update({'lines.markersize': 8})

            # sort by fitted values
            if train_colors is not None:
                fitted_values, actual_values, train_colors = zip(*sorted(zip(fitted_values, actual_values, train_colors)))
            else:
                fitted_values, actual_values = zip(*sorted(zip(fitted_values, actual_values)))
            plt.figure()
            # make figure larger so that the margins are not cut off
            plt.figure(figsize=(8, 6))
            # if test_actual_values and test_fitted_values are not None
            # then plot the test_actual_values vs. test_fitted_values
            if test_actual_values is not None and test_fitted_values is not None:
                all_fitted_values = list(fitted_values) + list(test_fitted_values)
                plt.plot(all_fitted_values, all_fitted_values, color = "red", alpha = 0.5,
                        label = "perfect fit")
            else:
                plt.plot(fitted_values, fitted_values, color = "red", alpha = 0.5,
                        label = "perfect fit")

            # scatter plot of fitted values vs. actual values
            # x axis: predicted values
            # y axis: actual values
            if train_colors is not None:

                # color by the number of parameters
                plt.scatter(actual_values, fitted_values, label = "training fitted vs. actual; colored by model size", 
                            alpha=0.5, c = train_colors, marker = ".", cmap='rainbow')
            else:
                plt.scatter(actual_values, fitted_values, label = "training fitted vs. actual", 
                            alpha=0.5, color = "blue", marker = ".", cmap='rainbow')
            # if test_actual_values and test_fitted_values are not None
            # then plot the test_actual_values vs. test_fitted_values
            if test_actual_values is not None and test_fitted_values is not None:
                if test_colors is not None:
                    # color by the number of parameters
                    test_fitted_values, test_actual_values, test_colors = zip(*sorted(zip(test_fitted_values, test_actual_values, test_colors)))
                    plt.scatter(test_actual_values, test_fitted_values, label = "test fitted vs. actual; colored by model size", 
                                alpha = 0.5, c = test_colors, marker = "x", cmap='plasma')
                else:
                    # sort by fitted values
                    test_fitted_values, test_actual_values = zip(*sorted(zip(test_fitted_values, test_actual_values)))
                    plt.scatter(test_actual_values, test_fitted_values, label = "test fitted vs. actual", 
                                alpha = 0.5, color = "green", marker= "x", cmap='plasma')


            # log scale
            plt.yscale('log')
            plt.xscale('log')

            # set x, y lim
            min_val = 2.0
            max_val = 7
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)

            plt.xlabel("Actual Values")
            plt.ylabel("Fitted Values")
            plt.subplots_adjust(left=0.15)
            plt.legend()

            plt.savefig(output_file)
            plt.close()

            return None  

def plot_loss_curves(eval_df, output_file):
    
    key_cols = ['h_samples_type','actual_N','N','B','K',
                'lr','end_lr','momentum','lr_schedule','step_decay_schedule','optimizer']
    # group by key_cols and only keep the last row of each group
    eval_df_by_run = eval_df.groupby(key_cols).last().reset_index()
    # save eval_df_by_run as a csv
    eval_df_by_run.to_csv(output_file.replace('.png', '_eval_df_by_run.csv'), index = False)

    # count the number of groups
    num_groups = eval_df_by_run.shape[0]
    num_plots = max(2, num_groups)
    # create a figure with a grid of plots
    fig, axs = plt.subplots(num_plots, 1, figsize=(15, 5*num_groups))
    # loop through each group
    for i in range(num_groups):
        # get the group
        group = eval_df_by_run.iloc[i]
        split = group['h_samples_type']
        # filter the eval_df by the group
        eval_df_i = eval_df[(eval_df[key_cols] == group[key_cols]).all(axis=1)]
        # sort by checkpoint
        eval_df_i = eval_df_i.sort_values('ckpt')

  
        # scatter plot of loss vs ckpt
        axs[i].scatter(eval_df_i['ckpt'], eval_df_i['loss'], label = 'Loss', color = 'blue', marker = "x")
        # scatter the nqs_loss
        axs[i].scatter(eval_df_i['ckpt'], eval_df_i['nqs_loss'], color = 'red', marker = "o")
        axs[i].plot(eval_df_i['ckpt'], eval_df_i['nqs_loss'], label = 'NQS', color = 'red', linestyle = '--')
        # in title, specify split and the group
        axs[i].set_title(f"Split: {split}, Group: {group[key_cols].values}")
        # log scale
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
        # add axis labels
        axs[i].set_xlabel("Checkpoint")
        axs[i].set_ylabel("Loss")
        axs[i].legend()

        # set y limit
        axs[i].set_ylim([3.1, 4.5])

    # save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return None




            
                
    # iterate through the keys,
    # each key has its own series, 
    # each series is to put a marker at each (opt_D, opt_N) and be labeled with the key
    # the shape of the marker is different for each key
    # if the df has a column NN_loss, then color the marker with a color corresponding to the NN_loss
    # using a shared color map





def isoflop_plot(isoflop_samples,
                 seq_len,
                 opts, 
                 output_file,
                 x_axis,
                 to_get_nn_samples = None,
                 get_chin_loss_from_x = None):  
    
    samples = isoflop_samples.copy()
    # compute D and C for each row in samples
    million = 1e6
    billion = 1e9
    samples['D'] = samples['K']/million * seq_len * samples['B'] # D is in million
    #samples['C'] = 6 * samples['D']/billion * samples['N'] # C is in billion
    samples['N'] = samples['N']/million # N is in million
    max_X = samples[x_axis].max()
    min_X = samples[x_axis].min()

    if to_get_nn_samples is not None:
        # get the nn samples
        #nn_samples = to_get_nn_samples
        to_get_nn_samples['D'] = to_get_nn_samples['K']/million * seq_len * to_get_nn_samples['B'] # D is in million
        to_get_nn_samples['N'] = to_get_nn_samples['N']/million # N is in million
        max_X = max(max_X, to_get_nn_samples[x_axis].max())
        min_X = min(min_X, to_get_nn_samples[x_axis].min())

    # shapes dict
    # key: one of chin1, chin2, nqs
    # value: shape of the marker
    shapes = {'chinchilla_1': 'v', 'chinchilla_2': '^', 'nqs': '*', 'NN': 'X'}
    shape_size = {'chinchilla_1': 15, 'chinchilla_2': 15, 'nqs': 18, 'NN': 12}
    shape_colors = {'chinchilla_1': 'orange', 'chinchilla_2': 'orange', 'nqs': 'green', 'NN': 'blue'}
    font_size = 14
    font_weight = 'normal'
    # color scale
    # a function that takes a value and returns a number for color map
    max_val = samples['nqs_loss'].max()
    min_val = samples['nqs_loss'].min()
    norm = LogNorm(vmin=min_val, vmax=max_val)
    cmap = plt.cm.bwr
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    # color map function
    def color_map(val):
        return mappable.to_rgba(val)
    
    # group samples by C
    # for each C, plot scatter plot of nqs_loss vs D
    Cs = samples['C'].unique()


    fig, ax = plt.subplots(figsize=(10, 5))
    first_C_0 = True
    for C in Cs:
        if first_C_0:
            label_0 = "NQS Predicted Loss"
            first_C_0 = False
        else:
            label_0 = None
        df = samples[samples['C'] == C]
        # sort by x_axis
        df = df.sort_values(x_axis)
        ax.scatter(df[x_axis], df['nqs_loss'], 
                   #c = color_map(df['nqs_loss']), 
                   color = "green",
                   alpha = 0.7, s = 5)
        # add a line
        ax.plot(df[x_axis], df['nqs_loss'],
                color = "green", alpha = 0.7, 
                linewidth = 2,
                label = label_0)
        # connect the dots with a line 
        # fit a quadratic curve to the points in log scale, plot the curve
        log_X = np.log(df[x_axis])
        log_nqs_loss = np.log(df['nqs_loss'])
        # fit a quadratic curve
       # p = np.polyfit(log_X, log_nqs_loss, 2)
       # Xs = np.exp(np.linspace(np.log(min_X), np.log(max_X), 100))
       # fit_nqs_loss = np.exp(np.polyval(p, np.log(Xs)))
       # ax.plot(Xs, fit_nqs_loss, 
        #        color = 'black', alpha =0.2, linestyle = '--',
       #         label = 'Quadratic Fitted to NQS Predicted Loss' )

        # fit a quadratic curve
    # add color bar
    #fig.colorbar(mappable, ax=ax)
    # if opts is not None,
    # the opts is a dictionary with keys {'chin1', 'chin2', 'nqs'}
    # the value of each key is a dataframe with columns ['C','opt_N','opt_D'] and potenitally a column 'NN_loss'

    # iterate through the keys,
    # each key has its own series,
    # each series is to put a marker at each (opt_D, NN_loss) and be labeled with the key
    # the shape of the marker is different for each key
    # if the df has a column NN_loss, then color the marker with a color corresponding to the NN_loss
    # using a shared color map

    if opts is not None:
        print("opts keys: ", opts.keys())
        # iterate through the keys in opts
        for key in opts.keys():
            print("key: ", key)
            print("opts[key]: ", opts[key])
        #raise ValueError("Check the opts keys: ", opts.keys())
        for key in opts.keys():
            if key == 'nqs':
                df = opts[key]
                # iterate through the rows of df
                for i in range(df.shape[0]):
                    row = df.iloc[i]
                    # if first time seeing this key, label it with the key
                    if  i == 0:
                        # add a vertical line at row['x_axis]/million
                        #ax.axvline(x = row[x_axis]/million, color = shape_colors[key], linestyle = '--',
                        #        linewidth = 2, alpha = 0.5,
                        #        label = key + 'Optimal')
                        if False:
                            ax.plot(row[x_axis]/million, row['NN_loss'], 
                                    marker = shapes[key],
                                    markersize = shape_size[key],
                                    label = key + 'Optimal; NN_loss is ' + str(round(row['NN_loss'], 2)),
                                    color = "blue", #color_map(row['NN_loss']),
                                    # set edgecolor = 'black'
                                    markeredgewidth = 1, markeredgecolor = 'black',
                                    linestyle = '',
                                    alpha = 0.5
                                    )
                    else:
                        # add a vertical line at row['x_axis]/million
                        #ax.axvline(x = row[x_axis]/million, color = shape_colors[key], linestyle = '--',
                        #            linewidth = 2, alpha = 0.5)
                        if False:
                            ax.plot(row[x_axis]/million, row['NN_loss'], 
                                    marker = shapes[key],
                                    markersize = shape_size[key], 
                                    color = "blue", #color_map(row['NN_loss']),
                                    #color = color_map(row['NN_loss']),
                                    # set edgecolor = 'black'
                                    markeredgewidth = 1, markeredgecolor = shape_colors[key],
                                    linestyle = '',
                                    alpha = 0.5
                                    )
    if to_get_nn_samples is not None:
        first_C_1 = True
        for C in Cs:
            if first_C_1:
                label_1 = "NN Actual Loss"
                first_C_1 = False
            else:
                label_1 = None
            # get the nn samples for this C
            nn_samples_C = to_get_nn_samples[to_get_nn_samples['C'] == C]
            # plot the nn samples
            ax.scatter(nn_samples_C[x_axis], nn_samples_C['NN_loss'], 
                       color = "blue", alpha = 0.9, s = 30,
                       marker = shapes['NN'],
                       label = label_1)
            # add a line
            ax.plot(nn_samples_C[x_axis], nn_samples_C['NN_loss'],
                    color = "blue", alpha = 0.9, 
                    linewidth = 2) #,
                    #label = 'NN Actual Loss')
            # fit a quadratic curve
            log_X = np.log(nn_samples_C[x_axis])
            log_nn_loss = np.log(nn_samples_C['NN_loss'])
            # fit a quadratic curve
            p = np.polyfit(log_X, log_nn_loss, 2)
            Xs = np.exp(np.linspace(np.log(min_X), np.log(max_X), 100))
            fit_nn_loss = np.exp(np.polyval(p, np.log(Xs)))
           # ax.plot(Xs, fit_nn_loss,
            #        color = 'blue', alpha =0.2, linestyle = '--',
            #        label = 'Quadratic Fitted to NN Actual Loss' )
    
    #get_chin_loss_from_x = None
    if get_chin_loss_from_x is not None:
        
        # empty list that collects optimal points
        optimal_points = []
        # iterate through the Cs
        for C in Cs:
            # plot the chin loss against C
            # get_chin_loss_from_x is a function that takes x and C and returns the chin loss
            #x_coords = np.linspace(min_X, max_X, 100)
            # x_coords should be evenly spaced on the log scale
            x_coords = 10**np.linspace(np.log10(min_X), np.log10(max_X), 100)

            chin_loss = np.zeros(x_coords.shape)
            for i in range(x_coords.shape[0]):
                chin_loss[i] = get_chin_loss_from_x(C, x_coords[i] * million)
            # plot the chin loss
            ax.plot(x_coords, chin_loss, color = 'orange', alpha = 0.5, linewidth = 2,
                    label = 'Chinchilla Predicted Loss')
            # save x_coords, chin_loss in a csv file
            chin_loss_df = pd.DataFrame({'x_coords': x_coords, 'chin_loss': chin_loss})
            chin_loss_df.to_csv(output_file.replace('.png', f'_chin_loss_C_{C}.csv'), index = False)
            
            # add a star where the chin loss is minimum
            min_chin_loss_index = np.argmin(chin_loss)
            min_chin_loss_x = x_coords[min_chin_loss_index]
            min_chin_loss_y = chin_loss[min_chin_loss_index]
            ax.scatter(min_chin_loss_x, min_chin_loss_y,
                        color = 'orange', alpha = 0.5, 
                        marker = shapes['chinchilla_1'],
                        s = shape_size['chinchilla_1'])
            # add the point to the optimal points list
            optimal_points.append((C, min_chin_loss_x, min_chin_loss_y))
        # add a line that goes through all the optimal points
        if len(optimal_points) > 1:
            optimal_points = np.array(optimal_points)
            ax.plot(optimal_points[:, 1], optimal_points[:, 2],
                    color = 'orange', alpha = 0.3,
                    linewidth = 3, linestyle = '-',
                    label = 'Chinchilla Optimal Loss')
            # add a vertical line at min_chin_loss
    # add a vertical line at x = 1e6
            

    # add legend, place it outside the plot, at the top right
    # make sure the legend does not overlap with the plot, and fits in the figure
    ax.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
    # set x limit with min and max D
    #ax.set_xlim([min_D, max_D])
    # set y limit with min and max nqs_loss
    #ax.set_ylim([min_val, max_val])
    # add axis labels
    x_label = x_axis + ' (million)'
    ax.set_xlabel(x_label, fontsize = font_size, fontweight = font_weight)
    #ax.set_xlabel('D (million)', fontsize = font_size, fontweight = font_weight)
    ax.set_ylabel('Loss', fontsize = font_size, fontweight = font_weight)
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # set y lim 2.0 to 8
    ax.set_ylim([1.6, 8])
    #ax.set_yscale('log')
    # save the figure
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(output_file)
    plt.close()

    return None
                 


def risk_contour_plot(contour_samples, 
                      seq_len,
                      opts,
                      output_file
                      ):
    
    
    # compute D and C for each row in samples
    samples = contour_samples.copy()
    million = 1e6
    billion = 1e9
    samples['D'] = samples['K']/million * seq_len * samples['B'] # D is in million
    samples['C'] = 6 * samples['D']/billion * samples['N'] # C is in billion
    samples['N'] = samples['N']/million # N is in million
    samples_grid = samples[samples['type'] == 'grid']
    max_D = samples_grid['D'].max()
    min_D = samples_grid['D'].min()
    max_N = samples_grid['N'].max()
    min_N = samples_grid['N'].min()

    # shapes dict
    # key: one of chin1, chin2, nqs
    # value: shape of the marker
    shapes = {'chinchilla_1': 'v', 'chinchilla_2': 'o', 'nqs': '*'}
    shape_size = {'chinchilla_1': 10, 'chinchilla_2': 10, 'nqs': 12}
    font_size = 14
    font_weight = 'normal'
    # color scale
    # a function that takes a value and returns a number for color map
    max_val = samples['nqs_loss'].max()
    min_val = samples['nqs_loss'].min()
    norm = LogNorm(vmin=min_val, vmax=max_val)
    cmap = plt.cm.bwr
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    # color map function
    def color_map(val):
        return mappable.to_rgba(val)


    # samples contain columns ['N','D', 'nqs_loss']
    # use samples to make a figure where the x axis is D, the y axis is N, with contours of nqs_loss (col of samples)
    fig, ax = plt.subplots()

    # scatter plot the samples
    contour_levels = np.exp(np.linspace(np.log(min_val), np.log(max_val), 15))
    #ax.scatter(samples['D'], samples['N'], c = samples['nqs_loss'], cmap = cmap, norm = norm, alpha = 0.2,s = 1)
    # plot the contour (use color_map to color the contour)
    ax.tricontour(samples['D'], samples['N'], samples['nqs_loss'], cmap = cmap, 
                   norm = norm, alpha = 0.5, levels = contour_levels)
    # add color bar
    fig.colorbar(mappable, ax=ax)
    
    # if opts is not None, 
    # the opts is a dictionary with keys {'chin1', 'chin2', 'nqs'}
    # the value of each key is a dataframe with columns ['C','opt_N','opt_D'] and potenitally a column 'NN_loss'
    # iterate through the keys, to collect all the 'C' values
    # for each C, plot a line representing C = 6ND
        # plot the lines for C

    Cs = []
    for key in opts.keys():
        df = opts[key]
        Cs.extend(df['C'].tolist())
    Cs = list(set(Cs))
    for C in Cs:
        # plot a line representing C = 6ND
        # the line is N = C*billion/(6D*million)/million = C/(6D) * 1e-3
        # log linear space for D
        Ds = np.exp(np.linspace(np.log(min_D), np.log(max_D), 100))
        Ns = C/(6*Ds) * 1e-3
        # zip Ds and Ns
        NDs = list(zip(Ds, Ns))
        # filter out the NDs that are outside the range of samples
        NDs = [ND for ND in NDs if 
               ND[0] >= min_D and ND[0] <= max_D 
               and ND[1] >= min_N and ND[1] <= max_N]
        # round
        C_in_B = round(C, 1)
        ax.plot([ND[0] for ND in NDs], [ND[1] for ND in NDs], 
                color = 'black', alpha = 0.5, linestyle = '--')
        #, label = 'C = ' + str(C_in_B) + 'B')

    if opts is not None:
        for key in opts.keys():
            df = opts[key]
            # iterate through the rows of df
            for i in range(df.shape[0]):
                row = df.iloc[i]
                # if first time seeing this key, label it with the key
                if i == 0:
                    ax.plot(row['D']/million, row['N']/million, 
                            marker = shapes[key],
                            markersize = shape_size[key],
                            label = key + ' NN_loss: ' + str(round(row['NN_loss'], 2)),
                            color = color_map(row['NN_loss']),
                            # set edgecolor = 'black'
                            markeredgewidth = 1, markeredgecolor = 'black',
                            linestyle = ''
                            
                            )
                else:
                    ax.plot(row['D']/million, row['N']/million, 
                            marker = shapes[key],
                            markersize = shape_size[key], 
                            color = color_map(row['NN_loss']),
                            # set edgecolor = 'black'
                            markeredgewidth = 1, markeredgecolor = 'black',
                            linestyle = ''
                            )
                
    # set x limit with min and max D
    ax.set_xlim([min_D, max_D])
    # set y limit with min and max N
    ax.set_ylim([min_N, max_N])
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # add legend
    ax.legend()
    # add axis labels   
    ax.set_xlabel('D (million)', fontsize = font_size, fontweight = font_weight)
    ax.set_ylabel('N (million)', fontsize = font_size, fontweight = font_weight)
    # save the figure
    plt.savefig(output_file)
    plt.close()
    return None



def deepseek_plot(h_samples_deepseek_with_nn_loss,
                      h_samples_test_with_nn_loss,
                      deepseek_opt_calc,
                      y_axis,
                      out_file):
        
        get_deepseek_opt, df_annotate = deepseek_opt_calc(h_samples_deepseek_with_nn_loss, y_axis = y_axis)
        h_samples_deepseek_with_nn_loss["type"] = "deepseek_train"

        # concatenate the two dataframes
        df = pd.concat([h_samples_deepseek_with_nn_loss, h_samples_test_with_nn_loss], axis=0)
        min_val = df["NN_loss"].min()
        max_val = df["NN_loss"].max()
        # define plotting utilities, color scale and norm using the min and max of NN_loss
        norm = LogNorm(vmin=min_val, vmax=max_val)

        #wistia
        cmap = plt.cm.bwr
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])

            # color map function
        def color_map(val):
            return mappable.to_rgba(val)

        # define marker shapes dictionary
        shapes = {'deepseek_train': 'o', 'deepseek': 'X', 'NQS': '*'}
        shape_size = {'deepseek_train': 10, 'deepseek': 10, 'NQS': 12}
        font_size = 14
        font_weight = 'normal'

        # scatter plot, C is the x-axis, y_axis is the y-axis, color is the NN_loss
        fig, ax = plt.subplots()
        # first plot the deepseek_train data
        df_train = df[df["type"] == "deepseek_train"]
        ax.scatter(df_train["C"], df_train[y_axis], color = "grey", alpha = 0.5,
                   marker = shapes["deepseek_train"], s = shape_size["deepseek_train"], label = "Deepseek Train")
        # filter for df_train where keep == True
        df_train_keep = df_annotate[df_annotate["keep"] == True]
        #df_train_keep = df_train_keep[df_train_keep["keep"] == True]
        # plot the deepseek_train data where keep == True
        ax.scatter(df_train_keep["C"], df_train_keep[y_axis], c = df_train_keep["NN_loss"], cmap = cmap, norm = norm, 
                   marker = shapes["deepseek_train"], s = shape_size["deepseek_train"], label = "Deepseek Train Keep")
        # get the unique C values
        C_vals = df_train["C"].unique()
        deepseek_opt_y_vals = [get_deepseek_opt(C) for C in C_vals]
        ax.plot(C_vals, deepseek_opt_y_vals, color = "grey", label = "Deepseek Opt", linewidth = 5, alpha = 0.5)
                
        # next plot the deepseek data
        df_deepseek = df[df["type"] == "deepseek"]

        # label with NN_loss
        ax.plot(df_deepseek["C"], df_deepseek[y_axis], 
                   marker = shapes["deepseek"], 
                   markersize = shape_size["deepseek"], label = "Deepseek: NN Loss = " + str(round(df_deepseek["NN_loss"].values[0], 2)),
                   color = color_map(df_deepseek["NN_loss"].values[0]),
                   markeredgewidth = 1, markeredgecolor = 'black',
                   linestyle = '')
        # next plot the NQS data
        df_NQS = df[df["type"] == "NQS"]
        ax.plot(df_NQS["C"], df_NQS[y_axis]
                     , marker = shapes["NQS"], 
                     markersize = shape_size["NQS"], label = "NQS: NN Loss = " + str(round(df_NQS["NN_loss"].values[0], 2)),
                     color = color_map(df_NQS["NN_loss"].values[0]),
                     markeredgewidth = 1, markeredgecolor = 'black',
                   linestyle = '')
        
        
        
        ax.set_xlabel("C", fontsize = font_size, fontweight = font_weight)
        ax.set_ylabel(y_axis, fontsize = font_size, fontweight = font_weight)
        ax.set_title("Deepseek Optimal " + y_axis + " vs. C")
        ax.set_xscale("log")
        ax.set_yscale("log")

        # add colorbar
        fig.colorbar(mappable, ax=ax)

        ax.legend()
        plt.savefig(out_file)

        return None
     
