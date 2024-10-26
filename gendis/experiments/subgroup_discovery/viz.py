import matplotlib.pyplot as plt
import numpy as np

def plot_error_distributions(df, path):
    for lb in df['label'].value_counts().index:
        plt.hist(df.loc[df['label']==lb, 'error'], alpha=0.5, label=lb)
    plt.legend(loc='upper right')
    img_file = f'{path}/error_dist.png'
    plt.savefig(img_file)

def plot_shaps(shaps, path, ind_label, x_label='Time', y_label='Value'):
    """
    Plots multiple shapelets on separate subplots.
    
    Parameters:
    - shaps: List of shapelets (each shapelet is a 1D array).
    - title: Optional title for the entire figure.
    - x_label: Label for the x-axis (shared across all subplots).
    - y_label: Label for the y-axis (shared across all subplots).
    """
    # Plot setup
    k = len(shaps)
    axs_multiplier = 1
    width = 2 * axs_multiplier * 6.4
    height = k * axs_multiplier * 4.8

    # Create subplots
    fig, axs = plt.subplots(
        k, 1, 
        sharex=True,  # Share the x-axis among all subplots
        figsize=(width, height), 
        gridspec_kw={'hspace': 0.1}  # Adjust space between plots
    )

    # If only one shapelet, axs won't be an array, so we make it one for consistency
    if k == 1:
        axs = [axs]

    # Plot each shapelet
    for i, shap in enumerate(shaps):
        shap_x = np.arange(0, len(shap))
        axs[i].plot(shap_x, shap)
        axs[i].set_ylabel(f'Shapelet {i+1}')  # Label each subplot with shapelet index

    # Set common labels
    axs[-1].set_xlabel(x_label)  # Set x-axis label only for the last subplot
    for ax in axs:
        ax.set_ylabel(y_label)

    plt.tight_layout()
    img_file = f'{path}/shapelets_ind_{ind_label}.png'
    plt.savefig(img_file)


def plot_best_matching_shaps(gendis, individual, X_input, y, path, plot_i):
    X_input = X_input.copy()
    y = y.copy()
    
    # Get distances matrix and subgroup mask
    distances, subgroup = gendis.transform(X_input, y, shapelets=individual, return_positions=True)

    # Filter datasets based on subgroup mask
    distances = distances[subgroup]
    X_input = X_input[subgroup]
    
    # Create ordering index based on sum of distances
    d_cols = distances.filter(like='D_')
    distances['D_sum'] = d_cols.sum(axis=1)
    sort_indices = distances['D_sum'].argsort()
    
    # Order datasets by ordering index
    distances = distances.iloc[sort_indices]
    X_input = X_input[sort_indices]
    
    k=5
    
    # Plot setup
    axs_multiplier = 1
    width = 2*axs_multiplier*6.4
    height = (k//5 + 1)*axs_multiplier*4.8
    f, axs = plt.subplots(
        k, 1, 
        sharex=True, 
        figsize=(width, height),
        gridspec_kw={'hspace': 0.1,}
    )
    
    for i in range(k):
        # Plot timeseries
        if i >= len(X_input):
            break

        timeseries = X_input[i]
        axs[i].plot(timeseries, alpha=0.4)
        
        # Plot shapelets
        for j, shap in enumerate(individual):
            position = distances.loc[distances.index[i], f'L_{j}']
            shap_x = np.arange(position, position+len(shap))
            axs[i].plot(shap_x, shap, alpha=0.8)
            
    plt.xticks(np.arange(0, len(timeseries)+1, 30.0))
    plt.tight_layout()
    if False:
        plt.show() # For future reference
    else:
        img_file = f'{path}/shapelets_matching_plots_top_{plot_i}.png'
        plt.savefig(img_file)
