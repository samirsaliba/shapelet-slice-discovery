import matplotlib.pyplot as plt

def plot_error_distributions(df, img_path):
    for lb in df['label'].value_counts().index:
        plt.hist(df.loc[df['label']==lb, 'error'], alpha=0.5, label=lb)
    plt.legend(loc='upper right')
    img_file = f'{path}/error_dist.png'
    plt.savefig(img_file)

def plot_best_matching_shaps(gendis, individual, X_input, y):
    X_input = X_input.copy()
    y = y.copy()
    
    # Get distances matrix and subgroup mask
    shaps = individual['shaps']
    distances = gendis.transform(X_input, y, shapelets=shaps, return_positions=True)
    in_sg_mask = distances["in_subgroup"]==1
    
    # Filter datasets based on subgroup mask
    distances = distances[in_sg_mask]
    X_input = X_input[in_sg_mask]
    
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
        timeseries = X_input[i]
        axs[i].plot(timeseries, alpha=0.4)
        
        # Plot shapelets
        for j, shap in enumerate(shaps):
            position = distances.loc[distances.index[i], f'L_{j}']
            shap_x = np.arange(position, position+len(shap))
            axs[i].plot(shap_x, shap, alpha=0.8)
            
    plt.xticks(np.arange(0, len(timeseries)+1, 30.0))
    plt.tight_layout()
    if show:
        plt.show()
    else:
        img_file = f'{path}/shapelets_matching_plots_top_{plot_i}.png'
        plt.savefig(img_file)
