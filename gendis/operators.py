import numpy as np
import random
from tslearn.clustering import TimeSeriesKMeans

##########################################################################
#                       Initialization operators                         #
##########################################################################
# Interface
# ---------
# INPUT: 
#    - X (np.array)
#    - n_shapelets (int)
# OUTPUT: 
#    - shapelets (np.array)

def random_shapelet(X, n_shapelets, min_len_series, max_len, min_len=4):
    """Extract a random subseries from the training set"""
    shaps = []
    for _ in range(n_shapelets):
        rand_row = np.random.randint(X.shape[0])
        rand_length = np.random.randint(max(4, min_len), min(min_len_series, max_len))
        rand_col = np.random.randint(min_len_series - rand_length)
        shaps.append(X[rand_row][rand_col:rand_col+rand_length])
    # if n_shapelets > 1:
    #     return np.array(shaps)
    # else:
    #     return np.array(shaps[0])
    return np.array(shaps)


def kmeans(X, n_shapelets, min_len_series, max_len, n_draw=None, min_len=4):
    """Sample subseries from the timeseries and apply K-Means on them"""
    # Sample `n_draw` subseries of length `shp_len`
    if n_shapelets == 1:
        return random_shapelet(X, n_shapelets, min_len_series, max_len)
    if n_draw is None:
        n_draw = max(n_shapelets, int(np.sqrt(len(X))))
    shp_len = np.random.randint(max(4, min_len), min(min_len_series, max_len))
    indices_ts = np.random.choice(len(X), size=n_draw, replace=True)
    start_idx = np.random.choice(min_len_series - shp_len, size=n_draw, replace=True)
    end_idx = start_idx + shp_len

    subseries = np.zeros((n_draw, shp_len))
    for i in range(n_draw):
        subseries[i] = X[indices_ts[i]][start_idx[i]:end_idx[i]]

    tskm = TimeSeriesKMeans(n_clusters=n_shapelets, metric="euclidean", verbose=False)
    return tskm.fit(subseries).cluster_centers_

##########################################################################
#                         Mutatation operators                           #
##########################################################################
# Interface
# ---------
# INPUT: 
#    - shapelets (np.array)
# OUTPUT: 
#    - new_shapelets (np.array)

def smooth_shapelet(shapelets, toolbox):
    """Smooth a random shapelet"""
    rand_shapelet = np.random.randint(len(shapelets))
    shap = shapelets[rand_shapelet]

    window = 5
    shap = np.ravel(shap)
    shap_mva = np.convolve(shap, np.ones(window), 'valid') / window
    fill = np.full(shape=len(shap)-len(shap_mva), fill_value=shap_mva[-1])
    shap = np.concatenate([shap_mva, fill])
    shapelets[rand_shapelet] = shap
    return shapelets,


def add_shapelet(shapelets, toolbox):
    """Add a shapelet to the individual"""
    shapelets.append(toolbox.create(n_shapelets=1))
    return shapelets,


def remove_shapelet(shapelets, toolbox):
    """Remove a random shapelet from the individual"""

    if len(shapelets) == 1:
        return replace_shapelet(shapelets, toolbox)
    
    rand_shapelet = np.random.randint(len(shapelets))
    shapelets.pop(rand_shapelet)

    return shapelets,


def replace_shapelet(shapelets, toolbox):
    """
    Replace a random shapelet in the individual with a newly created one.
    
    Parameters:
    shapelets (list of lists): The individual (list of shapelets)
    toolbox: The toolbox with the function to create a new shapelet
    
    Returns:
    shapelets (list of lists): The modified individual with one shapelet replaced
    """
    if len(shapelets) > 0:
        # Randomly select an index to remove
        remove_index = random.randint(0, len(shapelets) - 1)
        shapelets.pop(remove_index)
    
    shapelets.append(toolbox.create(n_shapelets=1))
    return shapelets,


def mask_shapelet(shapelets, toolbox):
    shap_min_n = 4
    """Mask part of a random shapelet from the individual"""
    rand_shapelet = np.random.randint(len(shapelets))
    len_shap = len(shapelets[rand_shapelet])
    if len_shap > shap_min_n:
        rand_start = np.random.randint(len_shap - shap_min_n)
        rand_end = np.random.randint(rand_start + shap_min_n, len_shap)
        shapelets[rand_shapelet] = shapelets[rand_shapelet][rand_start:rand_end]

    return shapelets,


##########################################################################
#                         Crossover operators                            #
##########################################################################
# Interface
# ---------
# INPUT: 
#    - ind1 (np.array)
#    - ind2 (np.array)
# OUTPUT: 
#    - new_ind1 (np.array)
#    - new_ind2 (np.array)

def crossover_AND(ind1, ind2):
    """
    Perform crossover by creating a new individual from the union of two individuals' shapelets.
    
    Parameters:
    ind1 (list of lists): First parent individual
    ind2 (list of lists): Second parent individual
    
    Returns:
    new_ind1 (list of lists): New individual created from the union of ind1 and ind2
    new_ind2 (list of lists): Duplicate of new_ind1 (for consistency with interface)
    """
    return ind2.extend(ind1), ind1


def crossover_uniform(ind1, ind2):
    """
    Perform uniform crossover with a 50% mixing ratio on shapelets.
    
    Parameters:
    ind1 (list of lists): First parent individual
    ind2 (list of lists): Second parent individual
    
    Returns:
    new_ind1 (list of lists): First new individual created by uniform crossover
    new_ind2 (list of lists): Second new individual created by uniform crossover
    """
    max_length = max(len(ind1), len(ind2))
    
    new_ind1 = []
    new_ind2 = []
    
    for i in range(max_length):
        if i < len(ind1) and i < len(ind2):
            shapelet1 = ind1[i]
            shapelet2 = ind2[i]
        elif i < len(ind1):
            shapelet1 = ind1[i]
            shapelet2 = None
        else:
            shapelet1 = None
            shapelet2 = ind2[i]
        
        if np.random.rand() < 0.5:
            if shapelet1 is not None:
                new_ind1.append(shapelet1)
            if shapelet2 is not None:
                new_ind2.append(shapelet2)
        else:
            if shapelet2 is not None:
                new_ind1.append(shapelet2)
            if shapelet1 is not None:
                new_ind2.append(shapelet1)
    
    return new_ind1, new_ind2
