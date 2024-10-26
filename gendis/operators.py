import copy
import numpy as np
import random
from .individual import Shapelet, ShapeletIndividual
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
    """Extract random subseries from the training set"""
    shapelets = []
    for _ in range(n_shapelets):
        rand_row = np.random.randint(X.shape[0])
        rand_length = np.random.randint(max(4, min_len), min(min_len_series, max_len))
        rand_col = np.random.randint(min_len_series - rand_length)
        shap_series = X[rand_row][rand_col : rand_col + rand_length]
        shapelet = Shapelet(shap_series, index=rand_row, start=rand_col)
        shapelets.append(shapelet)

    return shapelets


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
        subseries[i] = X[indices_ts[i]][start_idx[i] : end_idx[i]]

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


def smooth_shapelet(X, individual, toolbox, min_len, max_len):
    """Smooth a random shapelet"""
    rand_shapelet = np.random.randint(len(individual))
    shap = individual[rand_shapelet]

    window = 5
    shap = np.ravel(shap)
    shap_mva = np.convolve(shap, np.ones(window), "valid") / window
    fill = np.full(shape=len(shap) - len(shap_mva), fill_value=shap_mva[-1])
    shap = np.concatenate([shap_mva, fill])
    shap.reset_id()
    individual[rand_shapelet] = shap
    return individual


def remove_shapelet(X, individual, toolbox, max_len, remove_last=False):
    """Remove a random shapelet from the individual"""
    if len(individual) == 1:
        raise NotImplementedError

    if remove_last:
        individual.pop()
        return individual

    rand_shapelet = np.random.randint(len(individual))
    individual.pop(rand_shapelet)
    individual.register_op("remove_rand")

    return individual


def mask_shapelet(X, individual, toolbox, min_len, max_len):
    shap_min_n = 4
    """Mask part of a random shapelet from the individual"""
    rand_shapelet = np.random.randint(len(individual))
    len_shap = len(individual[rand_shapelet])
    if len_shap > shap_min_n:
        rand_start = np.random.randint(len_shap - shap_min_n)
        rand_end = np.random.randint(rand_start + shap_min_n, len_shap)
        shap_series = individual[rand_shapelet][rand_start:rand_end]
        individual[rand_shapelet] = Shapelet(
            shap_series, index=rand_shapelet, start=rand_start
        )

    return individual


def replace_shapelet(X, individual, toolbox, min_len, max_len):
    individual = remove_shapelet(X, individual, toolbox, min_len, max_len)
    return add_shapelet(X, individual, toolbox, min_len, max_len)


def add_shapelet(X, individual, toolbox, min_len, max_len):
    """
    Mutation operator that adds a new shapelet from the same time series instance as an existing one,
    ensuring the new shapelet is not identical to any already in the individual and meets the minimum length requirement.

    Parameters:
    individual (ShapeletIndividual): The individual to mutate
    X (np.ndarray): The time series data (2D array where rows are instances and columns are time steps)
    max_len (int): Maximum length allowed for a shapelet
    min_length (int): Minimum length allowed for a shapelet

    Returns:
    individual (ShapeletIndividual): The mutated individual with an additional shapelet
    """
    # Select an existing shapelet from the individual (e.g., the first one)
    existing_shapelet = individual[0]

    # Extract properties from the current shapelet
    index = existing_shapelet.index
    current_start = existing_shapelet.start
    current_length = len(existing_shapelet)

    # Get the total length of the time series
    time_series_length = X.shape[1]

    # Ensure the new shapelet is not exactly the same as the existing one
    while True:
        # Randomly select a new start position and length, ensuring it meets the min_length
        new_start = random.randint(0, time_series_length - max_len)
        new_length = random.randint(min_len, max_len)

        # Check if the new shapelet is different from the existing one
        if not (new_start == current_start and new_length == current_length):
            break

    # Create the new shapelet
    timeseries = X[index, new_start : new_start + new_length]
    new_shapelet = Shapelet(timeseries, index=index, start=new_start)

    # Add the new shapelet to the individual's shapelet list
    individual.append(new_shapelet)
    individual.register_op("mut_add")
    individual.reset()
    return individual


def slide_shapelet(shapelets, X, max_slide=5):
    """Slide a random shapelet forwards or backwards within the X data.

    Args:
        shapelets (list): A list of Shapelet objects.
        X (np.ndarray): The time series matrix where each row is a time series.
        max_slide (int): Maximum number of positions to slide forwards or backwards.

    Returns:
        shapelets (list): The modified list of Shapelet objects.
    """
    # Select a random shapelet from the individual
    rand_shapelet_idx = np.random.randint(len(shapelets))
    shapelet = shapelets[rand_shapelet_idx]

    # Get the corresponding time series row from X based on shapelet index
    timeseries = X[shapelet.index]

    # Determine the length of the shapelet
    shapelet_length = len(shapelet)

    # Define the range within which we can slide the shapelet
    start_min = max(0, shapelet.start - max_slide)
    start_max = min(len(timeseries) - shapelet_length, shapelet.start + max_slide)

    # Choose a new random start position within the valid range
    new_start = np.random.randint(start_min, start_max + 1)

    # Extract the new shapelet from the time series with the new start position
    new_shapelet_data = timeseries[new_start : new_start + shapelet_length]

    # Create a new Shapelet with the updated start position
    shapelets[rand_shapelet_idx] = Shapelet(
        new_shapelet_data, index=shapelet.index, start=new_start
    )

    return (shapelets,)


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
    Perform crossover by creating a new individual from the union of two individuals' shapelets,
    ensuring that both parents' shapelets are included in the child.

    Parameters:
    ind1 (ShapeletIndividual): First parent individual
    ind2 (ShapeletIndividual): Second parent individual

    Returns:
    child1 (ShapeletIndividual): New individual created from the union of ind1 and ind2 shapelets
    child2 (ShapeletIndividual): New individual (clone of ind1) to preserve DEAP operator interface
    """
    # Create a new individual containing shapelets from both parents
    new_shapelets = list(ind1) + list(ind2)
    child = ShapeletIndividual(list(new_shapelets))
    child.register_op("co_and")

    return (
        child,
        ind1,
    )  # Returning child and one parent clone for compatibility with DEAP


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
