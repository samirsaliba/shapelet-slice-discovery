import numpy as np
from .individual import Shapelet, ShapeletIndividual
import torch
from torch.nn import functional as F
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


def random_shapelet(X, n_shapelets, min_len, max_len, np_random):
    """Extract random subseries from the training set"""
    shapelets = []
    m = len(X[0])
    for _ in range(n_shapelets):
        row = np_random.integers(X.shape[0])
        length = np_random.integers(min_len, max_len)
        col = np_random.integers(0, m - length)
        shap_series = X[row][col : col + length]
        shapelet = Shapelet(shap_series, index=row, start=col)
        shapelets.append(shapelet)

    return shapelets


##########################################################################
#                         Mutatation operators                           #
##########################################################################
# Interface
# ---------
# INPUT:
#    - shapelets (np.array)
# OUTPUT:
#    - new_shapelets (np.array)


def smooth_shapelet(individual, np_random, device="cuda", **kwargs):
    """Smooth a random shapelet using PyTorch"""
    # TODO
    # raise NotImplementedError
    # Choose a random shapelet in the individual
    rand_shapelet = np_random.integers(len(individual))
    shap = individual[rand_shapelet]

    # Convert the shapelet to a PyTorch tensor
    shap_torch = torch.tensor(shap, dtype=torch.float32, device=device).view(1, 1, -1)

    # Define a smoothing kernel (moving average) and apply convolution
    window = 5
    kernel = torch.ones(1, 1, window, device=device) / window

    # Apply 1D convolution to compute the moving average
    shap_smoothed = F.conv1d(shap_torch, kernel, padding=0).squeeze()

    # Adjust the smoothed shapelet length to match the original by padding at the end
    if shap_smoothed.size(0) < shap_torch.size(2):
        fill_value = shap_smoothed[-1].item()
        padding_size = shap_torch.size(2) - shap_smoothed.size(0)
        shap_smoothed = torch.cat(
            [shap_smoothed, torch.full((padding_size,), fill_value, device=device)]
        )

    # Update the shapelet and reset its ID
    individual[rand_shapelet] = Shapelet(
        shap_smoothed.cpu().numpy(), index=shap.index, start=shap.start
    )
    individual[rand_shapelet]
    individual.register_op("smooth")
    individual.reset()

    return individual


def remove_shapelet(individual, np_random, remove_last=False):
    """Remove a random shapelet from the individual"""
    assert (
        len(individual) > 0
    ), "remove_shapelet requires an individual with at least 1 shapelets"

    if remove_last:
        individual.pop()
        individual.pop_uuid()
        return individual
    rand_shapelet = np_random.integers(len(individual))

    individual.pop(rand_shapelet)
    individual.reset()

    return individual


def mask_shapelet(individual, min_len, np_random, **kwargs):
    """Mask part of a random shapelet from the individual"""
    rand_shapelet = np_random.integers(len(individual))
    len_shap = len(individual[rand_shapelet])

    if len_shap > min_len:
        rand_start = np_random.integers(len_shap - min_len)
        rand_end = np_random.integers(rand_start + min_len, len_shap)
        shap = individual[rand_shapelet]
        masked_series = Shapelet(
            shap[rand_start:rand_end], index=shap.index, start=rand_start
        )

        individual[rand_shapelet] = masked_series

    return individual


def replace_shapelet(X, individual, min_len, max_len, np_random):
    assert (
        len(individual) > 0
    ), "replace_shapelet requires an individual with at least one shapelet"

    individual = remove_shapelet(individual=individual, np_random=np_random)
    return add_shapelet(
        X=X,
        individual=individual,
        min_len=min_len,
        max_len=max_len,
        np_random=np_random,
    )


def add_shapelet(X, individual, min_len, max_len, np_random):
    """
    Mutation operator that adds a new shapelet from the same time series instance as an existing one,
    ensuring the new shapelet is not identical to any already in the individual and meets the minimum length requirement.

    Parameters:
    X (np.ndarray): The time series data (2D array where rows are instances and columns are time steps)
    individual (ShapeletIndividual): The individual to mutate
    max_len (int): Maximum length allowed for a shapelet
    min_length (int): Minimum length allowed for a shapelet

    Returns:
    individual (ShapeletIndividual): The mutated individual with an additional shapelet
    """
    if len(individual) == 0:
        individual.append(
            random_shapelet(
                X=X,
                n_shapelets=1,
                min_len=min_len,
                max_len=max_len,
                np_random=np_random,
            )[0]
        )
        return individual

    if not hasattr(individual, "subgroup"):
        return individual

    individual.reset()
    index = individual[-1].index  # np_random.choice(np.where(individual.subgroup)[0])

    # Get the total length of the time series
    time_series_length = X.shape[1]
    new_start = np_random.integers(0, time_series_length - max_len)
    new_length = np_random.integers(min_len, max_len)

    # Create the new shapelet
    timeseries = X[index][new_start : new_start + new_length]
    new_shapelet = Shapelet(timeseries, index=index, start=new_start)

    # Add the new shapelet to the individual's shapelet list
    individual.append(new_shapelet)
    return individual


def slide_shapelet(X, individual, max_slide=20, **kwargs):
    """Slide a random shapelet forwards or backwards within the X data.

    Args:
        shapelets (list): A list of Shapelet objects.
        X (np.ndarray): The time series matrix where each row is a time series.
        max_slide (int): Maximum number of positions to slide forwards or backwards.

    Returns:
        shapelets (list): The modified list of Shapelet objects.
    """
    # TODO
    raise NotImplementedError

    individual.reset()

    rand_shapelet_idx = np_random.integers(len(individual))
    shapelet = individual[rand_shapelet_idx]

    # Get the corresponding time series row from X based on shapelet index
    timeseries = X[shapelet.index]

    # Determine the length of the shapelet
    shapelet_length = len(shapelet)

    # Define the range within which we can slide the shapelet
    start_min = max(0, shapelet.start - max_slide)
    start_max = min(len(timeseries) - shapelet_length, shapelet.start + max_slide)

    # Choose a new random start position within the valid range
    new_start = np_random.integers(start_min, start_max + 1)

    # Extract the new shapelet from the time series with the new start position
    new_shapelet_data = timeseries[new_start : new_start + shapelet_length]

    # Create a new Shapelet with the updated start position
    individual[rand_shapelet_idx] = Shapelet(
        new_shapelet_data, index=shapelet.index, start=new_start
    )

    return individual


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


def crossover_AND(ind1, ind2, **kwargs):
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

    return (
        child,
        ind1,
    )  # Returning child and one parent clone for compatibility with DEAP


def crossover_uniform(ind1, ind2, np_random, **kwargs):
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

        if np_random.rand() < 0.5:
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
