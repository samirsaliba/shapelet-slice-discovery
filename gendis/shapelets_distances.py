from dtaidistance import dtw as _dtw
from dtaidistance import ed_cc
import numpy as np

try:
    from gendis.pairwise_dist import _pdist, _pdist_location
except:
    from pairwise_dist import _pdist, _pdist_location


def calculate_shapelet_dist_matrix(
    X, shapelets, 
    dist_function,
    return_positions=False,
    cache=None
    ):
    """Calculate the distance matrix for a set of shapelets"""
    D = -1 * np.ones((len(X), len(shapelets)))
    L = None

    # First check if we already calculated distances for a shapelet
    if cache is not None:
        for shap_ix, shap in enumerate(shapelets):
            shap_hash = hash(tuple(shap.flatten()))
            cache_val = cache.get(shap_hash)
            if cache_val is not None:
                D[:, shap_ix] = cache_val

    # Fill up the 0 entries
    res = dist_function(X, [shap.flatten() for shap in shapelets], D)
    D = res[:,0,:]

    if return_positions:
        L = res[:,1,:]

    # Fill up our cache
    if cache is not None:
        for shap_ix, shap in enumerate(shapelets):
            shap_hash = hash(tuple(shap.flatten()))
            cache.set(shap_hash, D[:, shap_ix])

    return D, L

def _row_dist_sliding_helper(x, shaps, dist_fn):
    step = 3
    shap_dists = []
    shap_positions = []
    for shap in shaps:
        remainder = len(x) % step
        min_dist = np.inf
        pos = None
        for k in range(0, len(x) - len(shap) + 1 - remainder, step):
            dist = dist_fn(x[ k : k+len(shap) ], shap)
            if dist < min_dist: 
                min_dist = dist
                pos = k
        shap_dists.append(min_dist)
        shap_positions.append(pos)
    return np.array(shap_dists), np.array(shap_positions)


def _distance_wrapper(timeseries_matrix, shaps, distances, dist_fn):
    apply_fn = lambda x: _row_dist_sliding_helper(x, shaps, dist_fn)
    return np.apply_along_axis(apply_fn, 1, timeseries_matrix)


def dtw(timeseries_matrix, shaps, distances):
    return _distance_wrapper(
        timeseries_matrix, shaps, distances, _dtw.distance_fast)

def euclidean(timeseries_matrix, shaps, distances):
    return _distance_wrapper(
        timeseries_matrix, shaps, distances, ed_cc.distance)