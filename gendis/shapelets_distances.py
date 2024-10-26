from dtaidistance import dtw as _dtw
from dtaidistance import ed_cc
import numpy as np


def calculate_shapelet_dist_matrix(
    X, shapelets, dist_function, return_positions=False, cache=None
):
    """Calculate the distance matrix for a set of shapelets"""
    D = -1 * np.ones((len(X), len(shapelets)))
    L = -1 * np.ones((len(X), len(shapelets)))

    if cache is None:
        cache = {}

    for shap_ix, shap in enumerate(shapelets):
        cache_val = cache.get(shap.id)
        if cache_val is not None:
            d, l = cache_val

        else:
            # res = np.apply_along_axis(sliding_window_dist, 1, X, shap=shap)
            res = dist_function(X, shap, D, L)
            d, l = res[:, 0], res[:, 1]
            cache.set(shap.id, (d, l))

        D[:, shap_ix] = d
        L[:, shap_ix] = l

    return D, L


def sliding_window_dist(x, shap):
    x = x.copy()
    step = 1
    shap_len = len(shap)

    # Extract all possible windows (subsequences) from x with length equal to the shapelet
    n_windows = (len(x) - shap_len) // step + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, shap_len)[::step]

    # Compute the distance between the shapelet and all windows
    distances = np.array([ed_cc.distance(window.copy(), shap) for window in windows])

    # Find the minimum distance and the corresponding position
    pos = np.argmin(distances)
    min_dist = distances[pos]

    return min_dist, pos


def _row_dist_sliding_helper(x, shap, dist_fn):
    step = 1
    remainder = len(x) % step
    min_dist = np.inf
    pos = None
    for k in range(0, len(x) - len(shap) + 1 - remainder, step):
        dist = dist_fn(x[k : k + len(shap)], shap)
        if dist < min_dist:
            min_dist = dist
            pos = k
    return min_dist, pos


def _distance_wrapper(timeseries_matrix, shaps, distances, dist_fn):
    apply_fn = lambda x: _row_dist_sliding_helper(x, shaps, dist_fn)
    return np.apply_along_axis(apply_fn, 1, timeseries_matrix)


def dtw(timeseries_matrix, shaps, distances, positions):
    return _distance_wrapper(timeseries_matrix, shaps, distances, _dtw.distance_fast)


def euclidean(timeseries_matrix, shaps, distances, positions):
    return _distance_wrapper(timeseries_matrix, shaps, distances, ed_cc.distance)
