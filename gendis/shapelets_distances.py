from dtaidistance import dtw as _dtw
from dtaidistance import ed_cc
import numpy as np

def _row_dist_sliding_helper(x, shaps, dist_fn):
    step = 3
    shap_dists = []
    for shap in shaps:
        remainder = len(x) % step
        min_dist = np.inf
        for k in range(0, len(x) - len(shap) + 1 - remainder, step):
            dist = dist_fn(x[ k : k+len(shap) ], shap)
            if dist < min_dist: 
                min_dist = dist
        shap_dists.append(min_dist)
    return np.array(shap_dists)

def _distance_wrapper(timeseries_matrix, shaps, distances, dist_fn):
    apply_fn = lambda x: _row_dist_sliding_helper(x, shaps, dist_fn)
    return np.apply_along_axis(apply_fn, 1, timeseries_matrix)

def dtw(timeseries_matrix, shaps, distances):
    return _distance_wrapper(timeseries_matrix, shaps, distances, _dtw.distance_fast)

def euclidean(timeseries_matrix, shaps, distances):
    return _distance_wrapper(timeseries_matrix, shaps, distances, ed_cc.distance)