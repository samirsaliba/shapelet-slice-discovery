import numpy as np
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
    if n_shapelets > 1:
        return np.array(shaps)
    else:
        return np.array(shaps[0])


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
