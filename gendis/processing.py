import copy
from dtaidistance.preprocessing import differencing
import numpy as np
import pandas as pd


def preprocess_input(X, y):
    _X = copy.deepcopy(X)
    if isinstance(_X, pd.DataFrame):
        _X = _X.values
    _X = np.apply_along_axis(lambda s: differencing(s, smooth=None), 1, _X)

    _y = copy.deepcopy(y)
    if isinstance(_y, pd.Series):
        _y = y.values

    return _X, _y


def undifferentiate_series(series, offset=0):
    return np.insert(np.cumsum(series), 0, 0) + offset
