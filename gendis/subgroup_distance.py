import numpy as np
from scipy.stats import wasserstein_distance, mannwhitneyu

class SubgroupDistance:
    """
    Parameters
    ----------
    distance_function : callable
        The distance function to use
    shapelet_dist_threshold : float
        The threshold for the distance matrix
    """

    def __init__(self, distance_function, shapelet_dist_threshold, min_support, alpha=1):
        self.shapelet_dist_threshold = shapelet_dist_threshold
        self.distance_function = distance_function
        if min_support is None: min_support = -1
        self.min_support = min_support
        self.alpha = alpha

    def __call__(self, D, y, verbose=False):
        return self.distance(D, y, verbose)

    def coverage_factor(self, n_sg, n):
        # See https://pysubgroup.readthedocs.io/en/latest/sections/components/targets.html#numeric-targets
        return (n_sg/n) ** self.alpha

    @staticmethod
    def filter_subgroup_shapelets(y, D, shapelet_dist_threshold, return_filter=False):
        """
        Parameters
        ----------
        y : array-like
            The target variable
        D : array-like
            The distance matrix
        shapelet_dist_threshold : float
            The threshold for the distance matrix
        return_filter : bool, optional
            If True, return the filter instead of the filtered arrays

        Returns
        -------
        y_in : array-like
            The target variable filtered by the distance matrix
        y_out : array-like
            The target variable filtered by the distance matrix

        """

        subgroup_filter = np.all(D <= shapelet_dist_threshold, axis=1)
        if return_filter:
            return subgroup_filter
        y_in, y_out = y[subgroup_filter], y[~subgroup_filter]
        return y_in, y_out

    def distance(self, D, y, verbose=False):
        subgroup_y, rest_y = self.filter_subgroup_shapelets(
            y, D, self.shapelet_dist_threshold)
        subgroup_error_mean = np.mean(subgroup_y)
        if min(len(subgroup_y), len(rest_y)) < self.min_support:
            res, dist = 0, None
        else:
            dist = self.distance_function(subgroup_y, rest_y)
            res = self.coverage_factor(n_sg=len(subgroup_y), n=len(y)) * dist

        return {
            'value': (res, -D.shape[1], subgroup_error_mean),
            'info': {
                'dist': dist,
                'subgroup_error_mean': subgroup_error_mean,
                'rest_error_mean': np.mean(rest_y),
                'subgroup_size': len(subgroup_y),
            }
        }

    @staticmethod
    def wasserstein_distance(y1, y2):
        return wasserstein_distance(y1.reshape(-1), y2.reshape(-1))

    @staticmethod
    def mannwhitneyu(y1, y2):
        return 1 - mannwhitneyu(y1.reshape(-1), y2.reshape(-1)).pvalue

    @staticmethod
    def simple_mean(y1, y2):
        return np.absolute(np.mean(y1.reshape(-1)) - np.mean(y2.reshape(-1)))
