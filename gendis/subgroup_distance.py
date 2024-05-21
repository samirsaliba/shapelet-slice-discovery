import numpy as np
from scipy.stats import wasserstein_distance, mannwhitneyu
from sklearn.preprocessing import StandardScaler

class SubgroupDistance:
    """
    Parameters
    ----------
    distance_function : callable
        The distance function to use
    shapelet_dist_threshold : float
        The threshold for the distance matrix
    """

    def __init__(
        self,
        distance_function, 
        shapelet_dist_threshold, 
        min_support,
        cov_alpha,
        cov_weights,
        coverage_weights=None,
        standardize=False
    ):
        self.cov_alpha = cov_alpha
        self.cov_weights = cov_weights
        self.shapelet_dist_threshold = shapelet_dist_threshold
        self.distance_function = distance_function
        self.min_support = min_support if min_support is not None else -1
        self.standardize = standardize

    def __call__(self, D, y, shaps=None):
        return self.distance(D, y)

    def coverage_factor(self, subgroup):
        """Multiplicative weighted covering score"""
        in_sg_weights = self.cov_weights[subgroup].sum()
        sg_weights_total = subgroup.sum() * self.cov_alpha
        return in_sg_weights / sg_weights_total

    def filter_subgroup_shapelets(self, D, y):
        """
        Parameters
        ----------
        y : array-like
            The target variable
        D : array-like
            The distance matrix
        shapelet_dist_threshold : float
            The threshold for the distance matrix

        Returns
        -------
        subgroup : array-like
            Boolean array indicating whether instance i
            belongs to subgroup

        """
        if self.standardize:
            _D = np.absolute(StandardScaler().fit_transform(D))
        else:
            _D = D
        return np.all(_D <= self.shapelet_dist_threshold, axis=1)


    def distance(self, D, y):
        subgroup = self.filter_subgroup_shapelets(D, y)
        subgroup_y = y[subgroup]
        subgroup_n = sum(subgroup)

        fitness = -np.inf
        subgroup_mean_distance = np.inf
        covW = None
        dist = None

        if subgroup_n  >= self.min_support:
            subgroup_mean_distance = D[subgroup].mean(0).min()
            dist = self.distance_function(subgroup_y, y)
            covW = self.coverage_factor(subgroup)
            fitness =  covW * dist

        sg_mean = np.mean(subgroup_y)
        return {
            'value': np.array([fitness]),
            'info': {
                'dist': dist,
                'coverage_weight': covW,
                'subgroup_error_mean': sg_mean,
                'population_mean': np.mean(y),
                'subgroup_size': subgroup_n,
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
