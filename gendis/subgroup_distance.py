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
        cov_alpha,
        cov_weights,
        coverage_weights=None,
        sg_size_beta=2,
        standardize=False
    ):
        self.cov_alpha = cov_alpha
        self.cov_weights = cov_weights
        self.sg_size_beta = sg_size_beta
        self.shapelet_dist_threshold = shapelet_dist_threshold
        self.distance_function = distance_function
        self.standardize = standardize

    def __call__(self, D, y, shaps=None):
        return self.distance(D, y)

    def update_weights(self, cov_weights):
        self.cov_weights=cov_weights

    def subgroup_size_factor(self, subgroup_n, y_n):
        """Score that favors larger subgroups"""
        return (subgroup_n / y_n) ** self.sg_size_beta

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

        if subgroup_n < 10:
            fitness = -np.inf
            covW = None
            sizeW = None
            dist = None

        else:
            dist = self.distance_function(subgroup_y, y)
            covW = self.coverage_factor(subgroup)
            sizeW = self.subgroup_size_factor(subgroup_n, len(y))
            fitness =  dist * covW * sizeW

        sg_mean = np.mean(subgroup_y)
        return {
            'value': np.array([fitness]),
            'info': {
                'fitness': fitness,
                'dist': dist,
                'subgroup_size_weight': sizeW,
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
