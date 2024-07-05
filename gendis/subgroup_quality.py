import numpy as np
from scipy.stats import wasserstein_distance, mannwhitneyu
from sklearn.preprocessing import StandardScaler

class SubgroupQuality:
    """
    Parameters
    ----------
    distance_function : callable
        The distribution distance function to use
    shapelet_dist_threshold : float
        The threshold for the distance matrix
    """

    def __init__(
        self,
        distance_function, 
        shapelet_dist_threshold, 
        sg_size_beta,
        standardize=False
    ):
        self.sg_size_beta = sg_size_beta
        self.shapelet_dist_threshold = shapelet_dist_threshold
        self.distance_function = distance_function
        self.standardize = standardize

    def __call__(self, D, y, shaps=None):
        return self.evaluate(D, y)

    def subgroup_size_factor(self, subgroup_n, y_n):
        """Score that favors larger subgroups"""
        return (subgroup_n / y_n) ** self.sg_size_beta

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


    def evaluate(self, D, y):
        subgroup = self.filter_subgroup_shapelets(D, y)
        subgroup_y = y[subgroup]
        subgroup_n = np.sum(subgroup)

        distribution_delta = self.distance_function(subgroup_y, y)
        sizeW = self.subgroup_size_factor(subgroup_n, len(y))
        fitness =  distribution_delta * sizeW

        sg_mean = np.mean(subgroup_y)
        return {
            'valid': subgroup_n>0,
            'value': np.array([fitness]),
            'subgroup': subgroup,
            'info': {
                'fitness': fitness,
                'distribution_delta': distribution_delta,
                'subgroup_size_weight': sizeW,
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
