from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import numpy as np

from scipy.stats import wasserstein_distance, mannwhitneyu 

try:
    from gendis.pairwise_dist import _pdist, _pdist_location
except:
    from pairwise_dist import _pdist, _pdist_location

def logloss_fitness(X, y, shapelets, cache=None, verbose=False):
    """Calculate the fitness of an individual/shapelet set"""
    D = np.zeros((len(X), len(shapelets)))

    # First check if we already calculated distances for a shapelet
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache_val = cache.get(shap_hash)
        if cache_val is not None:
            D[:, shap_ix] = cache_val

    # Fill up the 0 entries
    _pdist(X, [shap.flatten() for shap in shapelets], D)

    # Fill up our cache
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache.set(shap_hash, D[:, shap_ix])

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(D, y)
    preds = lr.predict_proba(D)
    cv_score = -log_loss(y, preds)

    return (cv_score, sum([len(x) for x in shapelets]))


def logloss_fitness_location(X, y, shapelets, cache=None, verbose=False):
    """Calculate the fitness of an individual/shapelet set"""
    D = np.zeros((len(X), len(shapelets)))
    L = np.zeros((len(X), len(shapelets)))

    # First check if we already calculated distances for a shapelet
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache_val = cache.get(shap_hash)
        if cache_val is not None:
            D[:, shap_ix] = cache_val

    # Fill up the 0 entries
    _pdist_location(X, [shap.flatten() for shap in shapelets], D, L)

    # Fill up our cache
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache.set(shap_hash, D[:, shap_ix])

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(np.hstack((D, L)), y)
    preds = lr.predict_proba(np.hstack((D, L)))
    cv_score = -log_loss(y, preds)

    return (cv_score, sum([len(x) for x in shapelets]))

class ErrorDistribution:
    def __init__(self, distance_function= 'mannwhitneyu'):
        if distance_function == 'mannwhitneyu':
            self.distance_function = mannwhitneyu
        elif distance_function == 'wasserstein':
            self.distance_function = wasserstein_distance

        self.distance_function = distance_function

    def distance(X, y, shapelets, dist_threshold, cache=None, verbose=False):
        D = np.zeros((len(X), len(shapelets)))
        L = np.zeros((len(X), len(shapelets)))

        # First check if we already calculated distances for a shapelet
        for shap_ix, shap in enumerate(shapelets):
            shap_hash = hash(tuple(shap.flatten()))
            cache_val = cache.get(shap_hash)
            if cache_val is not None:
                D[:, shap_ix] = cache_val

        # Fill up the 0 entries
        _pdist_location(X, [shap.flatten() for shap in shapelets], D, L)

        # Fill up our cache
        for shap_ix, shap in enumerate(shapelets):
            shap_hash = hash(tuple(shap.flatten()))
            cache.set(shap_hash, D[:, shap_ix])

        subgroup_filter = np.all(D > dist_threshold, axis=1)
        subgroup_y = y[subgroup_filter]
        rest_y = y[~subgroup_filter]
        dist = self.distance_function(subgroup_y, rest_y)
        return (dist, sum([len(x) for x in shapelets]))
 
import numpy as np

class DistributionDistance:
    def __init__(self, distance_function):
        self.distance_function = distance_function

    def _calculate_shapelet_dist_matrix(X, y, shapelets, cache=None, verbose=False):
        D = np.zeros((len(X), len(shapelets)))
        L = np.zeros((len(X), len(shapelets)))

        # First check if we already calculated distances for a shapelet
        for shap_ix, shap in enumerate(shapelets):
            shap_hash = hash(tuple(shap.flatten()))
            cache_val = cache.get(shap_hash)
            if cache_val is not None:
                D[:, shap_ix] = cache_val

        # Fill up the 0 entries
        _pdist_location(X, [shap.flatten() for shap in shapelets], D, L)

        # Fill up our cache
        for shap_ix, shap in enumerate(shapelets):
            shap_hash = hash(tuple(shap.flatten()))
            cache.set(shap_hash, D[:, shap_ix])

        return D

    def distance(X, y, shapelets, dist_threshold, cache=None, verbose=False):
        D = self._calculate_shapelet_dist_matrix(X, y, shapelets, cache, verbose)
        subgroup_filter = np.all(D > dist_threshold, axis=1)
        subgroup_y = y[subgroup_filter]
        rest_y = y[~subgroup_filter]
        dist = self.distance_function(subgroup_y, rest_y)
        return (dist, sum([len(x) for x in shapelets]))
 
    @staticmethod
    def wasserstein_distance(y1, y2):
        return wasserstein_distance(y1, y2)

    @staticmethod
    def mannwhitneyu(y1, y2):
        return mannwhitneyu(y1, y2)
