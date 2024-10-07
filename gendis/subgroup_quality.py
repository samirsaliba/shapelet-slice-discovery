import numpy as np
from scipy.stats import wasserstein_distance, mannwhitneyu
from sklearn.preprocessing import StandardScaler

try:
    from LRUCache import LRUCache
except:
    from gendis.LRUCache import LRUCache

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
        sg_size_beta,
        standardize=False,
        max_it = 100,
    ):  
        self.cache = LRUCache(2048)
        self.sg_size_beta = sg_size_beta
        self.distance_function = distance_function
        self.standardize = standardize
        self.max_it = max_it

    def __call__(self, D, y, shaps=None):
        return self.evaluate(D, y, shaps=shaps)

    def subgroup_size_factor(self, subgroup_n, y_n):
        """Score that favors larger subgroups"""
        return (subgroup_n / y_n) ** self.sg_size_beta

    def apply_sigmoid_filter(self, distances, threshold):
        """
        Parameters
        ----------
        distances : array-like
            A 1D array of distances for each instance to a specific shapelet.
        threshold : float
            The threshold for the distance.

        Returns
        -------
        subgroup : array-like
            Boolean array indicating whether each instance i belongs to the subgroup,
            considering the distance to this specific shapelet.
        """        
        # Apply the sigmoid function to the distances
        belonging_prob = 1 / (1 + np.exp(-(distances - threshold)))

        # Return True if the sigmoid output is greater than or equal to 0.5 
        # meaning within the subgroup
        return belonging_prob > 0.5

    def compute_fitness_for_shapelet(self, dists, y, shapelet, threshold):
        """
        Compute the fitness for a single shapelet based on its threshold.

        Parameters
        ----------
        dists : array-like
            Distance matrix column.
        y : array-like
            Target variable.
        shapelet : array-like
            The shapelet for which fitness is being computed.
        threshold : float
            The threshold used for filtering the subgroup.

        Returns
        -------
        fitness : float
            The fitness value for the shapelet and threshold.
        """
        # Filter the subgroup using the threshold for this shapelet
        subgroup = self.apply_sigmoid_filter(dists, threshold)
        subgroup_y = y[subgroup]

        # Compute the fitness (e.g., based on distribution delta and size factor)
        distribution_delta = self.distance_function(subgroup_y, y)
        sizeW = self.subgroup_size_factor(len(subgroup_y), len(y))
        fitness = distribution_delta * sizeW

        return fitness

    def compute_gradient_threshold(self, dists, y, shapelet, threshold, epsilon):
        """
        Compute the gradient of the fitness function with respect to a single shapelet's threshold.
        """
        # Compute perturbed thresholds
        thresh_plus = threshold + epsilon
        thresh_minus = threshold - epsilon

        # Compute fitness for perturbed thresholds
        fitness_plus = self.compute_fitness_for_shapelet(dists, y, shapelet, thresh_plus)
        fitness_minus = self.compute_fitness_for_shapelet(dists, y, shapelet, thresh_minus)

        # Approximate gradient using finite differences
        grad_threshold = (fitness_plus - fitness_minus) / (2 * epsilon)

        return grad_threshold

    def get_optimal_threshold_shapelet(self, shapelet, dists, y):
        """
        Compute the optimal threshold for a given shapelet using gradient descent with momentum.
        """
        cache_threshold = self.cache.get(shapelet.id)
        if cache_threshold is not None:
            return cache_threshold

        # Start with percentiles as candidates for the initial threshold
        percentiles = [10, 20, 30, 50, 70, 80, 90]
        initial_thresholds = np.percentile(dists, percentiles)
        fitness_values = [
            self.compute_fitness_for_shapelet(dists, y, shapelet, t) 
            for t in initial_thresholds
        ]

        # Choose the threshold with the best fitness value as the starting point
        threshold = initial_thresholds[np.argmax(fitness_values)]

        # Perform gradient descent to find the optimal threshold
        learning_rate = 5 * np.min(np.diff(np.sort(np.unique(dists))))  # Larger initial rate
        momentum = 0.9  # Momentum factor
        velocity = 0  # Initial velocity for momentum
        epsilon = np.min(np.diff(np.sort(np.unique(dists))))

        for iteration in range(self.max_it):
            # Compute the fitness and gradient for the current threshold
            grad_threshold = self.compute_gradient_threshold(dists, y, shapelet, threshold, epsilon)

            # Update the velocity with momentum
            velocity = momentum * velocity - learning_rate * grad_threshold

            # Update the threshold
            threshold += velocity

            # Optionally reduce learning rate if oscillation is detected
            if np.sign(grad_threshold) != np.sign(velocity):
                learning_rate *= 0.5

        self.cache.set(shapelet.id, threshold)
        return threshold

    def get_shapelet_subgroup(self, shap, distances, y):
        threshold = self.get_optimal_threshold_shapelet(shap, distances, y)
        shap.threshold = threshold
        subgroup = self.apply_sigmoid_filter(distances, threshold)

        return threshold, subgroup

    def get_set_subgroup(self, shapelets, D, y):
        thresholds = []
        subgroups = []
        for i, shap in enumerate(shapelets):
            dists = D[:, i]
            threshold, subgroup = self.get_shapelet_subgroup(shap, dists, y)
            thresholds.append(threshold)
            subgroups.append(subgroup)

        subgroup = np.all(subgroups, axis=0)
        return subgroup, thresholds

    def evaluate(self, D, y, shaps):
        subgroup, thresholds = self.get_set_subgroup(shaps, D, y)
        subgroup_y = y[subgroup]
        subgroup_n = np.sum(subgroup)

        distribution_delta = self.distance_function(subgroup_y, y)
        sizeW = self.subgroup_size_factor(subgroup_n, len(y))
        fitness =  alpha * (distribution_delta * sizeW) - (1-alpha)*sum_distances # TODO
        sg_mean = np.mean(subgroup_y)
        return {
            'valid': subgroup_n>0,
            'value': np.array([fitness]),
            'subgroup': subgroup,
            'info': {
                'fitness': fitness,
                'distribution_delta': distribution_delta,
                'shaps_distances_thresholds': thresholds,
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
