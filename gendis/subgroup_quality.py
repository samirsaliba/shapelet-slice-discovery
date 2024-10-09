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
        self.alpha = 1.0
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

    def get_threshold_filter(self, distances, threshold):
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
        return distances < threshold

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
        subgroup = self.get_threshold_filter(dists, threshold)
        subgroup_y = y[subgroup]

        # Compute the fitness (e.g., based on distribution delta and size factor)
        distribution_delta = self.distance_function(subgroup_y, y)
        sizeW = self.subgroup_size_factor(len(subgroup_y), len(y))
        shap_distance_factor = np.median(dists[subgroup])

        fitness = self.alpha * (distribution_delta * sizeW) - (1-self.alpha)*shap_distance_factor

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
        Compute the optimal threshold for a given shapelet using binary search.
        """
        cache_threshold = self.cache.get(shapelet.id)
        if cache_threshold is not None:
            return cache_threshold

        # Start with percentiles as the initial range for the threshold
        percentiles = [0, 100]
        lower_bound, upper_bound = np.percentile(dists, percentiles)
        
        # Set initial best threshold to the middle of the range
        search_best_threshold = (lower_bound + upper_bound) / 2
        search_best_fitness = self.compute_fitness_for_shapelet(dists, y, shapelet, search_best_threshold)
        global_best_threshold = search_best_threshold
        global_best_fitness = search_best_fitness
        
        # Binary search parameters
        max_iterations = self.max_it
        tolerance = np.min(np.diff(np.sort(np.unique(dists))))  # Stop when the difference is smaller than this
        stagnation_counter = 0
        max_stagnation = 5  # Restart search after 5 non-improving iterations
                
        for iteration in range(max_iterations):
            # Compute the mid-point between lower and upper bounds
            mid_point = (lower_bound + upper_bound) / 2
            
            # Compute fitness for mid-point threshold
            fitness_mid = self.compute_fitness_for_shapelet(dists, y, shapelet, mid_point)
            
            # Compare mid-point fitness with best so far
            if fitness_mid > search_best_fitness:
                search_best_threshold = mid_point
                search_best_fitness = fitness_mid
                stagnation_counter = 0  # Reset stagnation counter

                if search_best_fitness > global_best_fitness:
                    global_best_fitness = search_best_fitness
                    global_best_threshold = search_best_threshold
            else:
                stagnation_counter += 1
            
            # Adjust bounds based on comparison with best threshold
            if mid_point < search_best_threshold:
                lower_bound = mid_point
            else:
                upper_bound = mid_point
            
            # Check for convergence or stagnation
            if upper_bound - lower_bound < tolerance or stagnation_counter >= max_stagnation:
                # Optionally restart search with random perturbation
                if stagnation_counter >= max_stagnation:                
                    # Introduce a random jump within the current bounds
                    random_jump = np.random.uniform(lower_bound, upper_bound)
                    random_jump_fitness = self.compute_fitness_for_shapelet(dists, y, shapelet, random_jump)
                    search_best_threshold = random_jump
                    search_best_fitness = random_jump_fitness
                    stagnation_counter = 0  # Reset stagnation counter after the jump

                    # Perturb the bounds slightly to explore a new region
                    bound_perturb_factor = 0.1  # Modify as needed to control the scale of perturbation
                    lower_bound = max(0, lower_bound + bound_perturb_factor * (upper_bound - lower_bound))
                    upper_bound = min(1, upper_bound - bound_perturb_factor * (upper_bound - lower_bound))

                else:
                    break

        self.cache.set(shapelet.id, global_best_threshold)
        shapelet.threshold = global_best_threshold
        return global_best_threshold

    def get_shapelet_subgroup(self, shap, distances, y):
        threshold = self.get_optimal_threshold_shapelet(shap, distances, y)
        subgroup = self.get_threshold_filter(distances, threshold)

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
        fitness =  distribution_delta * sizeW
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
