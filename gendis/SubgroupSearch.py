import numpy as np
from scipy.stats import wasserstein_distance, mannwhitneyu
from sklearn.preprocessing import StandardScaler

from .LRUCache import LRUCache


class SubgroupSearch:
    """TODO"""

    def __init__(
        self,
        distance_function,
        sg_size_beta,
        standardize=False,
        max_it=100,
    ):
        self.alpha = 0.8
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

    def compute_fitness_for_shapelet(self, dists, y, threshold):
        """
        Compute the fitness for a single shapelet based on its threshold.

        Parameters
        ----------
        dists : array-like
            Distance matrix column.
        y : array-like
            Target variable.
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

        fitness = (
            self.alpha * (distribution_delta * sizeW)
            - (1 - self.alpha) * shap_distance_factor
        )

        return fitness

    def compute_gradient_threshold(self, dists, y, shapelet, threshold, epsilon):
        """
        Compute the gradient of the fitness function with respect to a single shapelet's threshold.
        """
        # Compute perturbed thresholds
        thresh_plus = threshold + epsilon
        thresh_minus = threshold - epsilon

        # Compute fitness for perturbed thresholds
        fitness_plus = self.compute_fitness_for_shapelet(dists, y, thresh_plus)
        fitness_minus = self.compute_fitness_for_shapelet(dists, y, thresh_minus)

        # Approximate gradient using finite differences
        grad_threshold = (fitness_plus - fitness_minus) / (2 * epsilon)

        return grad_threshold

    def get_optimal_threshold_shapelet(self, shapelet, dists, y):
        """
        Compute the optimal threshold for a given shapelet using multiple runs of Simulated Annealing.
        """
        cache_threshold = self.cache.get(shapelet.id)
        if cache_threshold is not None:
            return cache_threshold

        # Start with percentiles as the initial range for the threshold
        percentiles = [0, 100]
        lower_bound, upper_bound = np.percentile(dists, percentiles)

        # Simulated annealing parameters
        perturbation = 0.05 * (upper_bound - lower_bound)
        num_restarts = 5  # Number of independent Simulated Annealing runs
        max_iterations = self.max_it // num_restarts
        initial_temperature = 1.0
        min_temperature = 0.001  # Stopping criterion for the cooling process
        cooling_rate = 0.95  # Cooling rate to decrease the temperature

        global_best_threshold = None
        global_best_fitness = -np.inf

        # Multiple Simulated Annealing runs
        for _ in range(num_restarts):
            # Set the starting point for the current run (random point within the range)
            current_threshold = np.random.uniform(lower_bound, upper_bound)
            current_fitness = self.compute_fitness_for_shapelet(
                dists, y, current_threshold
            )
            best_threshold = current_threshold
            best_fitness = current_fitness

            temperature = initial_temperature

            # Single run of Simulated Annealing
            for _ in range(max_iterations):
                if temperature < min_temperature:
                    break

                # Find a new point in the vicinity
                new_threshold = current_threshold + np.random.uniform(
                    -perturbation, perturbation
                )
                new_threshold = np.clip(new_threshold, lower_bound, upper_bound)
                new_fitness = self.compute_fitness_for_shapelet(dists, y, new_threshold)

                # Determine if the new threshold should be accepted
                fitness_difference = new_fitness - current_fitness
                if fitness_difference > 0:
                    # If the new solution is better, accept it
                    current_threshold = new_threshold
                    current_fitness = new_fitness
                else:
                    # If the new solution is worse, accept it with a probability depending on temperature
                    acceptance_probability = np.exp(fitness_difference / temperature)
                    if np.random.rand() < acceptance_probability:
                        current_threshold = new_threshold
                        current_fitness = new_fitness

                # Track the best solution in the current run
                if current_fitness > best_fitness:
                    best_threshold = current_threshold
                    best_fitness = current_fitness

                # Reduce the temperature (cooling)
                temperature *= cooling_rate

            # Track the global best solution across all runs
            if best_fitness > global_best_fitness:
                global_best_threshold = best_threshold
                global_best_fitness = best_fitness

        # Cache the best solution across all restarts
        self.cache.set(shapelet.id, global_best_threshold)
        try:
            assert global_best_threshold is not None, "None threshold"
        except Exception as e:
            print(shapelet.id)
            print(current_fitness)
            print(best_fitness)
            print(num_restarts, max_iterations, temperature)
            th = np.percentile(dists, 10)
            ft = self.compute_fitness_for_shapelet(dists, y, global_best_threshold)
            print(th, ft)
            raise (e)

        return global_best_threshold

    def get_shapelet_subgroup(self, shap, distances, y):
        threshold = self.get_optimal_threshold_shapelet(shap, distances, y)
        shap.threshold = threshold
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
        fitness = distribution_delta * sizeW
        sg_mean = np.mean(subgroup_y)
        return {
            "valid": subgroup_n > 0,
            "value": np.array([fitness]),
            "subgroup": subgroup,
            "thresholds": [thresholds],
            "info": {
                "fitness": fitness,
                "distribution_delta": distribution_delta,
                "subgroup_size_weight": sizeW,
                "subgroup_error_mean": sg_mean,
                "population_mean": np.mean(y),
                "subgroup_size": subgroup_n,
                "thresholds": [thresholds],
            },
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
