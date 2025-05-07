import numpy as np
from operator import itemgetter
from scipy.stats import wasserstein_distance, mannwhitneyu
from .LRUCache import LRUCache


class SubgroupSearch:
    """TODO"""

    threshold_options = ["simulated_annealing", "percentile"]

    def __init__(
        self,
        distance_function,
        threshold_search_mode,
        sg_size_beta,
        max_it=100,
        threshold_kappa=0.75,
        cache_size=4096,
        random_seed=None,
    ):
        self.distance_function = distance_function

        assert (
            threshold_search_mode in self.threshold_options
        ), f"Threshold mode not implemented. Available modes {self.threshold_options}"

        if threshold_search_mode == "simulated_annealing":
            self.threshold_search_function = self.simulated_annealing_search

        elif threshold_search_mode == "percentile":
            self.threshold_search_function = self.percentile_search

        self.sg_size_beta = sg_size_beta
        self.max_it = max_it
        self.kappa = threshold_kappa
        self.random_seed = random_seed
        self.np_random = np.random.default_rng(random_seed)

        self.cache = LRUCache(cache_size)

    def __call__(self, D, y, shaps):
        return self.evaluate(D, y, shapelets=shaps)

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
        return distances <= threshold

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
        shap_distance_factor = np.median(dists[subgroup]) / np.median(dists)

        fitness = (
            self.kappa * (distribution_delta * sizeW)
            - (1 - self.kappa) * shap_distance_factor
        )

        return fitness

    def percentile_search(self, shapelet, distances, y):
        if self.cache:
            cache_threshold = self.cache.get(shapelet.id)
            if cache_threshold is not None:
                return cache_threshold

        percentiles = np.percentile(distances, np.arange(0, 100))

        fitness_func = lambda threshold: self.compute_fitness_for_shapelet(
            distances, y, threshold
        )
        fitnesses = list(map(fitness_func, percentiles))
        index, fitness = max(enumerate(fitnesses), key=itemgetter(1))
        best_threshold = percentiles[index]

        if self.cache:
            self.cache.set(shapelet.id, best_threshold)
        return best_threshold

    def simulated_annealing_search(self, shapelet, distances, y):
        """
        Compute the optimal threshold for a given shapelet using multiple runs of Simulated Annealing.
        """
        if self.cache:
            cache_threshold = self.cache.get(shapelet.id)
            if cache_threshold is not None:
                return cache_threshold

        # Start with percentiles as the initial range for the threshold
        percentiles = [0, 100]
        lower_bound, upper_bound = np.percentile(distances, percentiles)

        # Simulated annealing parameters
        perturbation = 0.05 * (upper_bound - lower_bound)
        num_restarts = 5  # Number of independent Simulated Annealing runs
        max_iterations = self.max_it // num_restarts
        initial_temperature = 1.0
        min_temperature = 0.001  # Stopping criterion for the cooling process
        cooling_rate = 0.95  # Cooling rate to decrease the temperature

        # "Educated" first guess of 10% percentile
        global_best_threshold = np.percentile(distances, 10)
        global_best_fitness = self.compute_fitness_for_shapelet(
            distances, y, global_best_threshold
        )

        # Multiple Simulated Annealing runs
        for _ in range(num_restarts):
            # Set the starting point for the current run (random point within the range)
            current_threshold = self.np_random.uniform(lower_bound, upper_bound)
            current_fitness = self.compute_fitness_for_shapelet(
                distances, y, current_threshold
            )
            best_threshold = current_threshold
            best_fitness = current_fitness

            temperature = initial_temperature

            # Single run of Simulated Annealing
            for _ in range(max_iterations):
                if temperature < min_temperature:
                    break

                # Find a new point in the vicinity
                new_threshold = current_threshold + self.np_random.uniform(
                    -perturbation, perturbation
                )
                new_threshold = np.clip(new_threshold, lower_bound, upper_bound)
                new_fitness = self.compute_fitness_for_shapelet(
                    distances, y, new_threshold
                )

                # Determine if the new threshold should be accepted
                fitness_difference = new_fitness - current_fitness
                if fitness_difference > 0:
                    # If the new solution is better, accept it
                    current_threshold = new_threshold
                    current_fitness = new_fitness
                else:
                    # If the new solution is worse, accept it with a probability depending on temperature
                    acceptance_probability = np.exp(fitness_difference / temperature)
                    if self.np_random.random() < acceptance_probability:
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
        if self.cache:
            self.cache.set(shapelet.id, global_best_threshold)
        return global_best_threshold

    def get_shapelet_subgroup(self, shap, distances, y):
        threshold = self.threshold_search_function(shap, distances, y)
        shap.threshold = threshold
        subgroup = self.get_threshold_filter(distances, threshold)

        assert sum(subgroup) > 0, "Shapelet covering zero instances"
        return threshold, subgroup

    def fit(self, shapelets, D, y):
        thresholds = []
        subgroups = []
        for i, shap in enumerate(shapelets):
            distances = D[:, i]
            threshold, subgroup = self.get_shapelet_subgroup(shap, distances, y)
            thresholds.append(threshold)
            subgroups.append(subgroup)

        subgroup = np.all(subgroups, axis=0)
        return subgroup, thresholds

    def transform(self, D, thresholds):
        subgroups = []
        for i, threshold in enumerate(thresholds):
            distances = D[:, i]
            subgroup = self.get_threshold_filter(distances, threshold)
            subgroups.append(subgroup)

        return np.all(subgroups, axis=0)

    def evaluate(self, D, y, shapelets):
        subgroup, thresholds = self.fit(shapelets, D, y)
        subgroup_y = y[subgroup]
        subgroup_n = np.sum(subgroup)

        if subgroup_n < 1:
            for shap_i, shap in enumerate(shapelets):
                print(shap)
                print(shap.id, shap.index, shap.start, len(shap))
                print(f"threshold={thresholds[shap_i]}")
                print(
                    f"self-distance={D[shap.index, shap_i]}, min-dist={min(D[:, shap_i])}"
                )

            print(shapelets.op_history)
            raise ValueError("Individual covering zero instances")

        distribution_delta = self.distance_function(subgroup_y, y)
        sizeW = self.subgroup_size_factor(subgroup_n, len(y))
        fitness = distribution_delta * sizeW
        sg_mean = np.mean(subgroup_y)
        return {
            "valid": subgroup_n > 0,
            "value": np.array([fitness]),
            "subgroup": subgroup,
            "subgroup_size": subgroup_n,
            "thresholds": thresholds,
            "info": {
                "fitness": fitness,
                "distribution_delta": distribution_delta,
                "subgroup_size": subgroup_n,
                "subgroup_error_mean": sg_mean,
                "population_mean": np.mean(y),
                "subgroup_size_weight": sizeW,
                "thresholds": thresholds,
            },
        }

    @staticmethod
    def compile_pop_stats(pop, it, run_id=None):
        fitnesses = []
        sizes = []
        target_distribution_shift_values = []

        for ind in pop:
            fitnesses.append(ind.fitness.values[0])
            sizes.append(ind.subgroup_size)
            target_distribution_shift_values.append(ind.info["distribution_delta"])

        stats = {
            "run_id": run_id,
            "it": it,
            "fitness_mean": np.mean(fitnesses),
            "fitness_std": np.std(fitnesses),
            "fitness_max": np.max(fitnesses),
            "fitness_min": np.min(fitnesses),
            "size_mean": np.mean(sizes),
            "size_std": np.std(sizes),
            "size_max": np.max(sizes),
            "size_min": np.min(sizes),
            "target_distribution_mean": np.mean(target_distribution_shift_values),
            "target_distribution_std": np.std(target_distribution_shift_values),
            "target_distribution_max": np.max(target_distribution_shift_values),
            "target_distribution_min": np.min(target_distribution_shift_values),
        }

        return stats

    @staticmethod
    def wasserstein_distance(y1, y2):
        return wasserstein_distance(y1.reshape(-1), y2.reshape(-1))

    @staticmethod
    def mannwhitneyu(y1, y2):
        return 1 - mannwhitneyu(y1.reshape(-1), y2.reshape(-1)).pvalue

    @staticmethod
    def simple_mean(y1, y2):
        return np.absolute(np.mean(y1.reshape(-1)) - np.mean(y2.reshape(-1)))
