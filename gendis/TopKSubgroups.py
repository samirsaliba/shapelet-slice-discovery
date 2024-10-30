import copy
from functools import partial
import logging
import numpy as np


class TopKSubgroups:
    """
    A class to manage the top-k subgroups in a genetic algorithm population, balancing fitness and coverage diversity.

    Attributes:
        k (int): The number of top subgroups to maintain.
        coverage_alpha (float): A parameter controlling the importance of coverage when reweighting instances.
        subgroups (list): A list of the current top-k subgroups.
        last_update (int): Iteration at which the top-k subgroups were last updated.
        coverage (np.ndarray): An array tracking how many times each instance has been covered by the top-k subgroups.
        ids (set): A set of unique identifiers for individuals in the top-k subgroups.
    """

    def __init__(self, k, coverage_alpha):
        """
        Initializes the TopKSubgroups object with parameters to maintain the top-k population.

        Args:
            k (int): The number of top-k subgroups to track.
            coverage_alpha (float): The alpha value controlling the contribution of each instance's coverage to subgroup selection.
        """
        self.k = k
        self.coverage_alpha = coverage_alpha
        self.subgroups = None
        self.coverage = None
        self.ids = None
        self.last_update = 1

    def _update_coverage(self, subgroup, coverage):
        """
        Updates the coverage array when a new subgroup is added.

        Args:
            subgroup (np.ndarray): A boolean array representing the instances covered by the subgroup.
            coverage (np.ndarray): An array tracking how many times each instance has been covered by subgroups.

        Returns:
            tuple: A tuple containing updated weights and the modified coverage array.
        """
        coverage[subgroup] += 1
        base = [self.coverage_alpha]
        return np.power(base, coverage), coverage

    def _coverage_factor(self, subgroup, weights):
        """
        Calculates the weighted coverage score for a subgroup.

        Args:
            weights (np.ndarray): The weights associated with instance coverage.
            subgroup (np.ndarray): A boolean array representing the instances covered by the subgroup.

        Returns:
            float: The weighted coverage score for the subgroup.
        """
        in_sg_weights = weights[subgroup].sum()
        sg_weights_total = subgroup.sum() * self.coverage_alpha
        return in_sg_weights / sg_weights_total

    def update(self, pop, it, length_input, toolbox):
        """
        Updates the top-k population by selecting the k most diverse and fit individuals.

        Args:
            pop (list): The current population of individuals.
            it (int): The current iteration of the genetic algorithm.
            length_input (list): A list representing the length of input data.
        """
        coverage = np.ones(length_input)
        weights = np.power([self.coverage_alpha], coverage)
        # pop = list(map(copy.deepcopy, pop))

        if self.subgroups is None:
            pop_star = pop
            self.ids = []
            self.coverage = np.array([])
        else:
            pop_star = self.subgroups + pop

        best = copy.deepcopy(max(pop_star, key=lambda ind: ind.fitness.values[0]))
        best.coverage_weight = 1.0
        new_top_k = [best]
        new_ids = set([best.uuid])

        k = 0
        while k < self.k:
            weights, coverage = self._update_coverage(best.subgroup, coverage)

            fitness_values = []
            coverage_factors = []

            # for ind in pop_star:
            #     fitness_values.append(ind.fitness.values[0])
            #     coverage_factors.append(self._coverage_factor(weights, ind.subgroup))

            fitness_values, subgroups = zip(
                *[(x.fitness.values[0], x.subgroup) for x in pop_star]
            )
            fitness_values, subgroups = list(fitness_values), list(subgroups)

            _cov_func = partial(self._coverage_factor, weights=weights)
            coverage_factors = list(toolbox.map(_cov_func, subgroups))

            fitness_values = np.array(fitness_values)
            coverage_factors = np.array(coverage_factors)
            weighted_scores = fitness_values * coverage_factors

            found_new_best = False
            while pop_star:
                max_index = np.argmax(weighted_scores)
                best = pop_star[max_index]
                best_coverage = coverage_factors[max_index]

                weighted_scores = np.delete(weighted_scores, max_index)
                coverage_factors = np.delete(coverage_factors, max_index)
                pop_star.pop(max_index)

                if best.uuid not in new_ids:
                    found_new_best = True
                    break

            if found_new_best:
                best = copy.deepcopy(best)
                best.coverage_weight = best_coverage
                new_top_k.append(best)
                new_ids.add(best.uuid)
                k += 1
            else:
                logging.info("No more unique individuals to add to top-k.")
                break

        if not np.array_equal(coverage, self.coverage):
            self.last_update = it

        self.coverage = coverage
        self.subgroups = new_top_k
        self.ids = new_ids

    def to_dict(self):
        """
        Converts the top-k subgroups and relevant information into a dictionary format.

        Returns:
            list: A list containing the top-k subgroups in a JSON friendly format
        """
        return [ind.to_dict() for ind in self.subgroups]
