import copy
from deap import base, creator, tools
import logging
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import time
import torch
import warnings

# Ensure 'spawn' start method for compatibility, especially on some systems
mp.set_start_method("spawn", force=True)

warnings.filterwarnings("ignore")

from .individual import ShapeletIndividual
from .LRUCache import LRUCache
from .operators import (
    random_shapelet,
    crossover_AND,
    crossover_uniform,
    add_shapelet,
    remove_shapelet,
    replace_shapelet,
    smooth_shapelet,
)
from .shapelets_distances import calculate_shapelet_dist_matrix


class GeneticExtractor(BaseEstimator, TransformerMixin):
    """Feature selection with genetic algorithm.

    Parameters
    ----------
    TODO
    """

    def __init__(
        self,
        subgroup_search,
        top_k,
        population_size,
        iterations,
        mutation_prob,
        crossover_prob,
        wait=10,
        pop_restarts=5,
        max_shaps=None,
        max_len=None,
        min_len=0,
        init_ops=[random_shapelet],
        cx_ops=[crossover_AND, crossover_uniform],
        mut_ops=[add_shapelet, smooth_shapelet],
        cache_size=4096,
        n_jobs=1,
        verbose=False,
        random_seed=None,
    ):
        self.subgroup_search = subgroup_search
        self.top_k = top_k

        # Hyper-parameters
        self.population_size = population_size
        self.iterations = iterations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.wait = wait
        self.pop_restarts = pop_restarts
        self.max_shaps = max_shaps
        self.min_len = min_len
        self.max_len = max_len
        self.init_ops = init_ops
        self.cx_ops = cx_ops
        self.mut_ops = mut_ops
        self.cache_size = cache_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_seed = random_seed
        self.np_random = np.random.default_rng(random_seed)

        # Attributes
        self.is_fitted = False
        self.label_mapping = {}
        self.should_reset_pop = False
        self.pop_restart_counter = 0
        self.plot = False

    def _print_statistics(self, stats, start):
        if self.it == 1:
            # Print the header of the statistics
            print("it\t\tavg\t\tstd\t\tmax\t\ttime")
            # print('it\t\tavg\t\tmax\t\ttime')

        print(
            "{}\t\t{}\t\t{}\t\t{}\t{}".format(
                # print('{}\t\t{}\t\t{}\t{}'.format(
                self.it,
                np.around(stats["avg"], 4),
                np.around(stats["std"], 3),
                np.around(stats["max"], 6),
                np.around(time.time() - start, 4),
            )
        )

    def _create_individual(self, n_shapelets=None):
        """Generate a random shapelet set"""
        n_shapelets = 1
        init_op = self.np_random.choice(self.init_ops)
        return init_op(
            X=self.X,
            n_shapelets=n_shapelets,
            max_len=self.max_len,
            min_len=self.min_len,
            np_random=self.np_random,
        )

    def _eval_individual(self, ind):
        """Evaluate the fitness of an individual"""
        D, L = calculate_shapelet_dist_matrix(
            self.X_tensor,
            ind,
            cache=self.cache,
            device=self.device,
        )
        fit = self.subgroup_search(D=D, y=self.y, shaps=ind, L=L)
        ind.register_op("eval")
        ind.fitness.values = fit["value"]
        ind.valid = fit["valid"]
        ind.subgroup = fit["subgroup"]
        ind.subgroup_size = fit["subgroup_size"]
        ind.thresholds = fit["thresholds"]
        ind.info = fit["info"]
        return fit

    def _mutate_individual(self, ind):
        """Mutate an individual"""
        assert (
            len(ind) > 0
        ), "_mutate_individual requires an individual with at least one shapelet"
        if self.np_random.random() < self.mutation_prob:
            clone = copy.deepcopy(ind)
            mut_op = self.np_random.choice(self.mut_ops)

            if (
                mut_op.__name__ == "add_shapelet" and len(clone) >= self.max_shaps
            ) or clone.subgroup_size == 1:
                mut_op = replace_shapelet

            return mut_op(
                X=self.X,
                individual=clone,
                min_len=self.min_len,
                max_len=self.max_len,
                np_random=self.np_random,
            )
        return ind

    def _cross_individuals(self, ind1, ind2, toolbox):
        """Cross two individuals"""
        if self.np_random.random() < self.crossover_prob:
            ind1_clone = toolbox.clone(ind1)
            ind2_clone = toolbox.clone(ind2)

            cx_op = self.np_random.choice(self.deap_cx_ops)
            child1, _ = cx_op(
                ind1=ind1_clone, ind2=ind2_clone, np_random=self.np_random
            )
            child1.uuid_history.append(ind1_clone.uuid)
            child1.register_op("cx-child")
            ind1_clone.register_op("cx-parent")

            return child1, ind1_clone

        return ind1, ind2

    def _check_early_stopping(self):
        if self.it - self.top_k.last_update > self.wait:
            if self.pop_restart_counter < self.pop_restarts:
                # Will trigger population reset
                self.top_k.last_update = self.it
                self.should_reset_pop = True
            else:
                return True
        return False

    def _create_pop(self, toolbox):
        # Initialize the population and calculate their initial fitness values
        self.should_reset_pop = False
        self.pop_restart_counter += 1

        self.best = {
            "it": self.it,
            "score": float("-inf"),
            "info": None,
            "shapelets": [],
        }
        pop = toolbox.population(n=self.population_size)
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        return pop

    def fit(self, X, y):
        """Extract shapelets from the provided timeseries and labels.

        Parameters
        ----------
        X : array-like, shape = [n_ts, ]
            The training input timeseries. Each timeseries must be an array,
            but the lengths can be variable

        y : array-like, shape = [n_samples]
            The target values.
        """
        random_arr = self.np_random.integers(0, 100, size=10)
        logging.info(f"Random seed ({self.random_seed}) np check: {random_arr}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using torch:{self.device}")

        self.X = X
        self.X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = y

        if self.max_len is None:
            if len(self.X[0]) > 20:
                self.max_len = len(self.X[0]) // 2
            else:
                self.max_len = len(self.X[0])

        self.cache = LRUCache(self.cache_size)
        self.history = []

        creator.create("FitnessMax", base.Fitness, weights=[1.0])
        creator.create("Individual", ShapeletIndividual, fitness=creator.FitnessMax)

        # Register all operations in the toolbox
        toolbox = base.Toolbox()
        toolbox.register("clone", copy.deepcopy)

        if self.n_jobs > 1:
            raise NotImplementedError
            pool = mp.Pool(self.n_jobs)
            torch.multiprocessing.set_sharing_strategy("file_system")
            toolbox.register("map", pool.map)

        else:
            toolbox.register("map", map)

        self.deap_cx_ops = []
        for i, cx_op in enumerate(self.cx_ops):
            toolbox.register(f"cx{i}", cx_op)
            self.deap_cx_ops.append(getattr(toolbox, (f"cx{i}")))

        self.deap_mut_ops = []
        for i, mut_op in enumerate(self.mut_ops):
            toolbox.register(f"mutate{i}", mut_op)
            self.deap_mut_ops.append(getattr(toolbox, (f"mutate{i}")))

        toolbox.register("create", self._create_individual)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.create
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._eval_individual)
        # Small tournaments to ensure diversity
        toolbox.register("select", tools.selTournament, tournsize=2)

        # Set up the statistics. We will measure the mean, std dev and max
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)
        stats.register("min", np.min)

        self.it = 1
        pop = self._create_pop(toolbox)

        # The genetic algorithm starts here
        it_start = time.time()
        while self.it <= self.iterations:
            logging.info(f"it:{self.it}, it_time: {(time.time() - it_start):.2f}s")
            it_start = time.time()

            # Early stopping and pop reset
            if self._check_early_stopping():
                break

            if self.should_reset_pop:
                logging.info(
                    f"Restarting pop {self.pop_restart_counter+1}/{self.pop_restarts}"
                )
                pop = self._create_pop(toolbox)

            # Elitism
            elit_n = 1
            elitism_individuals = [toolbox.clone(x) for x in tools.selBest(pop, elit_n)]
            offspring = pop

            # Crossover
            if len(self.deap_cx_ops) > 0:
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    self._cross_individuals(child1, child2, toolbox)

            # Mutation
            if len(self.deap_mut_ops) > 0:
                offspring = list(toolbox.map(self._mutate_individual, offspring))

            invalid = [ind for ind in offspring if not ind.valid]
            _ = list(toolbox.map(toolbox.evaluate, invalid))

            # Replace population and update hall of fame, statistics & history
            new_pop = toolbox.select(offspring, self.population_size - elit_n)
            pop = new_pop + elitism_individuals
            it_stats = stats.compile(pop)
            self.history.append([self.it, it_stats])

            # Have we found a new best score?
            if it_stats["max"] > self.best["score"]:
                best = tools.selBest(pop + offspring, 1)[0]
                logging.info(f"Best ind: {best.info}")
                logging.info(f"Best ind covers {sum(best.subgroup)} instances")

                self.best = {
                    "it": self.it,
                    "score": best.fitness.values[0],
                    "info": best.info,
                    "shapelets": best,
                }

            self.top_k.update(pop, self.it, len(self.X), toolbox)
            self.it += 1

        self.pop = pop
        self.is_fitted = True
        del self.X, self.X_tensor, self.y

    def transform(self, X, shapelets, thresholds, return_positions=True):
        """After fitting the Extractor, we can transform collections of
        timeseries in matrices with distances to each of the shapelets in
        the evolved shapelet set.

        Parameters
        ----------
        X : array-like, shape = [n_ts, ]
            The training input timeseries. Each timeseries must be an array,
            but the lengths can be variable

        Returns
        -------
        """
        assert self.is_fitted, "Fit the gendis model first calling fit()"
        assert len(shapelets) > 0, "len(shapelets) must be > 0"

        index = None
        if hasattr(X, "index"):
            index = X.index

        device = "cuda" if torch.cuda.is_available() else "cpu"
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        logging.info(f"using torch:{device}")

        D, L = calculate_shapelet_dist_matrix(
            X_tensor, shapelets, cache=None, device=device
        )

        subgroup = self.subgroup_search.transform(D, thresholds)

        cols = [f"D_{i}" for i in range(D.shape[1])]
        if return_positions:
            data = np.hstack((D, L))
            cols += [f"L_{i}" for i in range(L.shape[1])]

        data = np.hstack((data, subgroup.reshape(-1, 1)))
        cols.append("in_subgroup")

        return pd.DataFrame(data=data, columns=cols, index=index), subgroup

    def save(self, path):
        """Write away all hyper-parameters and discovered shapelets to disk"""
        pickle.dump(self, open(path, "wb+"))

    @staticmethod
    def load(path):
        """Instantiate a saved GeneticExtractor"""
        return pickle.load(open(path, "rb"))
