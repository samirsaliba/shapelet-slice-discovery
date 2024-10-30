import copy
from deap import base, creator, tools
from dtaidistance.preprocessing import differencing
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

from .individual import (
    Shapelet,
    ShapeletIndividual,
)
from .LRUCache import LRUCache
from .operators import (
    random_shapelet,
    kmeans,
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
    population_size : int
        The number of individuals in our population. Increasing this parameter
        increases both the runtime per generation, as the probability of
        finding a good solution.

    iterations : int
        The maximum number of generations the algorithm may run.

    wait : int
        If no improvement has been found for `wait` iterations, then stop

    add_noise_prob : float
        The chance that gaussian noise is added to a random shapelet from a
        random individual every generation

    add_shapelet_prob : float
        The chance that a shapelet is added to a random shapelet set every gen

    remove_shapelet_prob : float
        The chance that a shapelet is deleted to a random shap set every gen

    crossover_prob : float
        The chance that of crossing over two shapelet sets every generation

    normed : boolean
        Whether we first have to normalize before calculating distances

    n_jobs : int
        The number of threads to use

    verbose : boolean
        Whether to print some statistics in every generation

    Attributes
    ----------
    shapelets : array-like
        The fittest shapelet set after evolution
    label_mapping: dict
        A dictionary that maps the labels to the range [0, ..., C-1]

    Example
    -------
    An example showing genetic shapelet extraction on a simple dataset:

    >>> from tslearn.generators import random_walk_blobs
    >>> from genetic import GeneticExtractor
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>> X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, noise_level=0.1)
    >>> X = np.reshape(X, (X.shape[0], X.shape[1]))
    >>> extractor = GeneticExtractor(iterations=5, population_size=10)
    >>> distances = extractor.fit_transform(X, y)
    >>> lr = LogisticRegression()
    >>> _ = lr.fit(distances, y)
    >>> lr.score(distances, y)
    1.0
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
        normed=False,
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
        self.normed = normed
        self.plot = False

        # Attributes
        self.is_fitted = False
        self.label_mapping = {}
        self.should_reset_pop = False
        self.pop_restart_counter = 0
        self.apply_differencing = True

    @staticmethod
    def preprocess_input(X, y):
        _X = copy.deepcopy(X)
        if isinstance(_X, pd.DataFrame):
            _X = _X.values
        _X = np.apply_along_axis(lambda s: differencing(s, smooth=None), 1, _X)

        y = copy.deepcopy(y)
        if isinstance(y, pd.Series):
            y = y.values

        return _X, y

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
        init_op = np.random.choice(self.init_ops)
        return init_op(
            X=self.X,
            n_shapelets=n_shapelets,
            max_len=self.max_len,
            min_len=self.min_len,
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
        if np.random.random() < self.mutation_prob:
            clone = copy.deepcopy(ind)
            mut_op = np.random.choice(self.mut_ops)

            if (
                mut_op.__name__ == "add_shapelet" and len(clone) >= self.max_shaps
            ) or clone.subgroup_size == 1:
                mut_op = replace_shapelet

            return mut_op(self.X, clone, self.min_len, self.max_len)
        return ind

    def _cross_individuals(self, ind1, ind2, toolbox):
        """Cross two individuals"""
        if np.random.random() < self.crossover_prob:
            ind1_clone = toolbox.clone(ind1)
            ind2_clone = toolbox.clone(ind2)

            cx_op = np.random.choice(self.deap_cx_ops)
            child1, _ = cx_op(ind1_clone, ind2_clone)
            child1.uuid_history.append(ind1_clone.uuid)
            child1.register_op("cx-child")
            ind1_clone.register_op("cx-parent")

            return child1, ind1_clone

        return ind1, ind2

    @staticmethod
    def rebuild_diffed(series):
        return np.insert(np.cumsum(series), 0, 0)

    def _check_early_stopping(self):
        if self.it - self.top_k.last_update > self.wait:
            if self.pop_restart_counter < self.pop_restarts:
                # Will trigger population reset
                self.top_k.last_update = self.it
                self.should_reset_pop = True
            else:
                return True

        return False

    def assert_healthy_individual(self, ind, msg):
        assert isinstance(
            ind, ShapeletIndividual
        ), f"Expected a ShapeletIndividual instance [{msg}]."
        assert hasattr(ind, "to_dict"), f"ShapeletIndividual does not have to_dict."

        for shap in ind:
            try:
                assert isinstance(
                    shap, Shapelet
                ), f"Expected a Shapelet instance [{msg}]."
                assert hasattr(shap, "id"), f"Shapelet does not have an 'id' attribute."
            except Exception as e:
                print(ind)
                raise (e)

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
        torch.cuda.empty_cache()
        self.X = X
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"using torch:{self.device}")
        self.X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = y

        self._min_length_series = min([len(x) for x in self.X])

        if self._min_length_series <= 4:
            raise Exception("Time series should be of at least length 4!")

        if self.max_len is None:
            if len(self.X[0]) > 20:
                self.max_len = len(self.X[0]) // 2
            else:
                self.max_len = len(self.X[0])

        if self.max_shaps is None:
            self.max_shaps = int(np.sqrt(self._min_length_series)) + 1

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
        self.invalid_counter = 0
        while self.it <= self.iterations:
            logging.info(
                f"""it:{self.it}, \
                it_time: {(time.time() - it_start):.2f}"""
            )
            it_start = time.time()
            self.invalid_counter = 0

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
                logging.info(f"Best ind: {best}")
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
        if self.apply_differencing:
            self.best["shaps_undiffed"] = [
                self.rebuild_diffed(x) for x in self.best["shapelets"]
            ]

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

    def fit_transform(self, X, y):
        """Combine both the fit and transform method in one.

        Parameters
        ----------
        X : array-like, shape = [n_ts, ]
            The training input timeseries. Each timeseries must be an array,
            but the lengths can be variable

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        D : array-like, shape = [n_ts, n_shaps]
            The matrix with distances
        """
        # First call fit, then transform
        self.fit(X, y)
        return self.transform(X)

    def save(self, path):
        """Write away all hyper-parameters and discovered shapelets to disk"""
        pickle.dump(self, open(path, "wb+"))

    def get_subgroups(self, X_diffed, y, shapelets=None):
        """
        Get the subgroups based on the provided shapelets (if not provided, the best found by gendis).

        Parameters:
        - X (array-like): Input time series data.
        - y (array-like): Target labels for the time series data.
        - shapelets (array-like, optional): Shapelets used for transformation. If not provided,
        the function assumes that shapelets have already been calculated.

        Returns:
        - sg_indexes (array): Indexes of instances belonging to subgroups.
        - not_sg_indexes (array): Indexes of instances not belonging to subgroups.
        """
        assert self.is_fitted, "Fit the gendis model first calling fit()"

        shapelets = self.best["shapelets"]

        D, _ = calculate_shapelet_dist_matrix(
            X_diffed,
            shapelets,
            dist_function=self.dist_function,
            return_positions=True,
            cache=None,
        )

        subgroup = self.subgroup_search.filter_subgroup_shapelets(D, y)
        [sg_indexes] = np.where(subgroup)
        [not_sg_indexes] = np.where(~subgroup)

        return sg_indexes, not_sg_indexes

    @staticmethod
    def load(path):
        """Instantiate a saved GeneticExtractor"""
        return pickle.load(open(path, "rb"))
