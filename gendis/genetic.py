# Standard lib
from collections import OrderedDict
import copy
import array
import time

# "Standard" data science libs
import numpy as np
from math import ceil, floor, isinf
import matplotlib.pyplot as plt
import pandas as pd

# Serialization
import pickle

# Evolutionary algorithms framework
from deap import base, creator, tools

# Parallelization
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing

# ML
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


# Pairwise distances
try:
    from pairwise_dist import _pdist
except:
    from gendis.pairwise_dist import _pdist

try:
    from shapelets_distances import calculate_shapelet_dist_matrix, dtw
except:
    from gendis.shapelets_distances import calculate_shapelet_dist_matrix, dtw


# Custom genetic operators
try:
    from initialization import random_shapelet, kmeans
    from crossover import crossover_AND, crossover_uniform
    from mutation import (
        add_shapelet, remove_shapelet, replace_shapelet, smooth_shapelet
    )

except:
    from gendis.initialization import random_shapelet, kmeans
    from gendis.crossover import crossover_AND, crossover_uniform
    from gendis.mutation import (
        add_shapelet, remove_shapelet, replace_shapelet, smooth_shapelet
    )

from dtaidistance.preprocessing import differencing

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value


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

    plot : object
        Whether to plot the individuals every generation (if the population 
        size is <= 20), or to plot the fittest individual

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
    dist_function_options = {
        'original': {
            'function': _pdist,
            'returns': False
        },
        'pairwise_euclidean': {
            'function': _pdist,
            'returns': False
        },
        'dtw': {
            'function': dtw,
            'returns': True
        }
    }
    


    def __init__(self, 
            dist_function,
            apply_differencing,
            population_size=50, 
            iterations=25, 
            verbose=False, 
            normed=False, 
            mutation_prob=0.1, 
            crossover_prob=0.4,
            coverage_alpha=0.5,
            wait=10, 
            plot=None, 
            max_shaps=None, 
            n_jobs=1, 
            max_len=None,
            min_len=0, 
            fitness=None,
            init_ops=[random_shapelet],
            cx_ops=[crossover_AND, crossover_uniform], 
            mut_ops=[add_shapelet, remove_shapelet, replace_shapelet, smooth_shapelet],
            dist_threshold=1.0,
        ):
        self._set_distance_function(dist_function)
        self._set_fitness_function(fitness)
        self.apply_differencing = apply_differencing
        # Hyper-parameters
        self.population_size = population_size
        self.iterations = iterations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.coverage_alpha = coverage_alpha
        self.plot = plot
        self.wait = wait
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.normed = normed
        self.min_len = min_len
        self.max_len = max_len
        self.max_shaps = max_shaps
        self.init_ops = init_ops
        self.cx_ops = cx_ops
        self.mut_ops = mut_ops
        self.is_fitted = False
        self.dist_threshold = dist_threshold

        # Attributes
        self.label_mapping = {}
        self.shapelets = []

    def _set_fitness_function(self, fitness):
        assert fitness is not None, "Please include a fitness function via fitness parameter.\
            See fitness.logloss_fitness for classification or \
            subgroup_distance.SubgroupDistance for subgroup search"
        assert callable(fitness)
        self.fitness = fitness

    def _set_distance_function(self, dist_function):
        dist_function = dist_function.lower()
        assert_error_msg = f"Distance function not recognized. \
            Options are {', '.join(self.dist_function_options)}"
        assert dist_function in self.dist_function_options.keys(), assert_error_msg
        self.dist_function = self.dist_function_options[dist_function]['function']
        self.dist_func_returns = self.dist_function_options[dist_function]['returns']
   
    def _convert_X(self, X):
        _X = copy.deepcopy(X)
        if isinstance(_X, list):
            for i in range(len(_X)):
                _X[i] = np.array(_X[i])
            _X = np.array(_X)

        if isinstance(_X, pd.DataFrame):
            _X = _X.values

        if _X.dtype != object:
            return _X.view(np.float64)
        else:
            return _X

    def _convert_y(self, y, convert_categorical=False):
        y = copy.deepcopy(y)

        if isinstance(y, pd.Series):
            y = y.values

        if convert_categorical:
            # Map labels to [0, ..., C-1]
            for j, c in enumerate(np.unique(y)):
                self.label_mapping[c] = j

            # Use pandas map function and convert to numpy
            y = np.reshape(pd.Series(y).map(self.label_mapping).values, (-1, 1))

        return y

    def _preprocess_series(self, X):
        if self.apply_differencing:
            return np.apply_along_axis(lambda s: differencing(s, smooth=None), 1, X)
        else:
            return X

    def _print_statistics(self, it, stats, start):
        if it == 1:
            # Print the header of the statistics
            print('it\t\tavg\t\tstd\t\tmax\t\ttime')
            #print('it\t\tavg\t\tmax\t\ttime')

        print('{}\t\t{}\t\t{}\t\t{}\t{}'.format(
        # print('{}\t\t{}\t\t{}\t{}'.format(
            it,
            np.around(stats['avg'], 4),
            np.around(stats['std'], 3),
            np.around(stats['max'], 6),
            np.around(time.time() - start, 4),
        ))

    def _plot_best(self, offspring, height):
        if self.population_size <= 20:
            if self.plot == 'notebook':
                f, ax = plt.subplots(4, height, sharex=True)
            for ix, ind in enumerate(offspring):
                ax[ix//height][ix % height].clear()
                for s in ind:
                    ax[ix//height][ix % height].plot(range(len(s)), s)
            plt.pause(0.001)
            if self.plot == 'notebook':
                plt.show()

        else:
            plt.clf()
            for shap in self.best['shapelets']:
                plt.plot(range(len(shap)), shap)
            plt.pause(0.001)

    def _update_best_individual(self, it, new_ind):
        """Update the best individual if we found a better one"""
        ind_score = self._eval_individual(new_ind, return_info=True)

        # Overwrite self.shapelets everytime so we can
        # pre-emptively stop the genetic algorithm
        best_shapelets = []
        for shap in new_ind:
            best_shapelets.append(shap.flatten())

        self.best = {
            'it': it,
            'score': ind_score['value'][0],
            'info': ind_score['info'],
            'shapelets': best_shapelets
        }
 

    def _eval_individual(self, shaps, return_info=False):
            """Evaluate the fitness of an individual"""
            D, _ = calculate_shapelet_dist_matrix(
                self.X, shaps, 
                dist_function=self.dist_function, 
                return_positions=False,
                dist_func_returns=self.dist_func_returns, 
                cache=self.cache, 
                verbose=self.verbose
                )
            fit = self.fitness(D=D, y=self.y, shaps=shaps)

            if return_info: return fit
            return fit["value"]

    def _safe_std(*args, **kwargs):
        try:
            return np.std(*args, **kwargs)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def rebuild_diffed(series):
        return np.insert(series.cumsum(), 0, 0)

    def _early_stopping_check(self, it):
        return (
            not isinf(self.best["score"]) # If score is inf don't stop
            and it - self.best["it"] > self.wait
        )

    def _update_coverage(self, ind, coverage):
        D, _ = calculate_shapelet_dist_matrix(
            self.X, ind["shapelets"], 
            dist_function=self.dist_function, 
            return_positions=False,
            dist_func_returns=self.dist_func_returns, 
            cache=self.cache, 
            verbose=self.verbose
        )
        subgroup = self.fitness.filter_subgroup_shapelets(D, self.y)

        # Since we have just created a new subgroup,
        # we add +1 to every subgroup member instance counts
        coverage[subgroup] += 1
        # Raise alpha to the 'counts' for each instance
        # That's how much each instance will contribute to a next iteration
        base = [cov_alpha]
        return np.power(base, coverage), coverage

    def _coverage_factor(self, cov_weights, subgroup):
        """Multiplicative weighted covering score"""
        in_sg_weights = cov_weights[subgroup].sum()
        sg_weights_total = subgroup.sum() * self.coverage_alpha
        return in_sg_weights / sg_weights_total

    def _update_kbest(pop):
        pop = list(map(toolbox.clone, pop))
        coverage = np.ones(len(X_train))
        cov_weights = np.power([self.coverage_alpha], coverage)

        best = max(pop, key=lambda ind: ind["fitness"])
        top_k = [best]

        for i in range(10):
            # Update coverage and coverage_weights
            cov_weights, coverage = self._update_coverage(best, coverage)
            
            # Now update fitnesses again
            # For each individual, update fitness based on cov_weights
            for ind in pop:
                ind["fitness"] *= _coverage_factor(cov_weights, ind["subgroup"])

            # Get best individual
            best = max(pop, key=lambda ind: ind["fitness"])
            top_k.append(best)

    def fit(self, X, y, convert_categorical_labels=False):
        """Extract shapelets from the provided timeseries and labels.

        Parameters
        ----------
        X : array-like, shape = [n_ts, ]
            The training input timeseries. Each timeseries must be an array,
            but the lengths can be variable

        y : array-like, shape = [n_samples]
            The target values.
        """
        self.X = self._preprocess_series(self._convert_X(X))
        self.y = self._convert_y(y, convert_categorical_labels)    
        self._min_length_series = min([len(x) for x in self.X])

        if self._min_length_series <= 4:
            raise Exception('Time series should be of at least length 4!')

        if self.max_len is None:
            if len(self.X[0]) > 20:
                self.max_len = len(self.X[0]) // 2
            else:
                self.max_len = len(self.X[0])

        if self.max_shaps is None:
            self.max_shaps = int(np.sqrt(self._min_length_series)) + 1

        # Sci-kit learn check for label vector.
        # check_array(y)
        self.cache = LRUCache(2048)

        # We will try to maximize the negative logloss of LR in CV.
        # In the case of ties, we pick the one with least number of shapelets
        # 0.0 refers to subgroup mean, doesn't change calculations but we'll use for logging
        # weights = (1.0) #(1.0, -1.0, 0.0)
        creator.create("FitnessMax", base.Fitness, weights=[1.0])

        # Individual are lists (of shapelets (list))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Keep a history of the evolution
        self.history = []

        # Register all operations in the toolbox
        toolbox = base.Toolbox()

        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        if self.n_jobs > 1:
            pool = Pool(self.n_jobs)
            toolbox.register("map", pool.map)
        else:
            toolbox.register("map", map)

        # Register all our operations to the DEAP toolbox
        # toolbox.register("merge", merge_crossover)
        deap_cx_ops = []
        for i, cx_op in enumerate(self.cx_ops):
            toolbox.register(f"cx{i}", cx_op)
            deap_cx_ops.append(getattr(toolbox, (f"cx{i}")))
        deap_mut_ops = []
        for i, mut_op in enumerate(self.mut_ops):
            toolbox.register(f"mutate{i}", mut_op)
            deap_mut_ops.append(getattr(toolbox, (f"mutate{i}")))

        def create_individual(n_shapelets=None):
            """Generate a random shapelet set"""
            if n_shapelets is None:
                n_shapelets = np.random.randint(2, self.max_shaps)

            init_op = np.random.choice(self.init_ops)
            return init_op(
                X=self.X, 
                n_shapelets=n_shapelets, 
                min_len_series=self._min_length_series, 
                max_len=self.max_len, 
                min_len=self.min_len
            )

        toolbox.register("create", create_individual)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.create)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._eval_individual)
        # Small tournaments to ensure diversity
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Set up the statistics. We will measure the mean, std dev and max
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])

        stats.register("avg", lambda arr: np.ma.masked_invalid(arr).mean())
        stats.register("std", lambda arr: np.ma.masked_invalid(arr).std())
        stats.register("max", lambda arr: np.ma.masked_invalid(arr).max())
        stats.register("min", lambda arr: np.ma.masked_invalid(arr).min())
        # stats.register("q25", lambda x: np.quantile(x, 0.25))
        # stats.register("q75", lambda x: np.quantile(x, 0.75))

        # Initialize the population and calculate their initial fitness values
        pop = toolbox.population(n=self.population_size)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Keep track of the best iteration, in order to do stop after `wait`
        # generations without improvement
        it = 1
        self.best = {
            'it': it,
            'score': float('-inf'),
            'info': None,
            'shapelets': []
        }

        # Set up a matplotlib figure and set the axes
        height = int(np.ceil(self.population_size/4))
        if self.plot is not None and self.plot != 'notebook':
            if self.population_size <= 20:
                f, ax = plt.subplots(4, height, sharex=True)
            else:
                plt.figure(figsize=(15, 5))
                plt.xlim([0, len(self.X[0])])

        # The genetic algorithm starts here
        while it <= self.iterations:

            # Early stopping
            if self._early_stopping_check(it):
                break

            gen_start = time.time()

            # Clone the population into offspring
            offspring = list(map(toolbox.clone, pop))

            # Plot the fittest individual of our population
            if self.plot is not None:
                self._plot_best(offspring, height)

            # Iterate over all individuals and apply CX with certain prob
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                for cx_op in deap_cx_ops:
                    if np.random.random() < self.crossover_prob:
                        cx_op(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

            # Apply mutation to each individual
            for idx, indiv in enumerate(offspring):
                for mut_op in deap_mut_ops:
                    if np.random.random() < self.mutation_prob:
                        mut_op(indiv, toolbox)
                        del indiv.fitness.values

            # Update the fitness values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population and update hall of fame, statistics & history
            new_pop = toolbox.select(offspring, self.population_size - 1)
            fittest_ind = tools.selBest(pop + offspring, 1)
            pop[:] = new_pop + fittest_ind
            it_stats = stats.compile(pop)
            self.history.append([it, it_stats])

            # Update bag of best individuals
            self._update_kbest(pop)

            # Print our statistics
            if self.verbose:
                self._print_statistics(it=it, stats=it_stats, start=gen_start)

            # Have we found a new best score?
            if it_stats['max'] > self.best['score']:
                best_ind = tools.selBest(pop + offspring, 1)[0]
                self._update_best_individual(
                    it=it,
                    new_ind=best_ind,
                )
            it += 1

        self.pop = pop
        self.top_10_best = []
        for ind in tools.selBest(pop, 10):
            ind_score = self._eval_individual(ind, return_info=True)            
            ind_info = {
                'score': ind_score['value'][0],
                'info': ind_score['info'],
                'shapelets': ind
            }
            ind_info = self.top_10_best.append(ind_info)

        self.best["shaps_undiffed"] = [self.rebuild_diffed(x) for x in self.best["shapelets"]]
        self.is_fitted = True
        del self.X, self.y


    def update_coverage_weights(self, X, y, cov_alpha, coverage):
        assert self.is_fitted, "Fit the gendis model first calling fit()"
        coverage = copy.deepcopy(coverage)

        X = self._preprocess_series(self._convert_X(X))

        D, _ = calculate_shapelet_dist_matrix(
            X, self.best["shapelets"], 
            dist_function=self.dist_function, 
            return_positions=False,
            dist_func_returns=self.dist_func_returns, 
            cache=self.cache, 
            verbose=self.verbose
        )
        subgroup = self.fitness.filter_subgroup_shapelets(D, y)

        # Since we have just created a new subgroup,
        # we add +1 to every subgroup member instance counts
        coverage[subgroup] += 1
        # Raise alpha to the 'counts' for each instance
        # That's how much each instance will contribute to a next iteration
        base = [cov_alpha]
        return np.power(base, coverage), coverage
        
        
    def transform(self, X, shapelets=None, return_positions=False, standardize=False):
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

        if shapelets is None:
            shapelets = self.best['shapelets']

        index = None
        if hasattr(X, 'index'):
            index = X.index

        X = self._preprocess_series(self._convert_X(X))
        check_is_fitted(self, ['is_fitted'])

        D, L = calculate_shapelet_dist_matrix(
            X, shapelets, 
            dist_function=self.dist_function, 
            dist_func_returns=self.dist_func_returns, 
            return_positions=return_positions,
            cache=None, 
            verbose=False
        )

        if standardize:
            scaler = StandardScaler()
            return np.absolute(scaler.fit_transform(D))   

        cols = [f'D_{i}' for i in range(D.shape[1])]
        if return_positions:
            data = np.hstack((D, L))
            cols += [f'L_{i}' for i in range(L.shape[1])]
        else:
            data, cols = D, None
            
        return pd.DataFrame(data=data, columns=cols, index=index), pd.DataFrame(data=X, index=index)

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
        D = self.transform(X)
        return D

    def save(self, path):
        """Write away all hyper-parameters and discovered shapelets to disk"""
        pickle.dump(self, open(path, 'wb+'))

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
        check_is_fitted(self, ['is_fitted'])
        shapelets = self.best["shapelets"]
        
        D, _ = calculate_shapelet_dist_matrix(
            X_diffed, shapelets, 
            dist_function=self.dist_function, 
            dist_func_returns=self.dist_func_returns, 
            return_positions=True,
            cache=None, 
            verbose=False
        )

        subgroup = self.fitness.filter_subgroup_shapelets(D, y)
        [sg_indexes] = np.where(subgroup)
        [not_sg_indexes] = np.where(~subgroup)

        return sg_indexes, not_sg_indexes

    @staticmethod
    def load(path):
        """Instantiate a saved GeneticExtractor"""
        return pickle.load(open(path, 'rb'))

    def plot_series_and_shapelets(
        self,
        X,
        y,
        shapelets,
        indexes_to_plot,
        row_n = 5,
        col_m = 2,
        adjust_w = 1,
        adjust_h = 0.5,
        series_offset = 0,
    ):
        default_w, default_h = (4.8, 6.4)
        figsize = (col_m*default_w*adjust_w, row_n*default_h*adjust_h)

        fig, axs = plt.subplots(row_n, col_m, figsize=figsize)
        fig.tight_layout(pad=3.0)

        D, L = self.transform(X=X, shapelets=shapelets)

        for i, series_idx in enumerate(indexes_to_plot[series_offset:row_n*col_m]):
            row, col = i//col_m, i%col_m
            ax = axs[row][col]

            series = X.iloc[series_idx].values
            model_error = y.iloc[series_idx]
            ax.plot(series, alpha=0.3)
            ax.title.set_text(f'Series index {series_idx}, model error of {model_error:.2f}')
            
            for shap_idx, shap in enumerate(shapelets): 
                dist = D[series_idx][shap_idx]
                loc = L[series_idx][shap_idx]

                k = loc * float(len(series) - len(shap)) 
                start = floor(k)
                end = ceil(start + len(shap))
                shap_idx = list(range(start, end))

                # Dotted line if dist is above threshold
                fmt = '--' if dist > self.dist_threshold else '-'
                ax.plot(shap_idx, shap, fmt)


        i+=1
        # Remove unused axes
        while i < (row_n*col_m):
            row, col = i//col_m, i%col_m
            axs[row][col].remove()
            i+=1
        
        return plt
