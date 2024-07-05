import copy
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit
import sys
from tqdm import tqdm

from gendis.genetic import GeneticExtractor
from gendis.subgroup_distance import SubgroupDistance

from util import parse_args, setup_logging
from viz import plot_error_distributions

# Setup

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_folder = f"./{timestamp}"

args = parse_args()
setup_logging(results_folder)
input_file_path = args.input_file_path

logging.info('[INFO] Script started')
logging.debug(f'[INFO] Received input file path: {input_file_path}')

# Reading, splitting data

df = pd.read_csv(input_file)

logging.info('[INFO] Error column info:')
logging.info(df.error.describe().to_dict())

logging.info('[INFO] Labels info:')
logging.info(df.label.value_counts().to_dict())

plot_error_distributions(results_folder)

X = df.drop(columns=['error', 'label'])
y = df['error']

sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
sss.get_n_splits(X, y)
train_index, test_index = next(sss.split(X, y))

X_train, y_train = X.iloc[train_index], y.iloc[train_index]
X_test, y_test = X.iloc[test_index], y.iloc[test_index]

# Gendis initialization

COVERAGE_ALPHA = 0.5
SUBGROUP_SIZE_BETA = 0.2
DIST_THRESHOLD = 150

logging.info('[INFO] Parameters SubgroupDistance')
logging.info({
    'coverage_alpha': COVERAGE_ALPHA,
    'subgroup_size_beta': SUBGROUP_SIZE_BETA,
    'shapelet_distance_threshold': DIST_THRESHOLD
})

subgroup_dist_func = SubgroupDistance(
    distance_function=SubgroupDistance.simple_mean, 
    shapelet_dist_threshold=DIST_THRESHOLD,
    standardize=False
)

args = {
    "k": 5,
    "fitness": subgroup_dist_func,
    "coverage_alpha": COVERAGE_ALPHA,
    "mut_ops": mut_ops,
    "cx_ops": cx_ops,
    "population_size": 100, 
    "iterations": 100,  
    "mutation_prob": 0.3, 
    "crossover_prob": 0.3,
    "max_shaps": 3,
    "wait": 20, 
    "min_len": 20,
    "max_len": 60,
    "n_jobs": multiprocessing.cpu_count() - 3,
    "verbose": True,
    "dist_threshold": DIST_THRESHOLD,
} 

logging.info('[INFO] Parameters Gendis')
logging.info(args)

# Preprocess and model fit

X_input, y_input = GeneticExtractor.preprocess_input(X_train, y_train)
gendis.fit(X_input, y_input)

# Log results

logging.info('[INFO] Results')
logging.info('[INFO] Best individual stats')
logging.info(gendis.best["info"])

logging.info('[INFO] Top-K info')
logging.info([x.info for x in gendis.top_k])

logging.info('[INFO] Top-K coverage')
logging.info(gendis.top_k_coverage)

# Plot best matching shapelets

for i, ind in enumerate(gendis.top_k):
    print(f"Evaluating ind {i}")
    plot_best_matching_shaps(
        gendis, ind, X_input, y_input
        path=results_folder, plot_i=i
    )

gendis.save(f"{results_folder}/gendis.pickle")
