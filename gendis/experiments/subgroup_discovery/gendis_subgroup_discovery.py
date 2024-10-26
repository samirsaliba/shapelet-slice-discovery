from datetime import datetime
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import time

from gendis.operators import add_shapelet
from gendis.operators import crossover_AND

from gendis.genetic import GeneticExtractor
from gendis.SubgroupSearch import SubgroupSearch
from gendis.TopKSubgroups import TopKSubgroups

from util import parse_args, save_json, setup_logging
from viz import plot_best_matching_shaps, plot_error_distributions, plot_shaps

# Setup
args = parse_args()
input_file_path = args.input_file_path
input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_folder = f"./results/{input_filename}_{timestamp}"
setup_logging(results_folder, timestamp)


logging.info("[INFO] Script started")
logging.info(f"[INFO] Received input file path: {input_file_path}")

# Reading, splitting data
df = pd.read_csv(input_file_path)

logging.info("[INFO] Error column info:")
logging.info(df.error.describe().to_dict())

logging.info("[INFO] Labels info:")
logging.info(df.label.value_counts().to_dict())

plot_error_distributions(df, results_folder)

try:
    X = df.drop(columns=["pattern_x0", "pattern_x1", "pattern_y", "error", "label"])
except:
    X = df.drop(columns=["error", "label"])
y = df["error"]

# sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
# sss.get_n_splits(X, y)
# train_index, test_index = next(sss.split(X, y))
# X_train, y_train = X.iloc[train_index], y.iloc[train_index]
# X_test, y_test = X.iloc[test_index], y.iloc[test_index]

X_train, y_train = X, y

# Subgroup Search parameters
K = 10
COVERAGE_ALPHA = 0.8
SUBGROUP_SIZE_BETA = 0.5
subgroup_args = {
    "K": K,
    "coverage_alpha": COVERAGE_ALPHA,
    "subgroup_size_beta": SUBGROUP_SIZE_BETA,
}

logging.info("[INFO] Parameters TopKSubgroups and SubgroupDistance")
logging.info(subgroup_args)

subgroup_search = SubgroupSearch(
    distance_function=SubgroupSearch.simple_mean,
    sg_size_beta=SUBGROUP_SIZE_BETA,
    standardize=False,
    max_it=300,
)

top_k = TopKSubgroups(K, COVERAGE_ALPHA)

mut_ops = [add_shapelet]
cx_ops = [crossover_AND]

funcs = {
    "top_k": top_k,
    "subgroup_search": subgroup_search,
    "mut_ops": mut_ops,
    "cx_ops": cx_ops,
}

gendis_args = {
    "population_size": 300,
    "iterations": 200,
    "mutation_prob": 0.3,
    "crossover_prob": 0.2,
    "max_shaps": 2,
    "wait": 50,
    "pop_restarts": 2,
    "min_len": 15,
    "max_len": 40,
    "n_jobs": multiprocessing.cpu_count() - 3,
    "verbose": False,
}
logging.info("[INFO] Parameters Gendis")
logging.info(gendis_args)
save_json({**gendis_args, **subgroup_args}, f"{results_folder}/parameters.json")

# Preprocess and model fit
args = {**gendis_args, **funcs}
X_input, y_input = GeneticExtractor.preprocess_input(X_train, y_train)
gendis = GeneticExtractor(**args)
t0 = time.time()
gendis.fit(X_input, y_input)
t1 = time.time()

# Log results
logging.info(f"[INFO] Finished training, it={gendis.it}, time={t1-t0}")
logging.info("[INFO] Best individual stats")
logging.info(gendis.best["info"])
save_json(gendis.best["info"], f"{results_folder}/best_info.json")

logging.info("[INFO] Top-K info")
topk_info = []
for ind in gendis.top_k.subgroups:
    data = ind.info
    data["coverage_weight"] = ind.coverage_weight
    data["subgroup_indexes"] = np.where(ind.subgroup)
    topk_info.append(data)
logging.info(topk_info)
save_json(topk_info, f"{results_folder}/topk_info.json")
save_json(gendis.top_k.to_dict(), f"{results_folder}/topk.json")

logging.info("[INFO] Top-K coverage")
logging.info(gendis.top_k.coverage)
save_json({"coverage": gendis.top_k.coverage}, f"{results_folder}/topk_coverage.json")

# Plot best matching shapelets
for i, ind in enumerate(gendis.top_k.subgroups):
    # plot_shaps(ind, path=results_folder, ind_label=f"{i}_{ind.uuid}")
    plot_best_matching_shaps(
        gendis, ind, X_input, y_input, path=results_folder, plot_i=i
    )

gendis.save(f"{results_folder}/gendis.pickle")
