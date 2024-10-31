from datetime import datetime
import logging
import multiprocessing
import numpy as np
import os
from os.path import basename, join, splitext
import pandas as pd
import random
from sklearn.model_selection import ShuffleSplit
import time


from gendis.operators import (
    add_shapelet,
    mask_shapelet,
    replace_shapelet,
    slide_shapelet,
    smooth_shapelet,
)
from gendis.operators import crossover_AND
from gendis.processing import preprocess_input

from gendis.genetic import GeneticExtractor
from gendis.SubgroupSearch import SubgroupSearch
from gendis.TopKSubgroups import TopKSubgroups
from gendis.visualization import (
    plot_best_matching_shaps,
    plot_coverage_heatmap,
    plot_error_distributions,
    plot_shaps,
)

from util import parse_args, save_json, setup_logging


def main():
    RANDOM_SEED_VAL = 1337
    random.seed(RANDOM_SEED_VAL)

    # Setup
    args = parse_args()
    input_file_path = args.input_file_path
    input_filename = splitext(basename(input_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = join(os.getcwd(), "results", f"{input_filename}_{timestamp}")
    setup_logging(results_folder, timestamp)

    logging.info("Script started")
    logging.info(f"Received input file path: {input_file_path}")

    # Reading, splitting data
    df = pd.read_csv(input_file_path)

    logging.info("Error column info:")
    logging.info(df.error.describe().to_dict())

    logging.info("Labels info:")
    logging.info(df.label.value_counts().to_dict())

    img_path = join(results_folder, "error_dist.png")
    plot_error_distributions(df, img_path)

    try:
        X = df.drop(columns=["pattern_x0", "pattern_x1", "pattern_y", "error", "label"])
    except:
        X = df.drop(columns=["error", "label"]).values
    y = df["error"]

    # sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    # sss.get_n_splits(X, y)
    # train_index, test_index = next(sss.split(X, y))
    # X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    # X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    X_train, y_train = X, y

    CACHE_SIZE = 8192

    # Subgroup Search parameters
    K = 10
    COVERAGE_ALPHA = 0.5
    SUBGROUP_SIZE_BETA = 0.5
    THRESHOLD_MAX_IT = 300
    subgroup_args = {
        "K": K,
        "coverage_alpha": COVERAGE_ALPHA,
        "subgroup_size_beta": SUBGROUP_SIZE_BETA,
        "cache_size": CACHE_SIZE,
        "random_seed": RANDOM_SEED_VAL,
        "max_it": THRESHOLD_MAX_IT,
    }

    logging.info("Parameters TopKSubgroups and SubgroupDistance")
    logging.info(subgroup_args)

    subgroup_search = SubgroupSearch(
        distance_function=SubgroupSearch.simple_mean,
        sg_size_beta=SUBGROUP_SIZE_BETA,
        max_it=THRESHOLD_MAX_IT,
        cache_size=CACHE_SIZE,
        random_seed=RANDOM_SEED_VAL,
    )

    top_k = TopKSubgroups(K, COVERAGE_ALPHA)

    mut_ops = [
        add_shapelet,
        mask_shapelet,
        replace_shapelet,
        # slide_shapelet,
        # smooth_shapelet,
    ]
    cx_ops = []

    funcs = {
        "top_k": top_k,
        "subgroup_search": subgroup_search,
        "mut_ops": mut_ops,
        "cx_ops": cx_ops,
    }

    gendis_args = {
        "population_size": 200,
        "iterations": 400,
        "mutation_prob": 0.5,
        "crossover_prob": 0,
        "max_shaps": 2,
        "wait": 80,
        "pop_restarts": 3,
        "min_len": 15,
        "max_len": 40,
        "n_jobs": 1,  # multiprocessing.cpu_count() - 3,
        "cache_size": CACHE_SIZE,
        "verbose": False,
        "random_seed": RANDOM_SEED_VAL,
    }
    logging.info("Parameters Gendis")
    logging.info(gendis_args)
    save_json(
        {**gendis_args, **subgroup_args},
        join(results_folder, "parameters.json"),
    )

    # Preprocess and model fit
    args = {**gendis_args, **funcs}
    X_input, y_input = preprocess_input(X_train, y_train)
    gendis = GeneticExtractor(**args)
    t0 = time.time()
    gendis.fit(X_input, y_input)
    t1 = time.time()

    # Log results
    logging.info(f"Finished training, it={gendis.it}, time={t1-t0}\n\n\n")
    logging.info("Best individual stats")
    logging.info(gendis.best["info"])
    save_json(gendis.best["info"], join(results_folder, "best_info.json"))

    logging.info("Top-K info")
    topk_info = []
    for ind in gendis.top_k.subgroups:
        data = ind.info
        data["coverage_weight"] = ind.coverage_weight
        data["subgroup_indexes"] = np.where(ind.subgroup)
        topk_info.append(data)
    logging.info(topk_info)
    save_json(topk_info, join(results_folder, "topk_info.json"))
    save_json(gendis.top_k.to_dict(), join(results_folder, "topk.json"))

    logging.info("Top-K coverage")
    logging.info(gendis.top_k.coverage)
    save_json(
        {"coverage": gendis.top_k.coverage},
        join(results_folder, "topk_coverage.json"),
    )
    img_path = join(results_folder, f"coverage_heatmap.png")
    plot_coverage_heatmap(gendis.top_k.coverage, img_path, cmap="YlGnBu")

    # Plot best matching shapelets
    for i, individual in enumerate(gendis.top_k.subgroups):
        img_path = join(results_folder, f"shapelets_matching_plots_top_{i}.png")
        distances, subgroup = gendis.transform(
            X_input, shapelets=individual, thresholds=individual.thresholds
        )
        plot_best_matching_shaps(
            X_train, distances, subgroup, individual, img_path=img_path
        )

    gendis.save(join(results_folder, "gendis.pickle"))


if __name__ == "__main__":
    main()
