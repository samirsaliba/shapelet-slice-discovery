import argparse
from datetime import datetime
import logging
import numpy as np
import os
from os.path import basename, join, splitext
import pandas as pd
import random
import time

from gendis.processing import preprocess_input

from gendis.crossover import point_crossover
from gendis.mutation import (
    add_shapelet,
    mask_shapelet,
    replace_shapelet,
    slide_shapelet,
    smooth_shapelet,
)

from gendis.genetic import GeneticExtractor
from gendis.SubgroupSearch import SubgroupSearch
from gendis.TopKSubgroups import TopKSubgroups

from gendis.evaluation import class_predominance, evaluate_subgroup, recall
from gendis.visualization import (
    plot_target_histogram,
    plot_target_histogram_per_subgroup,
)

from util import save_json, setup_logging

N_TESTS = 30
PARAM_TEST = "K"
PARAM_TYPE = int


def main():
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path", type=str, help="The path to the input file")
    parser.add_argument(f"--{PARAM_TEST}", type=PARAM_TYPE, required=True)
    args = parser.parse_args()

    input_file_path = args.input_file_path
    PARAM_VALUE_RUN = getattr(args, PARAM_TEST)

    input_filename = splitext(basename(input_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = join(
        os.getcwd(), "results", f"MULT_{PARAM_TEST}_{PARAM_VALUE_RUN}"
    )
    setup_logging(results_folder, timestamp)

    logging.info("Script started")
    logging.info(f"Received input file path: {input_file_path}")

    df = pd.read_csv(input_file_path)

    logging.info("Error column info:")
    logging.info(df.error.describe().to_dict())

    logging.info("Labels info:")
    logging.info(df.label.value_counts().to_dict())

    img_path = join(results_folder, "error_dist.png")
    bins = np.arange(0, 1, 0.025)
    plot_target_histogram(
        df, label_col="label", target_col="error", img_path=img_path, bins=bins
    )

    labels = df["label"]
    try:
        X = df.drop(
            columns=["pattern_x0", "pattern_x1", "pattern_y", "error", "label"]
        ).values
    except:
        X = df.drop(columns=["error", "label"]).values
    y = df["error"]

    random_seed_values = range(N_TESTS)

    X_train, y_train = X, y

    CACHE_SIZE = 8192

    # Subgroup Search parameters
    K = 10
    COVERAGE_ALPHA = 0.5
    SUBGROUP_SIZE_BETA = 0.5
    THRESHOLD_MAX_IT = 400
    THRESHOLD_KAPPA = 0.9
    SEARCH_MODE = "percentile"
    subgroup_args = {
        "K": K,
        "coverage_alpha": COVERAGE_ALPHA,
        "sg_size_beta": SUBGROUP_SIZE_BETA,
        "cache_size": CACHE_SIZE,
        "random_seed_values": list(random_seed_values),
        "max_it": THRESHOLD_MAX_IT,
        "threshold_kappa": THRESHOLD_KAPPA,
        "threshold_search_mode": SEARCH_MODE,
    }

    if PARAM_TEST in subgroup_args:
        subgroup_args[PARAM_TEST] = PARAM_VALUE_RUN

    logging.info("Parameters TopKSubgroups and SubgroupDistance")
    logging.info(subgroup_args)

    mut_ops = [
        add_shapelet,
        mask_shapelet,
        replace_shapelet,
        slide_shapelet,
        # smooth_shapelet,
    ]
    cx_ops = [point_crossover]

    gendis_args = {
        "population_size": 200,
        "iterations": 100,
        "mutation_prob": 0.5,
        "crossover_prob": 0,
        "max_shaps": 2,
        "wait": 50,
        "pop_restarts": 3,
        "min_len": 15,
        "max_len": 40,
        "n_jobs": 1,  # multiprocessing.cpu_count() - 3,
        "cache_size": CACHE_SIZE,
        "verbose": False,
    }
    if PARAM_TEST in gendis_args:
        gendis_args[PARAM_TEST] = PARAM_VALUE_RUN

    logging.info("Parameters Gendis")
    logging.info(gendis_args)

    log_funcs = {
        "mut_ops": [x.__name__ for x in mut_ops],
        "cx_ops": [x.__name__ for x in cx_ops],
    }
    save_json(
        {**gendis_args, **subgroup_args, **log_funcs},
        join(results_folder, "run_info.json"),
    )

    pop_history = []
    topk_history = []
    classes = []
    metrics = []
    coverage_counts = []

    for run_num in random_seed_values:
        logging.info(f"Starting run={run_num}")

        RANDOM_SEED_VAL = run_num
        random.seed(RANDOM_SEED_VAL)

        top_k = TopKSubgroups(
            subgroup_args["K"],
            coverage_alpha=subgroup_args["coverage_alpha"],
            run_id=run_num,
        )

        subgroup_search = SubgroupSearch(
            distance_function=SubgroupSearch.simple_mean,
            threshold_search_mode=subgroup_args["threshold_search_mode"],
            threshold_kappa=subgroup_args["threshold_kappa"],
            sg_size_beta=subgroup_args["sg_size_beta"],
            max_it=subgroup_args["max_it"],
            cache_size=subgroup_args["cache_size"],
            random_seed=run_num,
        )

        # Preprocess and model fit
        args = {
            **gendis_args,
            "top_k": top_k,
            "subgroup_search": subgroup_search,
            "mut_ops": mut_ops,
            "cx_ops": cx_ops,
            "run_id": run_num,
            "random_seed": run_num,
        }
        X_input, y_input = preprocess_input(X_train, y_train)
        gendis = GeneticExtractor(**args)

        # Doing this so the info logs inside fit do not pollute the logs
        t0 = time.time()
        gendis.fit(X_input, y_input)
        t1 = time.time()

        # Log results
        logging.info(
            f"Finished training for run={run_num}, it={gendis.it}, time={t1-t0}"
        )

        coverage_count = np.sum(gendis.top_k.coverage > 0)
        coverage_counts.append({"run": run_num, "coverage_count": coverage_count})

        pop_history.extend(gendis.history)
        topk_history.extend(gendis.top_k.stats_history)

        run_path = os.path.join(results_folder, f"run_{run_num}")
        os.makedirs(run_path)

        for ind_num, individual in enumerate(gendis.top_k.subgroups):

            img_path = join(run_path, f"histogram_sg_{ind_num}.png")
            plot_target_histogram_per_subgroup(
                y_train, individual.subgroup, img_path=img_path, bins=bins
            )

            predominant_class = class_predominance(
                individual.subgroup, labels=labels, n=1
            )

            class_name = predominant_class.index[0]
            ind_class = {
                "run": run_num,
                "top_k_subgroup": ind_num,
                "size": sum(individual.subgroup),
                "class": class_name,
                "proportion": predominant_class.values[0],
                "recall": recall(individual.subgroup, labels, target_class=class_name),
            }
            classes.append(ind_class)

            ind_metrics = evaluate_subgroup(individual.subgroup, labels=labels)
            for metric in ind_metrics:
                metric["top_k_subgroup"] = ind_num
                metric["run"] = run_num

            metrics.extend(ind_metrics)

    classes_df = pd.DataFrame(classes)
    classes_df.to_csv(join(results_folder, f"classes.csv"))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(join(results_folder, f"classes_metrics.csv"))

    coverage_counts_df = pd.DataFrame(coverage_counts)
    coverage_counts_df.to_csv(join(results_folder, f"coverage.csv"))

    pop_history_df = pd.DataFrame(pop_history)
    pop_history_df.to_csv(join(results_folder, f"pop_history.csv"))

    topk_history_df = pd.DataFrame(topk_history)
    topk_history_df.to_csv(join(results_folder, f"topk_history.csv"))


if __name__ == "__main__":
    main()
