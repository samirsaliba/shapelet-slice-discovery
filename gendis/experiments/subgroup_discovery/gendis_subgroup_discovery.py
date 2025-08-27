from datetime import datetime
import logging
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

from gendis.evaluation import (
    class_predominance,
    evaluate_subgroup,
    get_jaccard_df_summary,
    precision,
    recall,
    summarize_shapelet_distances,
)
from gendis.visualization import (
    plot_best_matching_shaps,
    plot_coverage_heatmap,
    plot_jaccard_heatmap,
    plot_shaps,
    plot_subgroup_alignment_comparison,
    plot_target_histograms_in_batches,
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

    img_path = join(results_folder, "error_dist")
    labels = df["label"].unique()

    plot_target_histograms_in_batches(
        df,
        label_col="label",
        target_col="error",
        bins=20,
        max_labels_per_plot=6,
        img_path=img_path,
    )

    labels = df["label"]

    X = df.drop(
        columns=[
            "pattern_x0",
            "pattern_x1",
            "pattern_y",
            "error",
            "label",
            "predicted",
        ],
        errors="ignore",
    ).values

    y = df["error"].values

    CACHE_SIZE = 8192

    # Subgroup Search parameters
    K = 10
    COVERAGE_ALPHA = 0.5
    SUBGROUP_SIZE_BETA = 0.5
    THRESHOLD_MAX_IT = None
    THRESHOLD_KAPPA = 0.9
    SEARCH_MODE = "percentile"
    subgroup_args = {
        "K": K,
        "coverage_alpha": COVERAGE_ALPHA,
        "subgroup_size_beta": SUBGROUP_SIZE_BETA,
        "cache_size": CACHE_SIZE,
        "random_seed": RANDOM_SEED_VAL,
        "max_it": THRESHOLD_MAX_IT,
        "threshold_kappa": THRESHOLD_KAPPA,
        "threshold_search_mode": SEARCH_MODE,
    }

    logging.info("Parameters TopKSubgroups and SubgroupDistance")
    logging.info(subgroup_args)

    subgroup_search = SubgroupSearch(
        distance_function=SubgroupSearch.mean_shift,
        threshold_search_mode=SEARCH_MODE,
        threshold_kappa=THRESHOLD_KAPPA,
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
        slide_shapelet,
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
        "population_size": 500,
        "iterations": 500,
        "mutation_prob": 1,
        "crossover_prob": 0,
        "max_shaps": 2,
        "wait": None,
        "pop_restarts": None,
        "min_len": 0.1,
        "max_len": 0.25,
        "n_jobs": 1,
        "cache_size": CACHE_SIZE,
        "verbose": False,
        "random_seed": RANDOM_SEED_VAL,
    }
    logging.info("Parameters Gendis")
    logging.info(gendis_args)

    # Preprocess and model fit
    args = {**gendis_args, **funcs}
    gendis = GeneticExtractor(**args)
    t0 = time.time()
    gendis.fit(X, y)
    t1 = time.time()

    save_json(
        {**gendis_args, **subgroup_args, "training_time": round(t1 - t0)},
        join(results_folder, "run_info.json"),
    )

    # Log results
    logging.info(f"Finished training, it={gendis.it}, time={t1-t0}\n\n\n")
    logging.info("Best individual stats")
    logging.info(gendis.best["info"])
    save_json(gendis.best["info"], join(results_folder, "best_info.json"))

    logging.info("Top-K info")
    topk_info = []
    for individual in gendis.top_k.subgroups:
        data = individual.info
        data["coverage_weight"] = individual.coverage_weight
        # data["subgroup_indexes"] = np.where(individual.subgroup)
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
    img_path = join(results_folder, f"coverage_heatmap.pdf")
    plot_coverage_heatmap(gendis.top_k.subgroups, img_path=img_path, cmap="YlGnBu")

    jaccard_df, jaccard_summary = get_jaccard_df_summary(gendis.top_k.to_dict())
    jaccard_df.to_csv(join(results_folder, "jaccard_matrix.csv"))
    save_json(jaccard_summary, join(results_folder, "jaccard_matrix_summary.json"))

    img_path = join(results_folder, f"jaccard_heatmap.pdf")
    plot_jaccard_heatmap(jaccard_df, img_path=img_path)

    topk_classes = []
    topk_metrics = []
    topk_dist_metrics = {}
    topk_alignment_plots_data = {}

    for i, individual in enumerate(gendis.top_k.subgroups):
        distances, subgroup = gendis.transform(
            X, shapelets=individual, thresholds=individual.thresholds
        )

        distance_metrics = summarize_shapelet_distances(
            distances=distances, subgroup=subgroup
        )
        topk_dist_metrics[f"subgroup_{i}"] = distance_metrics

        img_path = join(results_folder, f"sg_{i}_top_members.pdf")
        plot_best_matching_shaps(
            X,
            distances,
            subgroup,
            individual,
            undiff_shapelets=False,
            img_path=img_path,
        )

        img_path = join(results_folder, f"sg_{i}_shap_alignment_comparison.pdf")
        alignment_plot_data = plot_subgroup_alignment_comparison(
            X,
            distances,
            subgroup,
            individual,
            use_mean=True,
            undiff_shapelets=False,
            img_path=img_path,
        )
        alignment_plot_data["shapelet_thresholds"] = individual.thresholds
        topk_alignment_plots_data[f"subgroup_{i}"] = alignment_plot_data

        predominant_class = class_predominance(subgroup, labels=labels, n=1)

        ind_class = {
            "top_k_subgroup": i,
            "class": predominant_class.index[0],
            "proportion": predominant_class.values[0],
        }
        topk_classes.append(ind_class)

        metrics = evaluate_subgroup(subgroup, labels=labels)
        for metric in metrics:
            metric["top_k_subgroup"] = i

        topk_metrics.extend(metrics)

        # img_path = join(results_folder, f"shaps_{i}.pdf")
        # plot_shaps(individual, img_path=img_path)

    save_json(topk_dist_metrics, join(results_folder, "topk_distance_metrics.json"))
    save_json(
        topk_alignment_plots_data,
        join(results_folder, f"topk_alignment_plots_data.json"),
    )
    classes_df = pd.DataFrame(topk_classes)
    metrics_df = pd.DataFrame(topk_metrics)

    classes_df.to_csv(join(results_folder, f"topk_classes.csv"))
    metrics_df.to_csv(join(results_folder, f"topk_classes_metrics.csv"))

    gendis.save(join(results_folder, "gendis.pickle"))


if __name__ == "__main__":
    main()
