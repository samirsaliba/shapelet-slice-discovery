import argparse
import logging
import os
from os.path import basename, join, splitext
import pandas as pd

from gendis.processing import preprocess_input
from gendis.evaluation import (
    class_predominance,
    evaluate_subgroup,
    get_jaccard_df_summary,
    summarize_shapelet_distances,
)
from gendis.visualization import (
    plot_best_matching_shaps,
    plot_coverage_heatmap,
    plot_jaccard_heatmap,
    plot_subgroup_alignment_comparison,
    plot_target_histogram_overall,
)
from gendis.genetic import GeneticExtractor
from util import setup_logging, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path", required=True, help="CSV file with dataset"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to saved gendis.pickle"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file_path = args.input_file_path
    model_path = args.model_path

    results_folder = join(os.path.dirname(model_path), "post-process")

    timestamp = "postprocess"

    setup_logging(results_folder, timestamp)
    logging.info("Postprocessing started")

    df = pd.read_csv(input_file_path)

    img_path = join(results_folder, "error_dist.pdf")
    labels = df["label"].unique()

    plot_target_histogram_overall(
        df, target_col="error", bins=20, ymax=2500, img_path=img_path
    )

    logging.info("Loaded data")
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
    y = df["error"]

    X_input, y_input = preprocess_input(X, y)

    gendis = GeneticExtractor.load(model_path)
    logging.info("Loaded trained model")

    # Coverage heatmap
    plot_coverage_heatmap(
        gendis.top_k.subgroups,
        img_path=join(results_folder, "coverage_heatmap_post.pdf"),
        cmap="YlGnBu",
    )

    # Jaccard
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
            X_input, shapelets=individual, thresholds=individual.thresholds
        )

        distance_metrics = summarize_shapelet_distances(
            distances=distances, subgroup=subgroup
        )
        topk_dist_metrics[f"subgroup_{i}"] = distance_metrics

        plot_best_matching_shaps(
            X,
            distances,
            subgroup,
            individual,
            img_path=join(results_folder, f"sg_{i}_top_members_post.pdf"),
        )

        alignment_plot_data = plot_subgroup_alignment_comparison(
            X,
            distances,
            subgroup,
            individual,
            img_path=join(results_folder, f"sg_{i}_shap_alignment_comparison_post.pdf"),
        )

        alignment_plot_data["shapelet_thresholds"] = individual.thresholds
        topk_alignment_plots_data[f"subgroup_{i}"] = alignment_plot_data

        pred_class = class_predominance(subgroup, labels=labels, n=1)
        topk_classes.append(
            {
                "top_k_subgroup": i,
                "class": pred_class.index[0],
                "proportion": pred_class.values[0],
            }
        )

        metrics = evaluate_subgroup(subgroup, labels=labels)
        for metric in metrics:
            metric["top_k_subgroup"] = i
        topk_metrics.extend(metrics)

    save_json(
        topk_dist_metrics, join(results_folder, "topk_distance_metrics_post.json")
    )
    save_json(
        topk_alignment_plots_data,
        join(results_folder, f"topk_alignment_plots_data.json"),
    )
    pd.DataFrame(topk_classes).to_csv(
        join(results_folder, "topk_classes_post.csv"), index=False
    )
    pd.DataFrame(topk_metrics).to_csv(
        join(results_folder, "topk_classes_metrics_post.csv"), index=False
    )

    logging.info("Postprocessing completed")


if __name__ == "__main__":
    main()
