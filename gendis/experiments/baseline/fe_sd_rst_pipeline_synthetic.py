import datetime
import joblib
import multiprocessing
import numpy as np
import os
import pandas as pd
import pysubgroup as ps
import time

from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
from collections import defaultdict
from gendis.evaluation import class_predominance, evaluate_subgroup
from os.path import join
from sklearn.metrics import classification_report
from sklearn.tree import export_text

from helpers.plot import (
    plot_decision_surface_2d,
    plot_shapelet_distance_scatter_2d,
    plot_shapelets_on_series,
    plot_subgroup_series_with_defining_shapelets,
    plot_shapelet_distance_density_bubbleplot,
    plot_heatmap_jaccard,
)

from helpers.subgroup_evaluation import (
    compute_jaccard_matrix,
    compute_shapelet_subgroup_metrics,
    summarize_jaccard_matrix,
)

from helpers.tree import plot_decision_tree_model, get_shapelet_tree_clf

from helpers.util import (
    DROP_COLUMNS,
    RANDOM_STATE,
    analyze_shapelet_distances_for_subgroup,
    get_sg_label_column,
    get_subgroup_coverage_indices,
    group_shapelets_by_class,
    save_json,
    save_pickle,
)


def main(base_name):
    series_path = f"./data/{base_name}.csv"
    features_path = f"./processed/{base_name}_tsfresh.csv"

    timestamp_format = "%Y%m%d_%H%M"
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    output_folder = f"./results/{base_name}_{timestamp}"

    os.makedirs(output_folder, exist_ok=False)

    PARAMETERS = {
        "random_state": RANDOM_STATE,
        "n_jobs": multiprocessing.cpu_count() - 3,
        "sg_alpha": 0.5,
        "sg_beam_width": 100,
        "sg_result_set_size": 10,
        "sg_depth": 2,
        "rst_max_shapelets": 4,
        "rst_n_shapelet_samples": 10_000,
        "rst_time_limit_minutes": 3,
        "rst_remove_self_similar": True,
        "rst_min_shap_len_mult": 0.05,
        "rst_max_shap_len_mult": 0.15,
        "tree_depth": 2,
        "plot_shapelets_k": 5,
    }
    save_json(PARAMETERS, join(output_folder, "parameters.json"))

    RESULTS = {}

    start_time = time.time()

    df = pd.read_csv(series_path)
    series = df.drop(columns=DROP_COLUMNS, errors="ignore").reset_index()

    # Subgroup Discovery
    selected_features_df = pd.read_csv(features_path, index_col=0)
    target = ps.NumericTarget("error")
    searchspace = ps.create_selectors(selected_features_df, ignore=["error"])
    qf = ps.StandardQFNumeric(a=PARAMETERS["sg_alpha"])

    task = ps.SubgroupDiscoveryTask(
        selected_features_df,
        target,
        searchspace,
        result_set_size=PARAMETERS["sg_result_set_size"],
        depth=PARAMETERS["sg_depth"],
        qf=qf,
    )

    beam_result = ps.BeamSearch(beam_width=PARAMETERS["sg_beam_width"]).execute(task)
    beam_df = beam_result.to_dataframe()
    beam_df.to_csv(join(output_folder, "beam_search_df.csv"))

    print(f"[INFO] Beam search completed with {len(beam_df)} subgroups.")

    X = series.drop(columns="index")
    reports = defaultdict(dict)
    subgroup_classes = []
    subgroup_metrics = []

    shap_vs_sg_metrics = []

    shapelet_subgroup_classes = []
    shapelet_subgroup_metrics = []
    shapelet_index_sets = {}

    for sg in range(PARAMETERS["sg_result_set_size"]):
        sg_folder = join(output_folder, f"subgroup_{sg}")
        os.makedirs(sg_folder, exist_ok=False)

        subgroup_mask = get_sg_label_column(
            sg_df=beam_df, features_df=selected_features_df, subgroup_index=sg
        )

        # Comparing the Subgroup Mask vs Problem Labels
        sg_predominant_class = class_predominance(
            subgroup_mask, labels=df["label"], n=1
        )

        sg_class = {
            "subgroup": sg,
            "class": sg_predominant_class.index[0],
            "proportion": sg_predominant_class.values[0],
            "count": subgroup_mask.sum(),
        }
        reports[f"subgroup_{sg}"]["subgroup_x_labels_info"] = sg_class
        subgroup_classes.append(sg_class)

        sg_label_metrics = evaluate_subgroup(subgroup_mask, labels=df["label"])
        for metric in sg_label_metrics:
            metric["top_k_subgroup"] = sg
        subgroup_metrics.extend(sg_label_metrics)
        reports[f"subgroup_{sg}"]["subgroup_x_labels_metrics"] = sg_label_metrics

        # Random Shapelet Transform
        rst = RandomShapeletTransform(
            max_shapelets=PARAMETERS["rst_max_shapelets"],
            n_shapelet_samples=PARAMETERS["rst_n_shapelet_samples"],
            remove_self_similar=PARAMETERS["rst_remove_self_similar"],
            time_limit_in_minutes=PARAMETERS["rst_time_limit_minutes"],
            min_shapelet_length=int(PARAMETERS["rst_min_shap_len_mult"] * X.shape[1]),
            max_shapelet_length=int(PARAMETERS["rst_max_shap_len_mult"] * X.shape[1]),
            n_jobs=PARAMETERS["n_jobs"],
            random_state=PARAMETERS["random_state"],
        )
        rst.fit(X, y=subgroup_mask)
        rst.save(join(sg_folder, f"sg{sg}_RST"))

        print(f"[INFO] RST shapelets extracted: {len(rst.shapelets)}")

        X_distances = rst.transform(X)

        shapelet_dict = group_shapelets_by_class(rst.shapelets)
        save_pickle(shapelet_dict, join(sg_folder, "shapelet_dict.pickle"))

        plot_shapelets_on_series(
            X=X,
            shapelets=shapelet_dict[False],
            img_path=join(sg_folder, "shaps_in_subgroup_false.png"),
        )

        plot_shapelets_on_series(
            X=X,
            shapelets=shapelet_dict[True],
            img_path=join(sg_folder, "shaps_in_subgroup_true.png"),
        )

        beam_index_sets = get_subgroup_coverage_indices(
            subgroups=beam_df.subgroup, data=selected_features_df
        )
        save_pickle(beam_index_sets, join(sg_folder, "beam_index_sets.pickle"))

        json_friendly_index_sets = {
            int(k): [int(i) for i in v] for k, v in beam_index_sets.items()
        }
        save_json(json_friendly_index_sets, join(sg_folder, "beam_index_sets.json"))

        class_label = True
        df_single = analyze_shapelet_distances_for_subgroup(
            distances=X_distances,
            subgroup_idx=sg,
            index_sets=beam_index_sets,
            shapelet_dict=shapelet_dict,
            class_label=class_label,
        )
        df_single = df_single.sort_values("delta", ascending=False)
        df_single.to_csv(join(sg_folder, f"sg{sg}_shapelet_distances_stats.csv"))

        shap_ids = list(shapelet_dict[True].keys())

        plot_shapelet_distance_scatter_2d(
            distances=X_distances,
            subgroup_indexes=list(beam_index_sets[sg]),
            shapelet_ids=tuple(shap_ids),
            img_path=join(sg_folder, f"sg{sg}_shapelet_distances_scatter.png"),
        )

        plot_shapelet_distance_density_bubbleplot(
            distances=X_distances,
            subgroup_indexes=list(beam_index_sets[sg]),
            shapelet_ids=tuple(shap_ids),
            img_path=join(sg_folder, f"sg{sg}_shapelet_distances_bubble.png"),
            bins=20,
        )

        # Tree Classifier for Sg
        clf = get_shapelet_tree_clf(
            X_distances,
            subgroup_mask,
            shap_ids,
            tree_depth=PARAMETERS["tree_depth"],
            random_state=PARAMETERS["random_state"],
        )
        print(
            f"[INFO] Classifier accuracy: {clf.score(X_distances[:, shap_ids], subgroup_mask):.4f}"
        )
        joblib.dump(clf, join(sg_folder, f"sg{sg}_tree_classifier.pkl"))

        report = classification_report(
            subgroup_mask, clf.predict(X_distances[:, shap_ids]), output_dict=True
        )
        reports[f"subgroup_{sg}"]["tree_clf"] = report

        tree_text = export_text(clf, feature_names=[f"D_{i}" for i in shap_ids])

        with open(join(sg_folder, f"sg{sg}_tree_rules.txt"), "w") as f:
            f.write(tree_text)

        plot_decision_tree_model(
            clf=clf,
            feature_names=[f"D_{i}" for i in shap_ids],
            img_path=join(sg_folder, f"sg{sg}_tree_decision_tree.png"),
        )

        plot_decision_surface_2d(
            clf=clf,
            distances=X_distances,
            y_mask=subgroup_mask,
            shapelet_ids=shap_ids,
            img_path=join(sg_folder, f"sg{sg}_tree_decision_surface.png"),
        )

        shapelet_subgroup_mask = clf.predict(X_distances[:, shap_ids]).astype(bool)

        if shapelet_subgroup_mask.sum() == 0:
            print(f"[WARNING] Shapelet subgroup #{sg} has 0 members. Skipping metrics.")
            reports[f"subgroup_{sg}"]["shapelet_x_beam_metrics"] = {
                "size_sg": 0,
                "coverage": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }
            reports[f"subgroup_{sg}"]["shapelet_x_labels_info"] = {
                "subgroup": sg,
                "class": None,
                "proportion": 0.0,
                "count": 0,
            }
            reports[f"subgroup_{sg}"]["shapelet_x_labels_metrics"] = []
            continue

        shapelet_index_sets[sg] = set(np.where(shapelet_subgroup_mask)[0])

        # Comparing the Shapelet Subgroup vs Beam (Features) Subgroup
        shap_vs_sg = compute_shapelet_subgroup_metrics(
            errors=df["error"].to_numpy(),
            predicted_mask=shapelet_subgroup_mask,
            original_mask=subgroup_mask.astype(bool),
        )

        reports[f"subgroup_{sg}"]["shapelet_x_beam_metrics"] = shap_vs_sg
        shap_vs_sg_metrics.append(shap_vs_sg)

        # Comparing the Shapelet Subgroup vs Problem Labels
        shap_predominant_class = class_predominance(
            shapelet_subgroup_mask, labels=df["label"], n=1
        )

        shapelet_class = {
            "subgroup": sg,
            "class": shap_predominant_class.index[0],
            "proportion": shap_predominant_class.values[0],
            "count": shapelet_subgroup_mask.sum(),
        }
        reports[f"subgroup_{sg}"]["shapelet_x_labels_info"] = shapelet_class
        shapelet_subgroup_classes.append(shapelet_class)

        shap_label_metrics = evaluate_subgroup(
            shapelet_subgroup_mask, labels=df["label"]
        )
        for metric in shap_label_metrics:
            metric["top_k_subgroup"] = sg
        shapelet_subgroup_metrics.extend(shap_label_metrics)
        reports[f"subgroup_{sg}"][
            "shapelet_x_labels_metrics"
        ] = shapelet_subgroup_metrics

        k = PARAMETERS["plot_shapelets_k"]
        plot_subgroup_series_with_defining_shapelets(
            X=X,
            shapelets=shapelet_dict[True],
            subgroup_indices=list(beam_index_sets[0]),
            distances=X_distances,
            K=k,
            img_path=join(sg_folder, f"sg{sg}_top{k}_matches.png"),
        )

    ## Subgroup
    pd.DataFrame(subgroup_metrics).to_csv(
        join(output_folder, "subgroups_metrics.csv"), index=False
    )

    pd.DataFrame(subgroup_classes).to_csv(join(output_folder, f"subgroup_classes.csv"))

    ## Shapelet
    pd.DataFrame(shap_vs_sg_metrics).to_csv(
        join(output_folder, f"shapelet_versus_subgroup_classes.csv")
    )

    pd.DataFrame(shapelet_subgroup_classes).to_csv(
        join(output_folder, f"shapelet_defined_sg_classes.csv")
    )

    pd.DataFrame(shapelet_subgroup_metrics).to_csv(
        join(output_folder, f"shapelet_defined_sg_metrics.csv")
    )

    # Jaccard Index, Beam Subgroups
    jac_beam_df = pd.DataFrame(
        compute_jaccard_matrix(beam_index_sets),
        index=range(len(beam_index_sets)),
        columns=range(len(beam_index_sets)),
    )
    jac_beam_df.to_csv(join(output_folder, "jaccard_matrix_beam_subgroups.csv"))
    plot_heatmap_jaccard(
        jac_beam_df, img_path=join(output_folder, "jaccard_heatmap.png")
    )

    jac_shap_df = pd.DataFrame(
        compute_jaccard_matrix(list(shapelet_index_sets.values())),
        index=list(shapelet_index_sets.keys()),
        columns=list(shapelet_index_sets.keys()),
    )

    jac_shap_df.to_csv(join(output_folder, "jaccard_matrix_shapelet_subgroups.csv"))

    finish_time = time.time()

    RESULTS["jaccard_matrix_summary_beam_subgroups"] = summarize_jaccard_matrix(
        jac_beam_df
    )
    RESULTS["jaccard_matrix_summary_shapelet_subgroups"] = summarize_jaccard_matrix(
        jac_shap_df
    )
    RESULTS["total_time_in_seconds"] = "{:.2f}".format(finish_time - start_time)
    RESULTS["reports"] = reports

    save_json(RESULTS, join(output_folder, "results.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subgroup Discovery + RST Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset base name (without extension). E.g., NonInvasiveFetalECGThorax2_tsforest_errors",
    )
    args = parser.parse_args()

    main(args.dataset)
