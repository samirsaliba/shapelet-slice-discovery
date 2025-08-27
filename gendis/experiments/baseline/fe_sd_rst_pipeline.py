import datetime
import joblib
import multiprocessing
import os
from os.path import join
import pandas as pd
import time

import pysubgroup as ps
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
from sklearn.metrics import classification_report
from sklearn.tree import export_text

from helpers.plot import (
    plot_decision_surface_2d,
    plot_shapelet_distance_scatter_2d,
    plot_shapelets_on_series,
    plot_subgroup_series_with_defining_shapelets,
)

from helpers.tree import plot_decision_tree_model, get_shapelet_tree_clf

from helpers.util import (
    analyze_shapelet_distances_for_subgroup,
    get_sg_label_column,
    get_subgroup_coverage_indices,
    group_shapelets_by_class,
    save_json,
    save_pickle,
)

from helpers.subgroup_evaluation import (
    compute_jaccard_matrix,
    compute_shapelet_subgroup_metrics,
    summarize_jaccard_matrix,
)


def main(base_name):
    series_path = f"./data/{base_name}.csv"
    features_path = f"./processed/{base_name}_tsfresh.csv"

    timestamp_format = "%Y%m%d_%H%M"
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    output_folder = f"./results/{base_name}_{timestamp}"

    os.makedirs(output_folder, exist_ok=False)

    PARAMETERS = {
        "random_state": 0,
        "n_jobs": multiprocessing.cpu_count() - 3,
        "sg_alpha": 0.5,
        "sg_beam_width": 100,
        "sg_result_set_size": 10,
        "sg_depth": 2,
        "rst_max_shapelets": 4,
        "rst_n_shapelet_samples": 1_000,
        "rst_time_limit_minutes": 1,
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
    series = df.drop(columns=["label", "predicted", "error"]).reset_index()

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
    reports = {}
    shap_sg_metrics = []

    for sg in range(PARAMETERS["sg_result_set_size"]):
        sg_folder = join(output_folder, f"subgroup_{sg}")
        os.makedirs(sg_folder, exist_ok=False)

        sg_y = get_sg_label_column(
            sg_df=beam_df, features_df=selected_features_df, subgroup_index=sg
        )

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
        rst.fit(X, y=sg_y)
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

        index_sets = get_subgroup_coverage_indices(
            subgroups=beam_df.subgroup, data=selected_features_df
        )
        save_pickle(index_sets, join(sg_folder, "index_sets.pickle"))

        json_friendly_index_sets = {
            int(k): [int(i) for i in v] for k, v in index_sets.items()
        }
        save_json(json_friendly_index_sets, join(sg_folder, "index_sets.json"))

        class_label = True
        df_single = analyze_shapelet_distances_for_subgroup(
            distances=X_distances,
            subgroup_idx=sg,
            index_sets=index_sets,
            shapelet_dict=shapelet_dict,
            class_label=class_label,
        )
        df_single = df_single.sort_values("delta", ascending=False)
        df_single.to_csv(join(sg_folder, f"sg{sg}_shapelet_distances_stats.csv"))

        shap_ids = list(shapelet_dict[True].keys())

        plot_shapelet_distance_scatter_2d(
            distances=X_distances,
            subgroup_indexes=list(index_sets[sg]),
            shapelet_ids=tuple(shap_ids),
            img_path=join(sg_folder, f"sg{sg}_shapelet_distances_scatter.png"),
        )

        # Tree Classifier for Sg
        clf = get_shapelet_tree_clf(
            X_distances,
            sg_y,
            shap_ids,
            tree_depth=PARAMETERS["tree_depth"],
            random_state=PARAMETERS["random_state"],
        )
        print(
            f"[INFO] Classifier accuracy: {clf.score(X_distances[:, shap_ids], sg_y):.4f}"
        )
        joblib.dump(clf, join(sg_folder, f"sg{sg}_tree_classifier.pkl"))

        report = classification_report(
            sg_y, clf.predict(X_distances[:, shap_ids]), output_dict=True
        )
        reports[f"subgroup_{sg}"] = report

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
            y_mask=sg_y,
            shapelet_ids=shap_ids,
            img_path=join(sg_folder, f"sg{sg}_tree_decision_surface.png"),
        )

        shapelet_mask = clf.predict(X_distances[:, shap_ids]).astype(bool)

        shapelet_metrics = compute_shapelet_subgroup_metrics(
            errors=df["error"].to_numpy(),
            predicted_mask=shapelet_mask,
            original_mask=sg_y.astype(bool),
        )

        reports[f"subgroup_{sg}"]["shapelet_metrics"] = shapelet_metrics
        shap_sg_metrics.append(shapelet_metrics)

        k = PARAMETERS["plot_shapelets_k"]
        plot_subgroup_series_with_defining_shapelets(
            X=X,
            shapelets=shapelet_dict[True],
            subgroup_indices=list(index_sets[0]),
            distances=X_distances,
            K=k,
            img_path=join(sg_folder, f"sg{sg}_top{k}_matches.png"),
        )

    pd.DataFrame(shap_sg_metrics).to_csv(
        join(output_folder, "shapelet_subgroups_metrics.csv"), index=False
    )

    # Jaccard Index
    jaccard_mat = compute_jaccard_matrix(index_sets)
    jaccard_df = pd.DataFrame(
        jaccard_mat, index=range(len(index_sets)), columns=range(len(index_sets))
    )
    jaccard_df.to_csv(join(output_folder, "jaccard_matrix.csv"))

    finish_time = time.time()

    RESULTS["jaccard_matrix_summary"] = summarize_jaccard_matrix(jaccard_df)
    RESULTS["total_time_in_seconds"] = "{:.2f}".format(finish_time - start_time)
    RESULTS["tree_clf_reports"] = reports

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
