from collections import defaultdict
from functools import wraps
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


DROP_COLUMNS = [
    "label",
    "predicted",
    "error",
    "pattern_x0",
    "pattern_x1",
    "pattern_y",
]

RANDOM_STATE = 0


def get_sg_label_column(sg_df, features_df, subgroup_index):
    description = sg_df["subgroup"][subgroup_index]
    return description.covers(features_df)


def group_shapelets_by_class(shapelets):
    """
    Organize RandomShapeletTransform shapelets into a dictionary by class label.

    Parameters
    ----------
    shapelets : list of tuples
        Output from rst.shapelets. Each tuple contains:
        (info_gain, length, start_pos, dim, extracted_idx, class_val, z_norm_shapelet)

    Returns
    -------
    dict
        Dictionary where keys are class labels and values are lists of shapelet tuples.
    """
    shapelet_dict = defaultdict(dict)
    for shap_i, shap in enumerate(shapelets):
        class_val = shap[5]  # class value
        shapelet_dict[class_val][shap_i] = shap
    return dict(shapelet_dict)


def auto_savefig(func):
    """
    Decorator that adds optional saving and suppressing of plt.show().
    If the decorated function receives an `img_path` keyword argument,
    it will save the current figure there instead of showing it.

    Usage:
    @auto_savefig
    def plot_something(..., img_path=None):
        ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        img_path = kwargs.pop("img_path", None)
        func(*args, **kwargs)

        if img_path:
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    return wrapper


def get_subgroup_coverage_indices(subgroups, data):
    """
    Get the row indices covered by each subgroup.

    Parameters
    ----------
    subgroups : list
        List of pysubgroup.subgroup_description.Conjunction objects.

    data : pd.DataFrame or np.ndarray
        Dataset used to evaluate coverage.

    Returns
    -------
    dict
        Dictionary mapping subgroup index to the set of covered instance indices.
    """
    return {idx: set(np.where(sg.covers(data))[0]) for idx, sg in enumerate(subgroups)}


def analyze_shapelet_distances_for_subgroup(
    distances, subgroup_idx, index_sets, shapelet_dict, class_label
):
    """
    Analyze the shapelet distances for a given class label in one specific subgroup.

    Parameters
    ----------
    distances : np.ndarray
        Distance matrix of shape (n_samples, n_shapelets)

    subgroup_idx : int
        Index of the subgroup to analyze

    index_sets : dict
        Mapping from subgroup index to sets of instance indices

    shapelet_dict : dict
        Mapping from class label to {shapelet_id: shapelet_tuple}

    class_label : any
        The class whose shapelets we want to analyze

    Returns
    -------
    pd.DataFrame
        Table showing mean distance in and out of the selected subgroup for each shapelet.
    """
    rows_in = sorted(list(index_sets[subgroup_idx]))
    rows_out = sorted(list(set(range(distances.shape[0])) - set(rows_in)))

    results = []

    for shap_id in shapelet_dict[class_label].keys():
        dist_in = distances[rows_in, shap_id]
        dist_out = distances[rows_out, shap_id]

        result = {
            "subgroup": subgroup_idx,
            "shapelet_id": shap_id,
            "mean_in": np.mean(dist_in),
            "std_in": np.std(dist_in),
            "mean_out": np.mean(dist_out),
            "std_out": np.std(dist_out),
            "delta": np.mean(dist_out) - np.mean(dist_in),
        }
        results.append(result)

    return pd.DataFrame(results)


def save_pickle(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)
