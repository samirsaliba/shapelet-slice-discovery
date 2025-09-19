import pandas as pd
import numpy as np


def class_predominance(mask, labels, n=None):
    """
    Computes the predominance of each class within a subgroup.

    Parameters:
        mask (pd.Series or np.ndarray): Boolean mask for the subgroup.
        labels (pd.Series or np.ndarray): Ground truth labels.
        n (int or None): The n-most frequent classes to return. If none, returns all.

    Returns:
        dict: A dictionary with class labels as keys and their proportions as values.
    """
    labels = pd.Series(labels) if not isinstance(labels, pd.Series) else labels

    subgroup_labels = labels[mask]
    total = len(subgroup_labels)
    if total == 0:
        return {}
    predominance = subgroup_labels.value_counts(normalize=True)

    if n is not None:
        return predominance.iloc[:n]
    return predominance


def precision(mask, labels, target_class):
    """
    Computes the precision for a specific class in the subgroup.

    Parameters:
        mask (pd.Series or np.ndarray): Boolean mask for the subgroup.
        labels (pd.Series or np.ndarray): Ground truth labels.
        target_class: The class for which precision is computed.

    Returns:
        float: Precision value.
    """
    subgroup_labels = labels[mask]
    true_positives = (subgroup_labels == target_class).sum()
    total_selected = len(subgroup_labels)
    return true_positives / total_selected if total_selected > 0 else 0.0


def recall(mask, labels, target_class):
    """
    Computes the recall for a specific class in the subgroup.

    Parameters:
        mask (pd.Series or np.ndarray): Boolean mask for the subgroup.
        labels (pd.Series or np.ndarray): Ground truth labels.
        target_class: The class for which recall is computed.

    Returns:
        float: Recall value.
    """
    subgroup_labels = labels[mask]
    true_positives = (subgroup_labels == target_class).sum()
    total_actual = (labels == target_class).sum()
    return true_positives / total_actual if total_actual > 0 else 0.0


def f1_score(mask, labels, target_class):
    """
    Computes the F1-score for a specific class in the subgroup.

    Parameters:
        mask (pd.Series or np.ndarray): Boolean mask for the subgroup.
        labels (pd.Series or np.ndarray): Ground truth labels.
        target_class: The class for which F1-score is computed.

    Returns:
        float: F1-score value.
    """
    p = precision(mask, labels, target_class)
    r = recall(mask, labels, target_class)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_subgroup(mask, labels):
    """
    Computes precision, recall, and F1-score for all classes in a dataset.

    Parameters:
        mask (pd.Series or np.ndarray): Boolean mask for the subgroup.
        labels (pd.Series or np.ndarray): Ground truth labels.

    Returns:
        list: A list of dicts, where each item corresponds to a class
        and the dict contains class name, precision, recall,
        and F1-score for that class.
    """
    # Ensure inputs are pandas Series for consistency
    mask = pd.Series(mask) if not isinstance(mask, pd.Series) else mask
    labels = pd.Series(labels) if not isinstance(labels, pd.Series) else labels
    subgroup_labels = labels[mask]

    results = []
    unique_classes = labels.unique()

    for target_class in unique_classes:
        count = (subgroup_labels == target_class).sum()
        p = precision(mask, labels, target_class)
        r = recall(mask, labels, target_class)
        f1 = f1_score(mask, labels, target_class)

        results.append(
            {
                "class": target_class,
                "count": count,
                "precision": p,
                "recall": r,
                "f1_score": f1,
            }
        )

    return results


def compute_shapelet_subgroup_metrics(errors, predicted_mask, original_mask):
    """
    Compute interestingness metrics for a shapelet-based subgroup,
    and compare to the original subgroup mask for coverage and precision.

    Parameters
    ----------
    errors : np.ndarray of shape (n_samples,)
        Error values over which metrics are computed.

    predicted_mask : np.ndarray of bool
        Boolean array of predicted subgroup membership (e.g., from classifier).

    original_mask : np.ndarray of bool
        Boolean array of original subgroup membership (e.g., from beam_df.subgroup.covers()).

    Returns
    -------
    dict
        Dictionary of error-based metrics and overlap scores.
    """
    sg_errors = errors[predicted_mask]
    size_sg = predicted_mask.sum()
    size_total = len(errors)

    mean_sg = sg_errors.mean()
    mean_total = errors.mean()
    std_sg = sg_errors.std()
    std_total = errors.std()
    median_sg = np.median(sg_errors)
    median_total = np.median(errors)
    max_sg = sg_errors.max()
    max_total = errors.max()
    min_sg = sg_errors.min()
    min_total = errors.min()

    # Overlap metrics
    intersection = np.logical_and(predicted_mask, original_mask).sum()
    original_size = original_mask.sum()

    coverage = intersection / original_size if original_size > 0 else 0.0
    precision = intersection / size_sg if size_sg > 0 else 0.0
    recall = coverage
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    return {
        "size_sg": int(size_sg),
        "size_dataset": int(size_total),
        "mean_sg": float(mean_sg),
        "mean_dataset": float(mean_total),
        "std_sg": float(std_sg),
        "std_dataset": float(std_total),
        "median_sg": float(median_sg),
        "median_dataset": float(median_total),
        "max_sg": float(max_sg),
        "max_dataset": float(max_total),
        "min_sg": float(min_sg),
        "min_dataset": float(min_total),
        "mean_lift": float(mean_sg - mean_total),
        "median_lift": float(median_sg - median_total),
        "coverage": float(coverage),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def compute_jaccard_matrix(index_sets):
    """
    Compute the pairwise Jaccard index between sets of indices.

    Parameters
    ----------
    index_sets : list of sets
        Output from `get_subgroup_coverage_indices`.

    Returns
    -------
    np.ndarray
        Symmetric matrix of Jaccard indices, shape (n_subgroups, n_subgroups).
    """
    n = len(index_sets)
    jaccard_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            inter = len(index_sets[i] & index_sets[j])
            union = len(index_sets[i] | index_sets[j])
            jaccard = inter / union if union > 0 else 0.0
            jaccard_matrix[i, j] = jaccard
            jaccard_matrix[j, i] = jaccard  # symmetry

    return jaccard_matrix


def summarize_jaccard_matrix(jaccard_df):
    n = jaccard_df.shape[0]
    mask = ~np.eye(n, dtype=bool)  # exclude diagonal
    vals = jaccard_df.values[mask]
    return {"mean": vals.mean(), "std": vals.std(), "min": vals.min()}
