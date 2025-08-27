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


def summarize_shapelet_distances(distances, subgroup):
    """
    Summarize shapelet distance distributions for subgroup and outside.

    Parameters
    ----------
    distances : pd.DataFrame
        DataFrame with columns D_0, D_1, ..., one for each shapelet.

    subgroup : list or np.ndarray
        Indices of instances inside the subgroup.

    Returns
    -------
    dict
        Dictionary with descriptive statistics for each shapelet, split by subgroup vs outside.
    """
    result = {}

    # Columns corresponding to shapelet distances
    distance_cols = distances.filter(like="D_").columns

    # Create boolean mask
    inside_mask = np.zeros(len(distances), dtype=bool)
    inside_mask[subgroup] = True

    for col in distance_cols:
        stats = {}

        for label, mask in [("subgroup", inside_mask), ("outside", ~inside_mask)]:
            values = distances.loc[mask, col].values

            stats[label] = {
                "min": float(np.min(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
                "q1": float(np.percentile(values, 25)),
                "q3": float(np.percentile(values, 75)),
            }

        result[col] = stats

    return result


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


def get_jaccard_df_summary(topk):

    sgs_index_sets = {
        idx: set(np.where(item["subgroup"])[0].tolist())
        for idx, item in enumerate(topk)
    }

    jaccard_df = pd.DataFrame(
        compute_jaccard_matrix(sgs_index_sets),
        index=range(len(sgs_index_sets)),
        columns=range(len(sgs_index_sets)),
    )

    return jaccard_df, summarize_jaccard_matrix(jaccard_df)
