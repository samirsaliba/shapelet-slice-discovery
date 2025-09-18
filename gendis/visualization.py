import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
from .processing import undifferentiate_series


def plot_func(func):
    def wrapper(*args, **kwargs):
        img_path = kwargs.pop("img_path", None)
        result = func(*args, **kwargs)

        if img_path is not None:
            plt.savefig(img_path, format="pdf")
        else:
            plt.show()
        plt.clf()

        return result

    return wrapper


@plot_func
def plot_target_histogram_overall(df, target_col="error", bins=20, ymax=None, **kwargs):
    """
    Plot a single histogram for the entire target column, ignoring labels.
    Optionally fixes the y-axis max to ensure visual comparability.
    """
    plt.hist(
        df[target_col],
        bins=bins,
        histtype="bar",
        alpha=0.7,
        range=(0, 1),
        **kwargs,
    )
    mean_val = np.mean(df[target_col])
    plt.axvline(
        mean_val, linestyle="dashed", linewidth=1, label=f"Mean: {mean_val:.2f}"
    )
    plt.legend()
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    if ymax is not None:
        plt.ylim(0, ymax)
    plt.title(f"Histogram of {target_col} (Overall)")


@plot_func
def plot_target_histogram(
    df, label_col="label", target_col="error", bins=10, ymax=None, **kwargs
):
    mean_val = np.mean(df[target_col])

    # Group data by label
    labels = df[label_col].unique()  # Unique labels
    data = [df.loc[df[label_col] == lb, target_col] for lb in labels]

    # Plot histogram with histtype="bar"
    plt.hist(
        data, bins=bins, histtype="bar", label=labels, alpha=0.7, range=(0, 1), **kwargs
    )

    plt.axvline(
        mean_val, linestyle="dashed", linewidth=1, label=f"Mean: {mean_val:.2f}"
    )
    plt.legend(loc="upper right")
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {target_col} by {label_col}")
    if ymax is not None:
        plt.ylim(0, ymax)


@plot_func
def _plot_single_batch(
    batch_data,
    batch_labels,
    target_col,
    label_col,
    bins,
    batch_idx,
    num_batches,
    **kwargs,
):
    """Plot a single batch of histograms."""
    plt.hist(
        batch_data,
        bins=bins,
        histtype="bar",
        label=batch_labels,
        alpha=0.7,
        range=(0, 1),
        **kwargs,
    )
    plt.legend(loc="upper right")
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.title(
        f"Histogram of {target_col} by {label_col} (Batch {batch_idx + 1}/{num_batches})"
    )


def plot_target_histograms_in_batches(
    df,
    label_col="label",
    target_col="error",
    bins=10,
    max_labels_per_plot=5,
    img_path=None,
    **kwargs,
):
    labels = sorted(df[label_col].unique())  # Ensure sequential order
    num_labels = len(labels)

    if num_labels > max_labels_per_plot:
        num_batches = math.ceil(num_labels / max_labels_per_plot)
        for batch_idx in range(num_batches):
            batch_labels = labels[
                batch_idx * max_labels_per_plot : (batch_idx + 1) * max_labels_per_plot
            ]
            batch_data = [
                df.loc[df[label_col] == lb, target_col] for lb in batch_labels
            ]

            # Generate a file path for each batch if saving
            batch_img_path = (
                f"{img_path}_batch_{batch_idx + 1}.pdf" if img_path else None
            )

            # Plot the batch using the wrapper
            _plot_single_batch(
                batch_data,
                batch_labels=batch_labels,
                target_col=target_col,
                label_col=label_col,
                bins=bins,
                batch_idx=batch_idx,
                num_batches=num_batches,
                img_path=batch_img_path,
                **kwargs,
            )
    else:
        # Use original function for <= max_labels_per_plot
        plot_target_histogram(
            df,
            label_col=label_col,
            target_col=target_col,
            bins=bins,
            img_path=img_path,
            **kwargs,
        )


@plot_func
def plot_target_histogram_per_subgroup(y, subgroup_mask=None, **kwargs):
    if np.sum(subgroup_mask) <= np.sum(~subgroup_mask):
        plt.hist(y[~subgroup_mask], alpha=0.5, label="Out of sg.", **kwargs)
        plt.hist(y[subgroup_mask], alpha=0.5, label="In sg.", **kwargs)

    else:
        plt.hist(y[subgroup_mask], alpha=0.5, label="In sg.", **kwargs)
        plt.hist(y[~subgroup_mask], alpha=0.5, label="Out of sg.", **kwargs)

    plt.legend(loc="upper right")


@plot_func
def plot_shaps(shaps, x_label="Time", y_label="Value"):
    """
    Plots multiple shapelets on separate subplots.

    Parameters:
    - shaps: List of shapelets (each shapelet is a 1D array).
    - title: Optional title for the entire figure.
    - x_label: Label for the x-axis (shared across all subplots).
    - y_label: Label for the y-axis (shared across all subplots).
    """
    # Plot setup
    k = len(shaps)
    axs_multiplier = 1
    width = 2 * axs_multiplier * 6.4
    height = k * axs_multiplier * 4.8

    # Create subplots
    fig, axs = plt.subplots(
        k,
        1,
        sharex=True,  # Share the x-axis among all subplots
        figsize=(width, height),
        gridspec_kw={"hspace": 0.1},  # Adjust space between plots
    )

    # If only one shapelet, axs won't be an array, so we make it one for consistency
    if k == 1:
        axs = [axs]

    # Plot each shapelet
    for i, shap in enumerate(shaps):
        shap_x = np.arange(0, len(shap))
        axs[i].plot(shap_x, shap)
        axs[i].set_ylabel(f"Shapelet {i+1}")  # Label each subplot with shapelet index

    # Set common labels
    axs[-1].set_xlabel(x_label)  # Set x-axis label only for the last subplot
    for ax in axs:
        ax.set_ylabel(y_label)

    plt.tight_layout()


@plot_func
def plot_best_matching_shaps(
    X, distances, subgroup, individual, undiff_shapelets: bool
):
    # Filter datasets based on subgroup mask
    X = X.copy()
    distances = distances.copy()

    distances = distances.iloc[subgroup]
    if isinstance(X, pd.DataFrame):
        X = X.iloc[subgroup]
    else:
        X = X[subgroup]

    # Create ordering index based on sum of distances
    d_cols = distances.filter(like="D_")
    distances["D_sum"] = d_cols.sum(axis=1)
    sort_indices = distances["D_sum"].argsort()
    k = 5

    # Plot setup
    axs_multiplier = 1
    width = 2 * axs_multiplier * 6.4
    height = (k // 5 + 1) * axs_multiplier * 4.8
    f, axs = plt.subplots(
        k,
        1,
        sharex=True,
        figsize=(width, height),
        gridspec_kw={
            "hspace": 0.1,
        },
    )

    for i, idx in enumerate(sort_indices[0:k]):
        # Plot timeseries
        if i >= len(X):
            break

        timeseries = X[idx]
        axs[i].plot(timeseries, alpha=0.4)

        # Plot shapelets
        for j, shap in enumerate(individual):
            position = distances.loc[distances.index[idx], f"L_{j}"]

            if undiff_shapelets:
                offset = timeseries[int(position)]
                shap_to_plot = undifferentiate_series(shap, offset=offset)
            else:
                offset = timeseries[int(position)] - shap[0]
                shap_to_plot = shap + offset

            shap_x = np.arange(position, position + len(shap_to_plot))
            axs[i].plot(shap_x, shap_to_plot, alpha=0.8)

    plt.xticks(np.arange(0, len(timeseries) + 1, 30.0))
    plt.tight_layout()


@plot_func
def plot_subgroup_alignment_comparison(
    X, distances, subgroup, individual, use_mean=False, undiff_shapelets=True
):
    """
    For a given subgroup and individual (list of shapelets),
    plot 3 representative instances from inside the subgroup
    (min, median, max total distance) and 3 from outside.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The original time series dataset.

    distances : pd.DataFrame
        Distance DataFrame with columns for distances (D_*) and positions (L_*).

    subgroup : list or np.ndarray
        Indices of instances inside the subgroup.

    individual : list
        List of shapelets (as arrays).

    use_mean : bool
        Whether to use the mean of the distances or the sum of the distances to
        retrieve min, median, max instances

    undiff_shapelets : bool
        Whether to undifferentiate the shapelets before plotting.
    """
    _X = X.copy()
    distances = distances.copy()

    d_cols = [f"D_{j}" for j in range(len(individual))]
    d_sum = distances[d_cols].sum(axis=1)
    distances["D_sum"] = d_sum
    distances["D_mean"] = d_sum / len(d_cols)

    inside_mask = np.zeros(len(distances), dtype=bool)
    inside_mask[subgroup] = True

    def get_min_median_max_idxs(distances_subset):
        sort_key = "D_mean" if use_mean else "D_sum"
        sorted_idx = distances_subset.sort_values(sort_key, ascending=True).index
        min_idx = sorted_idx[0]
        median_idx = sorted_idx[len(sorted_idx) // 2]
        max_idx = sorted_idx[-1]
        return [min_idx, median_idx, max_idx]

    inside_idxs = get_min_median_max_idxs(distances[inside_mask])
    outside_idxs = get_min_median_max_idxs(distances[~inside_mask])

    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3)

    inside_data = []
    outside_data = []

    for col, idx_group in enumerate([inside_idxs, outside_idxs]):
        group_label = "In subgroup" if col == 0 else "Outside of sg."
        data = inside_data if col == 0 else outside_data

        for row, idx in enumerate(idx_group):
            ax = axs[row, col]
            timeseries = _X.iloc[idx] if isinstance(_X, pd.DataFrame) else _X[idx]
            ax.plot(timeseries, alpha=0.4, label="Series")

            positions_aux = []
            for j, shap in enumerate(individual):
                position = int(distances.loc[idx, f"L_{j}"])
                positions_aux.append(position)

                if undiff_shapelets:
                    offset = timeseries[position]
                    shap_to_plot = undifferentiate_series(shap, offset=offset)
                else:
                    offset = timeseries[position] - shap[0]
                    shap_to_plot = shap + offset

                shap_x = np.arange(position, position + len(shap_to_plot))
                ax.plot(shap_x, shap_to_plot, alpha=0.8, label=f"Shapelet {j}")

            instance_record = {
                "index": idx,
                "rank": ["min", "median", "max"][row],
                "distances": distances.loc[idx, d_cols].values.tolist(),
                "distances_sum": distances.loc[idx, "D_sum"],
                "distances_mean": distances.loc[idx, "D_mean"],
                "positions": positions_aux,
            }
            data.append(instance_record)

            if col == 0:
                ax.set_ylabel(["Min", "Median", "Max"][row])
            if row == 0:
                ax.set_title(group_label)

    plt.suptitle(
        "Subgroup vs Outside: Min / Median / Max Alignment to Shapelets", y=1.02
    )
    plt.tight_layout()

    return {
        "inside": inside_data,
        "outside": outside_data,
    }


@plot_func
def plot_coverage_heatmap(top_k, cmap="YlGnBu"):
    # Extract the boolean mask arrays (subgroups) from each object in top_k
    coverage_matrix = np.vstack([obj.subgroup for obj in top_k])

    plt.figure(figsize=(15, 8))
    sns.heatmap(coverage_matrix.astype(int), annot=False, cmap=cmap, cbar=True)

    plt.xlabel("Instance Index")
    plt.ylabel("Top-k Individuals")
    plt.title("Coverage Matrix for Instances by Top-k Individuals")


@plot_func
def plot_jaccard_heatmap(jaccard_df, cmap="Blues"):

    mask = np.tril(np.ones(jaccard_df.shape), k=0).astype(bool)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        jaccard_df,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"shrink": 0.75},
        linewidths=0.5,
        annot_kws={"size": 16},  # cell text size
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.title("Upper Triangular Jaccard Similarity Heatmap", fontsize=16)
    plt.tight_layout()
