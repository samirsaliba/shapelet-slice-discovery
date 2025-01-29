import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
from .processing import undifferentiate_series


def plot_func(func):
    def wrapper(*args, **kwargs):
        img_path = kwargs.pop("img_path", None)
        func(*args, **kwargs)

        if img_path is not None:
            plt.savefig(img_path)
        else:
            plt.show()
        plt.clf()

    return wrapper


@plot_func
def plot_target_histogram(df, label_col="label", target_col="error", bins=10, **kwargs):
    # Group data by label
    labels = df[label_col].unique()  # Unique labels
    data = [df.loc[df[label_col] == lb, target_col] for lb in labels]

    # Plot histogram with histtype="bar"
    plt.hist(
        data, bins=bins, histtype="bar", label=labels, alpha=0.7, range=(0, 1), **kwargs
    )

    plt.legend(loc="upper right")
    plt.xlabel(target_col)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {target_col} by {label_col}")
    plt.show()


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
                f"{img_path}_batch_{batch_idx + 1}.png" if img_path else None
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
def plot_best_matching_shaps(X, distances, subgroup, individual):
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
            offset = timeseries[int(position)]
            shap_undiffed = undifferentiate_series(shap, offset=offset)
            shap_x = np.arange(position, position + len(shap_undiffed))
            axs[i].plot(shap_x, shap_undiffed, alpha=0.8)

    plt.xticks(np.arange(0, len(timeseries) + 1, 30.0))
    plt.tight_layout()


@plot_func
def plot_coverage_heatmap(top_k, cmap="YlGnBu"):
    # Extract the boolean mask arrays (subgroups) from each object in top_k
    coverage_matrix = np.vstack([obj.subgroup for obj in top_k])

    plt.figure(figsize=(15, 8))
    sns.heatmap(coverage_matrix.astype(int), annot=False, cmap=cmap, cbar=True)

    plt.xlabel("Instance Index")
    plt.ylabel("Top-k Individuals")
    plt.title("Coverage Matrix for Instances by Top-k Individuals")
