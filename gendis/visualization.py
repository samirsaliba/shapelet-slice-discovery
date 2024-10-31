import matplotlib.pyplot as plt
import numpy as np
from .processing import undifferentiate_series


def plot_error_distributions(df, img_path):
    for lb in df["label"].value_counts().index:
        plt.hist(df.loc[df["label"] == lb, "error"], alpha=0.5, label=lb)
    plt.legend(loc="upper right")
    if img_path is not None:
        plt.savefig(img_path)
    else:
        plt.show()


def plot_shaps(shaps, img_path=None, x_label="Time", y_label="Value"):
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
    if img_path is not None:
        plt.savefig(img_path)
    else:
        plt.show()


def plot_best_matching_shaps(X, distances, subgroup, individual, img_path=None):
    # Filter datasets based on subgroup mask
    X = X.copy()
    distances = distances.copy()

    distances = distances[subgroup]
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

    if img_path is not None:
        plt.savefig(img_path)
    else:
        plt.show()
