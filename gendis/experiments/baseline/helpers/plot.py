from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


@auto_savefig
def plot_shapelets_on_series(X, shapelets, img_path=None):
    """
    Plot the shapelets overlaid on the series they were extracted from.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_timestamps, n_channels)
        Original input dataset.

    shapelets : list
        List of shapelet tuples as returned in RandomShapeletTransform.shapelets.

    """

    plt.figure(figsize=(10, 3 * len(shapelets)))
    for i, (shap_id, shapelet_tuple) in enumerate(shapelets.items()):
        (gain, length, start, dim, ts_idx, cls, shp_znorm) = shapelet_tuple
        end = start + length

        series = X.iloc[ts_idx].to_numpy()
        shapelet = X.iloc[ts_idx, start:end]

        plt.subplot(len(shapelets), 1, i + 1)
        plt.plot(series, label=f"Original series #{ts_idx}, dim {dim}")
        plt.plot(
            range(start, end), shapelet, color="red", label="Shapelet", linewidth=2
        )
        plt.title(
            f"Shapelet #{shap_id} from class {cls} (IG={gain:.3f}) at pos {start}"
        )
        plt.legend()

    plt.tight_layout()


@auto_savefig
def plot_shapelet_distance_scatter_2d(
    distances, subgroup_indexes, shapelet_ids, img_path=None
):
    """
    Plot a 2D scatter of distances to two shapelets, colored by subgroup membership.

    Parameters
    ----------
    distances : np.ndarray of shape (n_samples, n_shapelets)
        Distance matrix from shapelets to all series.

    subgroup_indexes : iterable of int
        Indices of instances that belong to the subgroup.

    shapelet_ids : tuple of (int, int)
        The two shapelet indices used for the axes.
    """
    x_idx, y_idx = shapelet_ids
    x = distances[:, x_idx]
    y = distances[:, y_idx]

    # Convert to boolean mask
    mask = np.zeros(distances.shape[0], dtype=bool)
    mask[list(subgroup_indexes)] = True

    plt.figure(figsize=(7, 6))
    plt.scatter(
        x[~mask], y[~mask], s=2, alpha=0.4, label="Outside subgroup", color="gray"
    )
    plt.scatter(x[mask], y[mask], s=2, alpha=0.4, label="Inside subgroup", color="red")

    plt.xlabel(f"Distance to Shapelet {x_idx}")
    plt.ylabel(f"Distance to Shapelet {y_idx}")
    plt.title(f"Subgroup coverage in 2D shapelet distance space")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()


@auto_savefig
def plot_shapelet_distance_density_bubbleplot(
    distances, subgroup_indexes, shapelet_ids, bins=50, img_path=None
):
    x_idx, y_idx = shapelet_ids
    x = distances[:, x_idx]
    y = distances[:, y_idx]

    mask = np.zeros(distances.shape[0], dtype=bool)
    mask[list(subgroup_indexes)] = True

    def get_bubble_data(x_vals, y_vals):
        counts, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=bins)
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        Xc, Yc = np.meshgrid(x_centers, y_centers)
        Xc = Xc.ravel()
        Yc = Yc.ravel()
        sizes = counts.T.ravel()
        nonzero = sizes > 0
        return Xc[nonzero], Yc[nonzero], sizes[nonzero]

    # Get bubble positions and sizes for each group
    x_out, y_out, s_out = get_bubble_data(x[~mask], y[~mask])
    x_in, y_in, s_in = get_bubble_data(x[mask], y[mask])

    plt.figure(figsize=(7, 6))
    plt.scatter(x_out, y_out, s=s_out * 3, alpha=0.2, color="gray", label="Outside")
    plt.scatter(x_in, y_in, s=s_in * 3, alpha=0.5, color="red", label="Inside")

    plt.xlabel(f"Distance to Shapelet {x_idx}")
    plt.ylabel(f"Distance to Shapelet {y_idx}")
    plt.title("Density-Based Bubble Plot in Shapelet Distance Space")
    plt.legend()
    plt.tight_layout()


@auto_savefig
def plot_decision_surface_2d(
    clf, distances, y_mask, shapelet_ids, resolution=300, img_path=None
):
    """
    Plot decision surface of a 2D classifier over shapelet distance space.

    Parameters
    ----------
    clf : trained sklearn classifier (e.g., DecisionTreeClassifier)

    distances : np.ndarray of shape (n_samples, n_shapelets)
        Shapelet distance matrix.

    y_mask : np.ndarray of bool
        Boolean mask of instances inside the subgroup (used for coloring points).

    shapelet_ids : tuple of (int, int)
        Indices of the two shapelets used as features.

    resolution : int
        Grid resolution for the contour plot.
    """
    x_idx, y_idx = shapelet_ids
    X = distances[:, [x_idx, y_idx]]
    y = y_mask.astype(int)

    # Create grid
    x_min, x_max = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
    y_min, y_max = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    # Predict over grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # Plot decision surface
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)

    # Plot actual points
    plt.scatter(
        X[~y_mask, 0],
        X[~y_mask, 1],
        s=10,
        c="gray",
        alpha=0.4,
        label="Outside subgroup",
    )
    plt.scatter(
        X[y_mask, 0], X[y_mask, 1], s=10, c="red", alpha=0.8, label="Inside subgroup"
    )

    plt.xlabel(f"Distance to Shapelet {x_idx}")
    plt.ylabel(f"Distance to Shapelet {y_idx}")
    plt.title("Decision surface in 2D shapelet distance space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def _find_best_match_position(series, shapelet):
    """
    Find the best (minimum distance) match position of a shapelet within a series.

    Parameters
    ----------
    series : np.ndarray
        1D time series.

    shapelet : np.ndarray
        1D shapelet.

    Returns
    -------
    int
        Start position of the best match.
    """
    min_dist = np.inf
    best_pos = 0
    slen = len(shapelet)
    for i in range(len(series) - slen + 1):
        window = series[i : i + slen]
        dist = np.linalg.norm(window - shapelet)
        if dist < min_dist:
            min_dist = dist
            best_pos = i
    return best_pos


@auto_savefig
def plot_subgroup_series_with_defining_shapelets(
    X, shapelets, subgroup_indices, distances, K=5, img_path=None
):
    """
    Plot K time series from a selected subgroup and overlay the shapelets that define the subgroup,
    prioritizing those with smallest total shapelet distances.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_timestamps)
        Wide-format univariate time series.

    shapelets : dict
        Dictionary of shapelet tuples, typically from shapelet_dict[class_label].
        Keys are shapelet IDs; values are shapelet tuples.

    subgroup_indices : list or set of int
        Indices of time series that belong to the subgroup.

    distances : np.ndarray of shape (n_samples, n_shapelets)
        Matrix of shapelet distances (same shape as X_transformed).

    K : int
        Number of top-matching subgroup members to plot.
    """
    subgroup_indices = sorted(list(subgroup_indices))
    if K > len(subgroup_indices):
        print(f"Only {len(subgroup_indices)} members in subgroup. Showing all.")
        K = len(subgroup_indices)

    # Compute sum of distances for each instance in the subgroup
    shapelet_ids = list(shapelets.keys())
    distance_sums = {
        idx: distances[idx, shapelet_ids].sum() for idx in subgroup_indices
    }

    # Sort indices by summed distance
    sorted_indices = sorted(distance_sums, key=distance_sums.get)
    selected_indices = sorted_indices[:K]

    # Match plotting style with other functions
    axs_multiplier = 1
    width = 2 * axs_multiplier * 6.4
    height = (K // 5 + 1) * axs_multiplier * 4.8

    fig, axs = plt.subplots(
        K,
        1,
        sharex=True,
        figsize=(width, height),
        gridspec_kw={"hspace": 0.1},
    )

    for i, ts_idx in enumerate(selected_indices):
        series = X.iloc[ts_idx].to_numpy()
        ax = axs[i] if K > 1 else axs

        ax.plot(series, alpha=0.4)

        for sid, shap_tuple in shapelets.items():
            gain, length, extraction_start, dim, extracted_shp_idx, cls, shp_z_norm = (
                shap_tuple
            )
            shapelet = X.iloc[
                extracted_shp_idx, extraction_start : extraction_start + length
            ].to_numpy()

            start_in_series = _find_best_match_position(
                series=series, shapelet=shapelet
            )
            end = start_in_series + length

            ax.plot(range(start_in_series, end), shapelet, linewidth=2, alpha=0.8)

        # ax.set_title(f"Series #{ts_idx} — Total distance: {distance_sums[ts_idx]:.4f}")

    plt.xticks(np.arange(0, series.shape[0] + 1, 30.0))
    plt.tight_layout()


@auto_savefig
def plot_heatmap_jaccard(df, img_path=None):
    mask = np.tril(np.ones(df.shape), k=0).astype(bool)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        df,
        mask=mask,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        square=True,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"shrink": 0.75},
        linewidths=0.5,
    )

    plt.title("Upper Triangular Jaccard Similarity Heatmap")
    plt.tight_layout()
