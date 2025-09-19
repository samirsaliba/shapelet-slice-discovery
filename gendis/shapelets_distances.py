import torch


def calculate_shapelet_dist_matrix(X, shapelets, cache=None, device="cuda"):
    """
    Calculate the distance matrix for a set of shapelets using PyTorch for GPU acceleration.

    Parameters:
    X (torch.Tensor): Time series data (2D tensor where rows are instances and columns are time steps)
    shapelets (list of Shapelet): List of shapelets to calculate distances against
    cache (dict): Optional cache to store previously computed distances
    device (str): 'cuda' for GPU or 'cpu' for CPU calculations

    Returns:
    (D (np.ndarray): Distances matrix, L (np.ndarray): Locations matrix)
    """
    # Move data to torch tensors
    torch.multiprocessing.set_sharing_strategy("file_system")

    D = torch.full((len(X), len(shapelets)), -1.0, device=device)
    L = torch.full((len(X), len(shapelets)), -1.0, device=device)

    for shap_ix, shap in enumerate(shapelets):
        # Move the shapelet to a tensor
        shap_torch = torch.tensor(shap, dtype=torch.float32, device=device)
        shap_len = len(shap)

        # Check cache
        cache_val = None
        if cache is not None:
            cache_val = cache.get(shap.id)

        if cache_val is not None:
            d, l = cache_val
        else:
            # Calculate distances and positions for each row in X
            d, l = sliding_window_dist(
                X, shap_torch, shap_len, norm="l2", offset_align=True
            )

            if cache is not None:
                cache.set(shap.id, (d.cpu(), l.cpu()))

        D[:, shap_ix] = d
        L[:, shap_ix] = l

    return D.cpu().numpy(), L.cpu().numpy()


def sliding_window_dist(X, shap, shap_len, norm="l1", offset_align=True):
    """
    Compute sliding window distances between time series and shapelet.

    Parameters:
    X (torch.Tensor): (N, T) time series data
    shap (torch.Tensor): (L,) shapelet
    shap_len (int): Length of the shapelet
    norm (str): "l1" or "l2"
    offset_align (bool): If True, offset shapelet to align with window's first point

    Returns:
    torch.Tensor: Min distances, torch.Tensor: matching start positions
    """
    windows = X.unfold(1, shap_len, step=1)  # (N, num_windows, L)

    if offset_align:
        # Compute offset to align shapelet start to window start
        delta = windows[:, :, 0] - shap[0]  # (N, num_windows)
        delta = delta.unsqueeze(-1)  # (N, num_windows, 1)
        aligned_shap = (
            shap.unsqueeze(0).unsqueeze(0) + delta
        )  # (1, 1, L) + (N, num_windows, 1)
        diffs = windows - aligned_shap
    else:
        diffs = windows - shap  # broadcasting works without offset

    if norm == "l1":
        distances = torch.sum(torch.abs(diffs), dim=2)
    elif norm == "l2":
        distances = torch.norm(diffs, dim=2)
    else:
        raise ValueError(f"Unsupported norm '{norm}'. Use 'l1' or 'l2'.")

    return distances.min(dim=1)
