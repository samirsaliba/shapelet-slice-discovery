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
            d, l = sliding_window_dist(X, shap_torch, shap_len)

            if cache is not None:
                cache.set(shap.id, (d.cpu(), l.cpu()))

        D[:, shap_ix] = d
        L[:, shap_ix] = l

    return D.cpu().numpy(), L.cpu().numpy()


def sliding_window_dist(X, shap, shap_len):
    """
    Calculate the minimum distance between a shapelet and each time series in X using sliding windows.

    Parameters:
    X (torch.Tensor): Time series data (2D tensor where rows are instances and columns are time steps)
    shap (torch.Tensor): Shapelet tensor
    shap_len (int): Length of the shapelet

    Returns:
    torch.Tensor, torch.Tensor: Minimum distances and positions for each time series instance
    """
    # Unfold X to get all sliding windows of the shapelet's length
    windows = X.unfold(1, shap_len, step=1)  # Shape: (N, num_windows, shap_len)

    # Calculate the Euclidean distance between each sliding window and the shapelet
    distances = torch.norm(windows - shap, dim=2)  # Broadcasting to calculate distances

    # Find the minimum distance and its index for each row in X
    min_distances, min_positions = distances.min(dim=1)

    return min_distances, min_positions
