from .individual import Shapelet


def random_shapelet(X, n_shapelets, min_len, max_len, np_random):
    """Extract random subseries from the training set"""
    shapelets = []
    m = len(X[0])
    for _ in range(n_shapelets):
        row = np_random.integers(X.shape[0])
        length = np_random.integers(min_len, max_len)
        col = np_random.integers(0, m - length)
        shap_series = X[row][col : col + length]
        shapelet = Shapelet(shap_series, index=row, start=col)
        shapelets.append(shapelet)

    return shapelets
