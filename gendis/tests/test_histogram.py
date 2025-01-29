import pandas as pd
import numpy as np
from gendis.visualization import (
    plot_target_histogram,
    plot_target_histograms_in_batches,
)


img_path = "./test_hist.png"
size = 500
bins = 20

error = np.random.rand(size)

# Example DataFrame
df = pd.DataFrame(
    {
        "label": np.random.randint(0, 2, size=size),
        "error": error,
    }
)

plot_target_histogram(
    df, label_col="label", target_col="error", bins=bins, img_path=img_path
)

# Batches
df = pd.DataFrame(
    {
        "label": np.random.choice([f"Class {i}" for i in range(1, 11)], size=size),
        "error": error,
    }
)

# Plot with batching
plot_target_histograms_in_batches(
    df,
    label_col="label",
    target_col="error",
    bins=10,
    max_labels_per_plot=4,
    img_path="histogram",
)
