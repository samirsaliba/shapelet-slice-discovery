import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join

from pathlib import Path


def main(filepath):
    path = Path(filepath)

    df = pd.read_csv(path, index_col=0)

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
    plt.savefig(join(path.parent, "jaccard_heatmap.png"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate heatmap plot csv dataframe")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="CSV dataframe file path eg ./data/gendis/multipattern[...]/jaccard_matrix.csv",
    )
    args = parser.parse_args()

    main(args.path)
