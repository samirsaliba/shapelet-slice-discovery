import pandas as pd
from os.path import join
from pathlib import Path

from gendis.visualization import (
    plot_jaccard_heatmap,
)


def main(filepath):
    path = Path(filepath)
    df = pd.read_csv(path, index_col=0)

    # offsetting the jaccard plot (subgroups 1...k instead of 0...k-1)
    new_labels = [i + 1 for i in df.index]
    df.index = new_labels
    df.columns = new_labels

    plot_jaccard_heatmap(df, img_path=join(path.parent, "jaccard_heatmap.pdf"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate heatmap plot csv dataframe")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="CSV jaccard dataframe file path eg ./data/gendis/multipattern[...]/jaccard_matrix.csv",
    )
    args = parser.parse_args()

    main(args.path)
