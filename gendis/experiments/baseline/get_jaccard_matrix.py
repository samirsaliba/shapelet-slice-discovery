from helpers.subgroup_evaluation import (
    compute_jaccard_matrix,
    summarize_jaccard_matrix,
)
from os.path import join

import json
import numpy as np
import pandas as pd
from pathlib import Path


def main(filepath):
    path = Path(filepath)

    with open(path, "r") as file:
        data = json.load(file)

    sgs_index_sets = {
        idx: set(np.where(item["subgroup"])[0].tolist())
        for idx, item in enumerate(data)
    }

    jaccard_df = pd.DataFrame(
        compute_jaccard_matrix(sgs_index_sets),
        index=range(len(sgs_index_sets)),
        columns=range(len(sgs_index_sets)),
    )
    jaccard_df.to_csv(join(path.parent, "jaccard_matrix.csv"))

    jac_summary = summarize_jaccard_matrix(jaccard_df)
    with open(
        join(path.parent, "jaccard_matrix_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(jac_summary, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Jaccard Matrix From Top-K Subgroups Dict"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Top-K json file path ./data/gendis/multipattern[...]/topk.json",
    )
    args = parser.parse_args()

    main(args.path)
