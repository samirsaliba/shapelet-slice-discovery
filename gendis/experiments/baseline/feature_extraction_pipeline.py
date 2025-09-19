import multiprocessing
import pandas as pd
from tsfresh import extract_features, select_features

from helpers.util import DROP_COLUMNS

N_JOBS = multiprocessing.cpu_count() - 3


def main(base_name):

    input_file_path = f"./data/{base_name}.csv"

    df = pd.read_csv(input_file_path)

    series = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore",
    ).reset_index()

    # Melt the dataframe to long format
    df_long = series.melt(id_vars="index", var_name="time", value_name="value")

    df_long["time"] = df_long["time"].astype(int)
    df_long = df_long.sort_values(by=["index", "time"]).reset_index(drop=True)

    extracted_features = extract_features(
        df_long, column_id="index", column_sort="time", n_jobs=N_JOBS
    )

    extracted_features = extracted_features.dropna(axis=1)

    selected_features = select_features(
        X=extracted_features, y=df["error"], ml_task="regression", n_jobs=N_JOBS
    )

    selected_features["error"] = df["error"]
    selected_features.to_csv(f"./processed/{base_name}_tsfresh.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Feature Extraction (tsfresh) Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset base name (without extension). E.g., NonInvasiveFetalECGThorax2_tsforest_error",
    )
    args = parser.parse_args()

    main(args.dataset)
