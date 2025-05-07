import argparse
import json
import logging
import numpy as np
import pathlib
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Process some input file.")
    parser.add_argument("input_file_path", type=str, help="The path to the input file")
    return parser.parse_args()


def setup_logging(path, timestamp):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    log_file = f"{path}/script_{timestamp}.log"

    # Define the log format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Set up logging to file
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format=log_format,
        level=logging.INFO,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, cls=NpEncoder)
