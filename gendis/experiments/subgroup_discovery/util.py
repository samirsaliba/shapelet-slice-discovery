import argparse
from datetime import datetime
import json
import logging
import numpy as np
import pathlib

def parse_args():
    parser = argparse.ArgumentParser(description='Process some input file.')
    parser.add_argument('input_file_path', type=str, help='The path to the input file')
    return parser.parse_args()

def setup_logging(path, timestamp):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
    log_file = f'{path}/script_{timestamp}.log'    
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )



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
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NpEncoder)
