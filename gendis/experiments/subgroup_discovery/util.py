import argparse
from datetime import datetime

import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Process some input file.')
    parser.add_argument('input_file_path', type=str, help='The path to the input file')
    return parser.parse_args()

def setup_logging(path):
    log_file = f'{path}/script_{timestamp}.log'
    
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG
    )

