#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import json
import numpy as np
from os.path import join
import pandas as pd
import sys

from sktime.datasets import load_UCR_UEA_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from sktime.classification.deep_learning.mlp import MLPClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.shapelet_based import ShapeletLearningClassifierTslearn

import logging
import pathlib
from memory import limit_memory

RANDOM_STATE = 0
N_JOBS = 13
TEST_SIZE = 0.7

folder = "./results"
datasets = [
    "LargeKitchenAppliances",
    "DistalPhalanxOutlineAgeGroup",
    "Strawberry",
    "ShapesAll",
    "FaceAll",
    "InsectWingbeatSound",
    "UWaveGestureLibraryAll",
]


pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{folder}/script_{timestamp}.log"

log_format = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(
    filename=log_file,
    filemode="a",
    format=log_format,
    level=logging.INFO,
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(console_handler)
logging.info("Script started")
logging.info(f"Datasets: \n {',\n'.join(datasets)}")


classifiers = {
    "cnn": (
        CNNClassifier,
        {"n_epochs": 2000, "n_conv_layers": 1, "random_state": RANDOM_STATE},
    ),
    # "knn": (
    #     KNeighborsTimeSeriesClassifier,
    #     {"distance": "euclidean", "algorithm": "ball_tree", "n_jobs": N_JOBS},
    # ),
    "rocket": (
        RocketClassifier,
        {"num_kernels": 1000, "n_jobs": N_JOBS, "random_state": RANDOM_STATE},
    ),
    # "shapelet": (
    #     ShapeletLearningClassifierTslearn,
    #     {"max_iter": 1000, "random_state": RANDOM_STATE},
    # ),
    "mlp": (
        MLPClassifier,
        {"n_epochs": 2000, "random_state": RANDOM_STATE},
    ),
}


def get_errors(y_test, y_pred_proba):
    adjusted_y_test = y_test.astype(int) - 1
    actual_class_probs = y_pred_proba[np.arange(len(y_test)), adjusted_y_test]
    errors = 1 - actual_class_probs
    return errors


# @limit_memory(ratio=0.75)
def main():
    for i, dataset in enumerate(datasets):
        logging.info(f"Loop started for dataset {dataset} - [{i+1}/{len(datasets)}]")

        X, y = load_UCR_UEA_dataset(name=dataset, return_type="numpy2d")

        df = pd.DataFrame(data=X)
        df["label"] = y
        df.to_csv(join(folder, f"{dataset}_raw.csv"), index=False)

        logging.info(f"len(df) = {len(df)}")

        X = df.drop(columns="label")
        y = df["label"]

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        train_index, test_index = next(sss.split(X, y))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        info = {
            "random_state": RANDOM_STATE,
            "test_size_ratio": TEST_SIZE,
            "dataset": dataset,
            "labels": df["label"].value_counts().to_dict(),
        }

        j = 0
        for model_name, (model, params) in classifiers.items():
            logging.info(
                f"DS [{i+1}/{len(datasets)}] - model {model_name} - [{j+1}/{len(classifiers)}] - Training started"
            )

            clf = model(**params)
            clf.fit(X_train.values, y_train.values)
            logging.info(
                f"DS [{i+1}/{len(datasets)}] - model {model_name} - [{j+1}/{len(classifiers)}] - Training finished"
            )

            y_pred = clf.predict(X_test.values)
            model_accuracy = accuracy_score(y_test.values, y_pred)
            info[model_name] = {
                "params": params,
                "accuracy": model_accuracy,
                "f1": f1_score(y_test.values, y_pred, average="weighted"),
            }

            logging.info(
                f"DS [{i+1}/{len(datasets)}] - model {model_name} - [{j+1}/{len(classifiers)}] - Accuracy: {model_accuracy}"
            )

            y_pred_proba = clf.predict_proba(X_test.values)
            errors = get_errors(y_test, y_pred_proba)

            enriched_df = pd.DataFrame(data=X_test)
            enriched_df["label"] = y_test
            enriched_df["predicted"] = y_pred
            enriched_df["error"] = errors
            enriched_df.to_csv(
                join(folder, f"{dataset}_{model_name}_errors.csv"), index=False
            )
            j += 1

        with open(join(folder, f"{dataset}_models_info.json"), "w") as f:
            json.dump(info, f)


if __name__ == "__main__":
    main()
