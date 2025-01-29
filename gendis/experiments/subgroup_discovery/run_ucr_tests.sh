#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh. Exiting."; exit 1; }
conda activate gendis_fresh || { echo "Failed to activate conda environment 'gendis_fresh'. Exiting."; exit 1; }


# Define the dataset prefix and models
DATASET_PREFIX="Strawberry_"
MODELS=("inception" "tsforest")

# Path to the dataset directory
DATASET_DIR="data/ucr_uea_datasets_v2"

# Iterate through the models
for MODEL in "${MODELS[@]}"; do
    # Construct the dataset file name
    DATASET_FILE="${DATASET_PREFIX}${MODEL}_errors.csv"
    
    # Full path to the dataset
    DATASET_PATH="${DATASET_DIR}/${DATASET_FILE}"
    
    # Construct the log file name
    LOG_FILE="${MODEL}-${DATASET_PREFIX%_}.log"
    
    # Run the script sequentially and check for errors
    echo "Running for ${DATASET_FILE}..."
    if python3 gendis_subgroup_discovery.py "${DATASET_PATH}" > "${LOG_FILE}" 2>&1; then
        echo "Successfully completed ${DATASET_FILE}"
    else
        echo "ERROR: Failed to process ${DATASET_FILE}"
        break  # Stop execution if an error occurs
    fi
done

echo "All jobs completed."

