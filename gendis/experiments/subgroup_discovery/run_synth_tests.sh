#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh. Exiting."; exit 1; }
conda activate gendis_fresh || { echo "Failed to activate conda environment 'gendis_fresh'. Exiting."; exit 1; }

DATASETS=(
  "multipattern_noise_1"
  "combo_patterns_noise_1"
)
DATASET_DIR="data/multipattern_synth"

for DATASET in "${DATASETS[@]}"; do
  
    DATASET_PATH="${DATASET_DIR}/${DATASET}.csv"
    LOG_FILE="${DATASET}.log"
    
    echo "Running for ${DATASET}..."

    # Run the Python script and capture output
    if python3 gendis_subgroup_discovery.py "${DATASET_PATH}" 2>&1 | tee "${LOG_FILE}"; then
      echo "Successfully completed ${DATASET}"
    else
      echo "ERROR: Failed to process ${DATASET}"
      break  # Stop if any failure
    fi

    echo "---------------------------------"
done

echo "All jobs completed."
