#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh. Exiting."; exit 1; }
conda activate gendis_fresh || { echo "Failed to activate conda environment 'gendis_fresh'. Exiting."; exit 1; }

# List of datasets (prefixes without the model suffix)
DATASET_PREFIXES=(
# "DistalPhalanxOutlineAgeGroup_"
  "NonInvasiveFetalECGThorax2_"
  "Strawberry_"
)

# List of classifiers (model names as used in the filenames)
MODELS=("inception" "tsforest")

# Path to the dataset directory
DATASET_DIR="data/ucr_uea_datasets_v2"

# Double loop: dataset prefix × model
for PREFIX in "${DATASET_PREFIXES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
  
    # Build dataset file name and path
    DATASET_FILE="${PREFIX}${MODEL}_errors.csv"
    DATASET_PATH="${DATASET_DIR}/${DATASET_FILE}"
    
    # Log file (one per dataset+model run)
    LOG_FILE="${MODEL}-${PREFIX%_}.log"
    
    echo "Running for ${DATASET_FILE}..."

    # Run the Python script and capture output
    if python3 gendis_subgroup_discovery.py "${DATASET_PATH}" 2>&1 | tee "${LOG_FILE}"; then
      echo "Successfully completed ${DATASET_FILE}"
    else
      echo "ERROR: Failed to process ${DATASET_FILE}"
      break  # Stop if any failure
    fi

    echo "---------------------------------"

  done
done

echo "All jobs completed."
