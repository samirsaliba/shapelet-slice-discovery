#!/bin/bash

# Define the dataset file path
DATASET="../data/multipattern_synth/multipattern_noise_1.csv"

# Define the parameter name and its values
PARAM_NAME="K"
# PARAM_VALUES=(0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0 1.5 2.0)
PARAM_VALUES=(5 7 10 12 15 17 20)

# Loop through each parameter value
for PARAM_VAL in "${PARAM_VALUES[@]}"; do
    echo "Running script with dataset: $DATASET and $PARAM_NAME: $PARAM_VAL"
    
    # Run the Python script with the parameter
    python3 subgroup_param_tests.py "$DATASET" --"$PARAM_NAME" "$PARAM_VAL"
    
    # Check the exit status of the Python script
    if [ $? -ne 0 ]; then
        echo "Error encountered while processing with $PARAM_NAME=$PARAM_VAL. Halting execution."
        exit 1
    fi
done

echo "All tests executed successfully."
