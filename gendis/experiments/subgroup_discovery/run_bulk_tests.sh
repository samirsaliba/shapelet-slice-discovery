#!/bin/bash

# Define the base file path and noise levels
BASE_PATH="../data/multipattern_synth/multipattern_noise_"
NOISE_LEVELS=(0 0.25 0.5 1 1.5 2 3 4)

# Loop through each noise level
for NOISE in "${NOISE_LEVELS[@]}"; do
    FILE="${BASE_PATH}${NOISE}.csv"
    
    echo "Running script with input file: $FILE"
    
    # Run the Python script
    python3 subgroup_multiple_tests.py "$FILE"
    
    # Check the exit status of the Python script
    if [ $? -ne 0 ]; then
        echo "Error encountered while processing $FILE. Halting execution."
        exit 1
    fi
done

echo "All scripts executed successfully."
