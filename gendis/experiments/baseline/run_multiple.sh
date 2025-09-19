#!/bin/bash

# List of datasets (without .csv extension)
datasets=(
  "DistalPhalanxOutlineAgeGroup_inception_errors"
  "DistalPhalanxOutlineAgeGroup_tsforest_errors"
  "NonInvasiveFetalECGThorax2_inception_errors"
  "NonInvasiveFetalECGThorax2_tsforest_errors"
  "Strawberry_inception_errors"
  "Strawberry_tsforest_errors"
)

# Loop over datasets
for dataset in "${datasets[@]}"; do
  echo "Processing dataset: $dataset"

  python feature_extraction_pipeline.py --dataset "$dataset"
  python fe_sd_rst_pipeline_synthetic.py --dataset "$dataset"

  echo "Finished processing $dataset"
done
