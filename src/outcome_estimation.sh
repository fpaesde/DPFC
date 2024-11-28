#!/bin/bash

# Array of names to iterate through
names=("sim" "sim_complex" "sim_non_linear" "sim_confounding")

# Loop through each name and run without --model "g"
for name in "${names[@]}"; do
    echo "Running: python train_test_script.py --name \"$name\""
    python evaluate_outcome_estimation.py --name "$name"
done
