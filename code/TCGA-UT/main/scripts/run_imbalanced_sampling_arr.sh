#!/bin/bash

# run script with apptainer
for i in {0..13}; do
    param=$(printf "%.1f" "$(echo "$i / 10" | bc -l)")
    echo "Starting with parameter $param"
    output_path=#TODO
    sbatch run_imbalanced_sampling.sh $output_path $param
done

