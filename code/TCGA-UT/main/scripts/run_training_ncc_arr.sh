#!/bin/bash

# run script with apptainer
for i in {0..13}; do
    		param=$(printf "%.1f" "$(echo "$i / 10" | bc -l)")
    		echo "Starting with parameter $param"
    		path="#TODO/TCGA-UT_imbalanced_parameter=""$param""_dataset_size=500_seed=0/imbalanced_dataset.csv"
    		out_path="#TODO/results_ncc/param=$param/"
    		sbatch run_training_ncc.sh $path $out_path
done

