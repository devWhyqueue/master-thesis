#!/bin/bash

# run script with apptainer
for i in {0..13}; do
	for seed in {0..2}; do
    		param=$(printf "%.1f" "$(echo "$i / 10" | bc -l)")
    		echo "Starting with parameter $param and seed $seed"
    		path="#TODO/TCGA-UT_imbalanced_parameter=""$param""_dataset_size=500_seed=0/imbalanced_dataset.csv"
    		out_path="#TODO/param=$param/seed=$seed"
    		sbatch run_training.sh $path $out_path $seed
	done
done

