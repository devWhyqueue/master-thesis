#!/bin/bash
#SBATCH --job-name=train_ncc
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/training/train_ncc%j.out
#SBATCH -D /PATH/TO/WORKING_DIR #TODO

# run script with apptainer
ds_path=$1
out_path=$2
apptainer run -B /home/space:/home/space:rw ./environment.sif python3 train.py \
        --dataset-structure-path="$ds_path" \
	--validation-dataset-structure-path=#TODO \
        --feature-path=#TODO \
        --preload-features \
	--results-save-path="$out_path" \
	--device="cpu" \
	--model="ncc" \
	--visualize \
	--class-names-path="#TODO/class_names.json"
