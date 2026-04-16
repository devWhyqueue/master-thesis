#!/bin/bash
#SBATCH --job-name=train_knn
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/training/train_knn%j.out
#SBATCH -D /PATH/TO/WORKING_DIR #TODO

# run script with apptainer
k=$1
apptainer run -B /home/space:/home/space:rw ./environment.sif python3 train.py \
        --dataset-structure-path=#TODO \
	--validation-dataset-structure-path=#TODO \
        --feature-path="/home/space/datasets/patho_ds/tcga-ut-balanced/virchow_v2/" \
        --preload-features \
	--results-save-path="#TODO/results/results_knn/k=$k/" \
	--device="cpu" \
	--model="knn" \
	--k="$k" \
	--visualize \
	--class-names-path="#TODO/class_names.json"
