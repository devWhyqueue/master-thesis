#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/training/train%j.out
#SBATCH -D /PATH/TO/WORKING_DIR #TODO

# run script with apptainer
ds_path=$1
out_path=$2
seed=$3
apptainer run -B /home/space:/home/space:rw ./environment.sif python3 train.py \
        --dataset-structure-path="$ds_path" \
	--validation-dataset-structure-path=#TODO \
        --feature-path="/home/space/datasets/patho_ds/tcga-ut-balanced/virchow_v2/" \
        --preload-features \
	--results-save-path="$out_path" \
	--device="cpu" \
	--learning-rate=0.001 \
	--n-epochs=50 \
	--loss="cross_entropy" \
	--alpha="uniform" \
	--batch-balancing \
	--seed=$seed \
	--visualize \
	--class-names-path="#TODO/class_names.json"

