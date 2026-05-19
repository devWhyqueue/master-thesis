#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/sampling/sample_imbalanced%j.out
#SBATCH -D /PATH/TO/WORKING_DIR #TODO

# run script with apptainer
out_path=$1
param=$2
apptainer run -B /home/space:/home/space:rw ./environment.sif python3 sample_TCGA_UT_imbalanced.py\
       	--balanced-dataset-path=#TODO \
       	--file-save-path="$out_path" \
	--parameter=$param \
       	--dataset-size=500 \
       	--sample-balanced-validation \
       	--n-slides-per-class=10 \
       	--visualize \
       	--overflow-strategy="redistribute" \
	--n-regions-per-slide=3 \
	--n-patches-per-region=10 \
	--store-class-names
