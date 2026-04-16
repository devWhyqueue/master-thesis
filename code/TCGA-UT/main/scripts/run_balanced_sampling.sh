#!/bin/bash
#SBATCH --job-name=sample_balanced
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/sampling/sample_balanced%j.out
#SBATCH -D /PATH/TO/WORKING_DIR #TODO

# run script with apptainer
apptainer run -B /home/space:/home/space:rw /#TODO/PATH/TO/environment.sif python3 sample_TCGA_UT_balanced.py --dataset-path=#TODO --file-save-path=#TODO --n-slides-per-class=100 --n-patches-per-slide=30

