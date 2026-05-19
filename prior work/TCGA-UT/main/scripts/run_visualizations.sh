#!/bin/bash
#SBATCH --job-name=viz
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/viz/viz%j.out
#SBATCH -D /PATH/TO/WORKING_DIR #TODO

# run script with apptainer
apptainer run -B /home/space:/home/space:rw ./environment.sif python3 visualizations.py \
	--plot-types scatter_accuracies_of_two_parameters difference_confusion_matrix confusion_matrix \
	--results-paths \
	"#TODO/results_batch_balancing/" \
	"#TODO/results_batch_balancing/" \
	--parameters 1.0 0.0 \
	--parameter-name "param" \
	--visualization-save-path "#TODO/visualization/"
