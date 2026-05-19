#!/bin/bash
#SBATCH --job-name=viz
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/viz/viz%j.out
#SBATCH -D /PATH/TO/WORKING_DIR #TODO

# run script with apptainer
apptainer run -B /home/space:/home/space:rw ./environment.sif python3 visualizations.py \
	--plot-types point_plot_compare_methods \
	--results-paths \
     "#TODO/results_cross_entropy_inverse_class_frequency" \
     "#TODO/results_batch_balancing/" \
     "#TODO/results_focal_loss_inverse_class_frequency/" \
     "#TODO/results_focal_loss_uniform/" \
     "#TODO/results_original_class_size_order/" \
     "#TODO/results_original_class_size_order/" \
	--parameters 1.0 1.0 1.0 1.0 1.0 0.0\
	--parameter-name "param" \
	--methods \
	"Weighted Cross Entropy" \
	"Batch Balancing" \
	"Weighted Focal Loss" \
	"Unweighted Focal Loss" \
	"Vanilla" \
	"Balanced" \
	--visualization-save-path "#TODO/visualization/"
