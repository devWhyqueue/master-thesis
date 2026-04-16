#!/bin/bash

# run script with apptainer
for k in 3 9 27; do
    echo "Starting with k $k"
    sbatch run_training_knn.sh $k
done

