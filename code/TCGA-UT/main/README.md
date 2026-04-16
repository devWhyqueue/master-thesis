# TCGA-UT-imbalanced
## Introduction
To investigate how different mitigation strategies help handle highly imbalanced data sets in histopathological settings, we introduce TCGA-UT-imbalanced, a data set based on TCGA-UT, which allows the user to control the degree of imbalance with a single parameter.
## Usage
### 1. Sample a balanced version of TCGA-UT 
To be able to sample an imbalanced version in a highly controlled manner, we first sample a balanced version of TCGA-UT, i.e., a version where all classes have the same number of slides and patches. 
To do this, run the Python script `sample_TCGA_UT_balanced.py`. It takes as arguments: 
- `--dataset-path`: The top-level path where TCGA-UT is stored. The directory at this path should have the following structure:
```
--datset-path
├──── class1
│     ├──── slide_1
│     │     └──── patch_1.jpg
│     │     └──── ...
│     └──── ...
├──── class2
│     ├──── slide_1
│     │     └──── patch_1.jpg
│     │     └──── ...
│     └──── ...
```
- `--file-save-path`: The script will store a CSV-file in this directory.
- `--n-slides-per-class`: The number of slides each class will have in the final balanced data set. Any class with fewer slides is excluded.
- `--n-patches-per-slide`: The number of patches taken from each slide. The first `--n-patches-per-slide` in order of patch id are taken. Since TCGA-UT always has groups of patches of size 10, this number should be divisible by 10 to always take entire groups
- `--slide-id-exclusion-path` (optional): A path to a JSON-file with a list of slide ids to exclude during sampling.
- `--store-slide-ids` (flag): A flag that tells the script to store a JSON-file with a list of slide-ids that have already been sampled.
- `--seed` (default=0): The seed for the random sampling of slide ids selected from each class.  

The bash script found in `scripts/run_balanced_sampling.sh` can be used to run this on a slurm cluster. Fields that need to be adapted to your working directory structure are marked as `#TODO`

### 2. Sample an imbalanced version of TCGA-UT
Once you sample a balanced version TCGA-UT and have stored the corresponding CSV, you can use it to sample an imbalanced version, which again will be stored in a CSV. This is done with the Python script `sample_TCGA_UT_imbalanced.py`. It takes the arguments:
- `--balanced-dataset-path`: The path to the CSV containing the balanced version of TCGA-UT.
- `--file-save-path`: The output directory where the imbalanced data set should be stored.
- `--parameter`: The parameter that controls the degree of imbalance.
- `--dataset-size`: The number of samples in the final data set.
- `--n-regions-per-slide`: The patches of each slide always come from connected regions of 10 patches. This argument controls how many of the regions should be taken.
- `--n-patches-per-region`: This controls the number of patches taken from each region.
- `--overflow-strategy` ('none' or 'redistribute': It is possible that particular combinations of `--parameter` and `--dataset-size` do not work, as they could require more samples than are available from certain classes in certain settings. If 'none' is selected, the script will exit with an error in such a case. If 'redistribute' is selected, the next closest feasible distribution over the classes is picked.
- `--class-order-file`: Should be a JSON-file containing a list with class names. Allows the user to specify an order in which the classes are sampled and stored. This is useful to control which classes should be the largest, and which should be the smallest.
- `--seed`: The random seed used for sampling.
- `--visualize` (Flag): A flag that tells the script to store a visualization of the class distribution.
- `--store-class-names`: Stores the class names in order of smallest to largest. This is useful for later scripts, as it allows the user to visualize the classes in order from smalles to largest.
- `--sample-balanced-validation` (Flag): This flag tells the script to sample a balanced validation set. Given the same seed, this will always be the same set, independent of the parameter.
- `--n-slides-per-class`: Number of slides per class in the balanced validation data set.  

The bash script found in `scripts/run_imbalanced_sampling_arr.sh` can be used to sample multiple imbalanced data sets for a range of parameters. It relies on the script `scripts/run_imbalanced_sampling.sh`, which can be used to sample an imbalanced data set for a single parameter. Fields needed to fill in are marked as `#TODO`

### Training and Testing
You can train a few different models on the previously sampled imbalanced data sets. This is done with the Python script `train.py`. It takes as arguments:
- `--dataset-structure-path`: The path to the (imbalanced) CSV file containing the data set.
- `--validation-dataset-structure-path`: A validation set CSV file, as previously generated along with imbalanced data set.
- `--feature-path`: The path to the foundation model features of the patches.
- `--preload-features` (Flag): Indicates that the features should be pre-loaded into RAM before training.
- `--model` (Options: 'mlp', 'knn', 'nccc'): Which model should be trained.
- `--k`: The 'k' used by the KNN-model.
- `--n-nodes-per-layer`: A list of the number of nodes per layer in the MLP (this also determines the number of hidden layers).
- `--results-save-path`: Where the trained model and evaluation results should be stored.
- `--store-timestamp` (Flag): This tells the script to add one additonal directory level with a timestamp to `--results-save-path` before saving the results. Not currently used a lot.
- `--loss` (Options: 'cross_entropy', 'focal_loss'): Which loss to use.
- `--gamma`: The Gamma parameter of the Focal Loss. Indicates how much the loss should focus on hard examples.
- `--alpha` (Options: 'uniform', 'inverse_class_frequency'): The weights that are applied to the loss based on class membership of the current sample. 'uniform' makes all weights 1 and 'inverse_class_frequency' scales the weights inversely according to the class frequency.
- `--weight-decay`: Weight Decay.
- `--learning-rate`: Learning Rate.
- `--n-epochs`: Number of Epochs (If Batch Balancing is selected, this is still used to train the model for n-epochs\*(dataset-size//batch-size) batches.)
- `--batch-size`: Batch size.
- `--batch-balancing` (Flag): Wether Batch Balancing of each batch should be performed.
- `--device`: Device.
- `--seed`: The seed for the random initialization and random sampling in batch balancing.
- `--visualize` (Flag): Whether or not to store visualizations of the validation result.
- `--class-names-path`: A JSON-File indicating the order in which the classes should be displayed in the visualizations.

The bash script `scripts/run_training_arr.sh` runs the training of an MLP on imbalanced datasets for a range of parameters. It relies on the script `scripts/run_training.sh`, which can be used to train an MLP on a single imbalanced data set for a single parameter. Fields needed to fill in are marked as `#TODO`. Similar scripts exist for the training of KNN- and NCC-based models.

### Visualization
Several different visualization methods have been implemented. The below explains for each one how it can be used.
#### Confusion Matrix
Requires the argument `--method` to be set to `confusion_matrix`. It takes the first path stored in `--results-paths` and visualizes the average confusion matrix over all seeds found in the directory at `{--results-paths[0]}/{--parameter-name}={--parameters[0]}`.
#### Difference Confusion Matrix
Requires the argument `--method` to be set to `difference_confusion_matrix`. It takes the first two paths stored in `--results-paths` and gathers results from directories one level below these two results paths. Specifically it loads results over all seeds from the folders that match `{--parameter-name}={--parameters[i]}` where i is either 0 or 1, for each results path. Then it calculates the difference over the two average confusion matrices for the two results paths.
#### Scatter of the Recalls for two Parameters
Requires the argument `--method` to be set to `scatter_accuracies_of_two_parameters`. It takes the first two paths stored in `--results-paths` and gathers results from directories one level below these two results paths. Specifically it loads results over all seeds from the folders that match `{--parameter-name}={--parameters[i]}` where i is either 0 or 1, for each results path. Then it calculates the average recall per class and plots a scatter plot comparing the recalls for each class over the two parameters. For the coloring of the dots to work, it expects the class names in the results paths to be stored in order of ascending class size.  

----
The bash script `scripts/run_visualizations.sh` plots all of the above plots and fields that need to be filled in are marked as `#TODO`.
#### Point Plot to Compare Methods
Requires the argument `--method` to be set to `point_plot_compare_methods`. It takes all of the paths stored in `--results-paths` and gathers results from directories one level below these results paths. Specifically it loads results over all seeds from the folders that match `{--parameter-name}={--parameters[i]}` where i is either running from 0 to the number of results paths, or is just 0, if only one parameter was passed. Then it calculates the average recall per class and plots a point plot comparing the recalls for each class over the different results paths. For the coloring of the dots to work, it expects the class names in the results paths to be stored in order of ascending class size. Optionally, one can pass labels for each method in the `--methods` argument.  

----
The bash script `scripts/run_visualizations_point_plot.sh` plots the point plot and fields that need to be filled in are marked as `#TODO`.
#### Bar Plot for the Number of Slides per Class
Requires the argument `--method` to be set to `number_of_slides_per_class_bar`. It plots a bar plot with the number slides per class for the CSV-based dataset found in `--dataset-path`.  


## Not Up-to-date and Unused Files
The following files are currently not updated to reflect the most recent version of the code:
- `test.py`

The following files are currently unused and can be removed in a future version:
- `split.py`
- `test_sample_TCGA_UT_balanced.py`
