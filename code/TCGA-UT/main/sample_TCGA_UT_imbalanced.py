import os
import json
import argparse
import numpy as np
from sample_TCGA_UT_balanced import sample_balanced_from_dataset_structure
from visualizations import number_of_slides_per_class_bar
import pandas as pd
import ast

def region_int(s):
    num = int(s)
    if num <= 0 or num > 10:
        raise argparse.ArgumentTypeError(f"Number needs to be at least 1 and at most 10, but {num} is not.")
    return num

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--balanced-dataset-path', type=str, required=True)
    parser.add_argument('--file-save-path', type=str, required=True)
    parser.add_argument('--parameter', type=float, required=True)
    parser.add_argument('--dataset-size', type=int, required=True)
    parser.add_argument('--n-regions-per-slide', type=int, default=None, required=True, help="Determines the number of regions to be sampled per slide. Use in conjunction with --n-patches-per-region. If None, all patches will be used.")
    parser.add_argument('--n-patches-per-region', type=region_int, default=10, required=True, help="Ignored if --n-regions-per-slide is None.")
    parser.add_argument('--overflow-strategy', type=str, choices=["none", "redistribute"], default="none")
    parser.add_argument('--class-order-file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--store-class-names', action='store_true')

    # balanced validation dataset args
    parser.add_argument('--sample-balanced-validation', action='store_true')
    parser.add_argument('--n-slides-per-class', type=int, default=None)

    args = parser.parse_args()
    if args.sample_balanced_validation and args.n_slides_per_class is None: raise argparse.ArgumentTypeError("--sample-balanced-validation is set, but --n-slides-per-class is not given.")
    return args

def main(): 
    args = get_args()

    df = pd.read_csv(args.balanced_dataset_path)
    # determine order of largest to smallest class
    if args.class_order_file is None:
        df = df.sort_values(["original_class_size", "cancer_type"], ascending=[False, True])
    else:
        with open(args.class_order_file) as f:
            class_order = json.load(f)
        print(f"class order: {class_order}", flush=True)
        class_order_idx = {v:i for i, v in enumerate(class_order)}
        df = df.sort_values(["cancer_type"], key=lambda s: s.map(class_order_idx))
    class_names = df["cancer_type"].unique()[::-1].tolist() # we later want to store this in ASCENDING order of class size
    if type(df["patch_ids"].iloc[0]) == str:
        df['patch_ids'] = df['patch_ids'].apply(ast.literal_eval)
    n_classes = len(df["cancer_type"].unique())
    class_size_balanced = len(df[df["cancer_type"] == df["cancer_type"][0]])

    out_path = os.path.join(args.file_save_path, f"TCGA-UT_imbalanced_parameter={args.parameter}_dataset_size={args.dataset_size}_seed={args.seed}")
    os.makedirs(out_path, exist_ok=True)

    if args.sample_balanced_validation:
        df_balanced = sample_balanced_from_dataset_structure(df, args.n_slides_per_class, n_patches=None, seed=args.seed, store_original_class_sizes=False)
        slide_ids_balanced = df_balanced["slide_id"].tolist()
        file_path_val = os.path.join(out_path, f"balanced_validation_n_slides_per_class={args.n_slides_per_class}_seed={args.seed}.csv")
        df_balanced.to_csv(file_path_val, index=False)
    else:
        slide_ids_balanced = []
    df = df[~df["slide_id"].isin(slide_ids_balanced)]

    max_size = (class_size_balanced - args.n_slides_per_class)
    
    p_unnormalized = np.array([np.power(c+1,-args.parameter) for c in range(n_classes)])
    p = p_unnormalized/np.sum(p_unnormalized)

    N_target = p*args.dataset_size
    if args.overflow_strategy == "none":
        assert N_target[0] <= max_size, f"Not enough samples for sampling {p[0]}% of the dataset from class 0."
    elif args.overflow_strategy == "redistribute":
        capped = N_target > max_size
        N_target[capped] = max_size
        N_rem = args.dataset_size - N_target[capped].sum()
        while True:
            p_unnormalized = np.array([np.power(c+1,-args.parameter) for c in range((~capped).sum())])
            p = p_unnormalized/np.sum(p_unnormalized)
            N_target[~capped] = p*N_rem
            newly_capped = N_target > max_size
            capped[N_target > max_size] = True
            N_target[capped] = max_size
            N_rem = N_rem - N_target[newly_capped].sum()
            if newly_capped.sum() == 0:
                break

    N_target_floor = np.floor(N_target)
    n_slides_per_class = dict(zip(df["cancer_type"].unique(), N_target_floor))
    fractional_remainders = dict(zip(df["cancer_type"].unique(), N_target - N_target_floor))
    generator = np.random.default_rng(args.seed)
    def sample_group(g):
        n = int(n_slides_per_class[g.name])
        return g.sample(n=n, random_state=generator)
    df_sampled = (
        df
        .groupby("cancer_type", group_keys=False, sort=False)
        .apply(sample_group)
    )
    df_remaining = df[~df["slide_id"].isin(df_sampled["slide_id"])]
    n_sampled = len(df_sampled)

    # distribute missing sample numbers across classes with largest fractional remainders
    n_missing_samples = args.dataset_size - n_sampled
    i = 0
    for cls, _ in sorted(fractional_remainders.items(), key=lambda item: item[1])[::-1]:
        new_sample = df_remaining[df_remaining["cancer_type"] == cls].sample(random_state=generator)
        df_sampled = pd.concat([df_sampled, new_sample], ignore_index=True)
        i += 1
        if i >= n_missing_samples: break

    # take only desired amount of patches
    if args.n_regions_per_slide is not None:
        assert np.all(df_sampled["patch_ids"].apply(len) >= args.n_regions_per_slide*args.n_patches_per_region), f"Not enough patches in original dataset to sample {args.n_regions_per_slide}x{args.n_patches_per_region} patches."
        def make_index_array(groups, patches, total_patches_per_group):
            return (np.arange(groups)[:, None] * total_patches_per_group + np.arange(patches)).ravel()
        df_sampled["patch_ids"] = df_sampled["patch_ids"].apply(lambda x: sorted(x, key=lambda item:(int(item.split("_")[0]), int(item.split("_")[1]))))
        patch_selection_indices = make_index_array(args.n_regions_per_slide, args.n_patches_per_region, 10)
        df_sampled["patch_ids"] = df_sampled["patch_ids"].apply(lambda x: np.array(x)[patch_selection_indices].tolist())
    df_sampled = df_sampled.sort_values(["original_class_size", "cancer_type"], ascending=[False, True])

    file_path = os.path.join(out_path, "imbalanced_dataset.csv")
    df_sampled.to_csv(file_path, index=False)

    args_path = os.path.join(out_path, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f)

    if args.store_class_names:
        with open(os.path.join(out_path, "class_names.json"), "w") as f:
            json.dump(class_names, f)

    if args.visualize:
        viz_path = os.path.join(out_path, "visualizations")
        os.makedirs(viz_path, exist_ok=True)
        fig_bar, _ = number_of_slides_per_class_bar(df_sampled)
        fig_bar.savefig(os.path.join(viz_path, "number_of_slides_per_class_bar_imbalanced.png"), dpi=300, bbox_inches="tight")
        if args.sample_balanced_validation:
            fig_bar, _ = number_of_slides_per_class_bar(df_balanced)
            fig_bar.savefig(os.path.join(viz_path, "number_of_slides_per_class_bar_balanced_validation.png"), dpi=300, bbox_inches="tight")

    print(f"Stored an unbalanced version of TCGA-UT with {args.dataset_size} slides and parameter={args.parameter} patches per slide in {file_path}.")




if __name__ == '__main__':
    main()
