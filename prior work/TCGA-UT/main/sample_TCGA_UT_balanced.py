import os
import json
import argparse
import numpy as np
from utils import convert_dataset_structure_to_dataframe
import ast

def region_int(s):
    num = int(s)
    if num <= 0 or num > 10:
        raise argparse.ArgumentTypeError(f"Number needs to be at least 1 and at most 10, but {num} is not.")
    return num

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--file-save-path', type=str, required=True)
    parser.add_argument('--n-slides-per-class', type=int, required=True)
    parser.add_argument('--n-patches-per-slide', type=int, required=True, help="Determines the number of patches to be sampled per slide.")
    parser.add_argument('--slide-id-exclusion-path', type=str, default=None)
    parser.add_argument('--store-slide-ids', action='store_true',
                        help="Whether to store all slides ids that have been included in this dataset.")
    parser.add_argument('-seed', type=int, default=0)

    args = parser.parse_args()
    assert args.n_patches_per_slide is not None or args.n_regions_per_slide is not None, argparse.ArgumentTypeError("Either --n-patches-per-slide or --n-regions-per-slide has to be set.")
    return args

def get_dataset_structure(path, slides_to_exclude=[]):
    classes = [cls for cls in os.listdir(path) if not cls.startswith(".")]
    sorting_key = lambda item:(int(item.split("_")[0]), int(item.split("_")[1]))
    dataset_structure = {cls: 
                         {slide_id: 
                          [patch_name for patch_name in sorted(os.listdir(os.path.join(path, cls, slide_id)), key=sorting_key) if patch_name.endswith(".jpg")] 
                          for slide_id in os.listdir(os.path.join(path, cls)) if not slide_id.startswith(".") and not slide_id in slides_to_exclude
                          } for cls in classes}
    
    return dataset_structure

def sample_balanced_from_dataset_structure(df, n_slides, n_patches=None, seed=0, store_original_class_sizes=True):
    
    original_dataset_sizes = df["cancer_type"].value_counts()
    if type(df["patch_ids"].iloc[0]) == str:
        df['patch_ids'] = df['patch_ids'].apply(ast.literal_eval)

    if n_patches is not None:
        df_filtered = df[df["patch_ids"].apply(len) >= n_patches]
    else:
        df_filtered = df
    df_filtered = df_filtered[df_filtered.groupby("cancer_type")["cancer_type"].transform("size") >= n_slides]
    if n_patches is not None:
        assert len(df_filtered) != 0, f"Not enough slides with {n_patches} patches."

    generator = np.random.default_rng(seed)
    df_slides_sampled = df_filtered.groupby("cancer_type", group_keys=False).sample(n=n_slides, random_state=generator)

    if n_patches is not None:
        df_slides_sampled["patch_ids"] = df_slides_sampled["patch_ids"].apply(lambda x: sorted(x, key=lambda item:(int(item.split("_")[0]), int(item.split("_")[1])))[:n_patches])
    else:
        df_slides_sampled["patch_ids"] = df_slides_sampled["patch_ids"].apply(lambda x: sorted(x, key=lambda item:(int(item.split("_")[0]), int(item.split("_")[1]))))

    if store_original_class_sizes:
        df_slides_sampled["original_class_size"] = df["cancer_type"].map(original_dataset_sizes)

    return df_slides_sampled

def main(): 
    args = get_args()
    path = args.dataset_path
    n_slides = args.n_slides_per_class
    n_patches = args.n_patches_per_slide

    if args.slide_id_exclusion_path is None: 
        slides_to_exclude = []
    else: 
        with open(args.slide_id_exclusion_path) as f:
            slides_to_exclude = json.load(f)

    dataset_structure = get_dataset_structure(path, slides_to_exclude=slides_to_exclude)
    df = convert_dataset_structure_to_dataframe(dataset_structure)
    df_sampled = sample_balanced_from_dataset_structure(df, n_slides, n_patches=n_patches, seed=args.seed)
    
    os.makedirs(args.file_save_path, exist_ok=True)
    file_path = os.path.join(args.file_save_path, f"TCGA-UT_{n_slides}_slides_per_class_{n_patches}_patches_per_slide_seed={args.seed}.csv")
    df_sampled.to_csv(file_path, index=False)
    if args.store_slide_ids:
        slide_ids = df_sampled["slide_id"].unique()
        file_path_slide_ids = os.path.join(args.file_save_path, f"slide_ids_TCGA-UT_{n_slides}_slides_per_class_{n_patches}_patches_per_slide_seed={args.seed}.json")
        with open(file_path_slide_ids, "w") as f:
            json.dump(slide_ids, f)
    
    print(f"Stored a balanced version of TCGA-UT with {n_slides} slides per class and {n_patches} per slide in {file_path}.")



if __name__ == '__main__':
    main()
