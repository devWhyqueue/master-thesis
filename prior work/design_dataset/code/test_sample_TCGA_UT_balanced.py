import os
import json
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--balanced-dataset-path', type=str, required=True)
    parser.add_argument('--original-dataset-path', type=str, required=True)
    parser.add_argument('--n-slides-per-class', type=int, required=True)
    parser.add_argument('--n-patches-per-slide', type=int, required=True)

    args = parser.parse_args()
    return args

def main(): 
    args = get_args()
    with open(args.balanced_dataset_path) as f:
        dataset_structure = json.load(f)
    for cls in dataset_structure.keys():
        print(f"{cls}: {len(dataset_structure[cls])}")
        assert len(dataset_structure[cls]) == args.n_slides_per_class, f"Number of slides for class {cls} is not {args.n_slides_per_class}!"
        for slide_id in dataset_structure[cls].keys():
            assert len(dataset_structure[cls][slide_id]) == args.n_patches_per_slide, f"Number of patches for slide {slide_id} is not {args.n_patches_per_class}!"
            sorting_key = lambda item:(int(item.split("_")[0]), int(item.split("_")[1]))
            sets = np.unique([int(patch_id.split("_")[0]) for patch_id in sorted(os.listdir(os.path.join(args.original_dataset_path, cls, slide_id)), key=sorting_key)])
            for i, patch_id in enumerate(sorted(dataset_structure[cls][slide_id], key=sorting_key)):
                patch_id_split = patch_id.split("_")
                assert int(patch_id_split[0]) == sets[i // 10], f"Patch {patch_id} of slide {slide_id} of class {cls} should not have been selected. It should belong to set {sets[i // 10]}."
                assert int(patch_id_split[1]) == i % 10, f"Patch {patch_id} of slide {slide_id} of class {cls} should not have been selected. It should be patch {i % 10} of set {sets[i // 10]}."
            print(f"All tests passed for slide {slide_id}.")
        print(f"All tests passed for class {cls}.")
    print(f"All tests passed.")



if __name__ == '__main__':
    main()
