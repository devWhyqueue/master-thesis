import argparse
import json

import pandas as pd
import numpy as np

def percentage_float(arg):
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0.0 or f > 1.0:
        raise argparse.ArgumentTypeError(f"Argument must be <=1.0 and >=0.0")
    return f

def positive_int(arg):
    try:
        i = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be an integer number")
    if i < 0:
        raise argparse.ArgumentTypeError(f"Argument must be >=0.")
    return i

def get_args():
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument('--dataset-structure-path', type=str, required=True)
    parser.add_argument('--metadata-path', type=str, required=True)

    # Saving args
    parser.add_argument('--out-file', type=str, required=True)

    # Split args
    parser.add_argument('--train-test', action="store_true", help="If set, a simple train-test split is performed. Otherwise, a CV will be performed.")
    parser.add_argument('--train-proportion', default=0.8, type=percentage_float)
    parser.add_argument('--n-folds', default=2, type=positive_int, help="How many CV folds should be created. Ignored if --train-test is set.")
    parser.add_argument('--seed', default=0, type=int)

    # Parse all args
    args = parser.parse_args()
    return args

def main():
    # Process args and create file structures
    args = get_args()
    
    # load metadata
    metadata = pd.read_csv(args.metadata_path)
    with open(args.dataset_structure_path) as f:
        dataset_structure = json.load(f)
    
    # split
    if args.train_test:
        slide_ids_and_cls = [(slide_id, cls) for cls, slides in dataset_structure.items() for slide_id in slides.keys()]
        tss = metadata["tissue_source_site"].unique()
        total_size = len(slide_ids_and_cls)
        train_slide_ids_cls = []
        generator = np.random.default_rng(args.seed)
        tss_shuffled = generator.choice(tss, size=len(tss), replace=False)
        for code in tss_shuffled:
            slide_ids_tss = metadata[metadata["tissue_source_site"] == code]["slide_submitter_id"].to_list()
            slide_ids_cls_tss = [tup for tup in slide_ids_and_cls if tup[0] in slide_ids_tss]
            slide_ids_and_cls = list(set(slide_ids_and_cls) - set(slide_ids_cls_tss))
            train_slide_ids_cls.extend(slide_ids_cls_tss)
            if len(train_slide_ids_cls)/total_size >= args.train_proportion:
                break
        test_slide_ids_cls = slide_ids_and_cls
        train_dict = {key: [] for key in dataset_structure.keys()}
        for tup in train_slide_ids_cls: train_dict[tup[1]].append(tup[0])
        test_dict = {key: [] for key in dataset_structure.keys()}
        for tup in test_slide_ids_cls: test_dict[tup[1]].append(tup[0])

        split_dict = {
            "train": train_dict,
            "test": test_dict
        }

        with open(args.out_file, "w") as f:
            json.dump(split_dict, f)
        
        print(f"Finished a train-test split with approx. {args.train_proportion*100}% of samples belonging to train (Actual number: {len(train_slide_ids_cls)/total_size * 100}%).")
    else:
        raise NotImplementedError("CV not yet implemented!")



if __name__ == '__main__':
    main()