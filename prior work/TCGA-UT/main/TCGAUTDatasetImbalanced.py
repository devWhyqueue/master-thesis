import os

import torch
import numpy as np
from torch.utils.data import Dataset
import json
import pandas as pd
import ast

class TCGAUTDatasetImbalanced(Dataset):

    def __init__(
        self,
        dataset_path,
        feature_path,
        subsets=None,
        args_path=None,
        preload_features=False,
        device="cpu"
    ):
        super(TCGAUTDatasetImbalanced, self).__init__()
        # Save args
        self.dataset_path = dataset_path
        self.feature_path = feature_path
        self.subsets = subsets
        self.args_path = args_path
        self.preload_features = preload_features
        self.device = device

        self.dataset_original = pd.read_csv(self.dataset_path)
        if type(self.dataset_original["patch_ids"].iloc[0]) == str:
            self.dataset_original['patch_ids'] = self.dataset_original['patch_ids'].apply(ast.literal_eval)
        if self.subsets is not None:
            self.dataset_original = self.dataset_original[self.dataset_original["slide_id"].isin(self.subsets)]
        # flatten dataset
        dataset_arr = []
        for i in range(len(self.dataset_original)):
            for patch_id in self.dataset_original.iloc[i]["patch_ids"]:
                dataset_arr.append({"cancer_type": self.dataset_original.iloc[i]["cancer_type"], "slide_id": self.dataset_original.iloc[i]["slide_id"], "patch_id": patch_id})
        self.dataset = pd.DataFrame(dataset_arr)
        
        # load imbalanced dataset args
        if args_path is not None:
            with open(self.args_path) as f:
                self.args = json.load(f)
        
        # (optionally) pre-load features
        if self.preload_features:
            print("Loading features into RAM")
            features_arr = []
            for cls in self.dataset["cancer_type"].unique():
                slides = self.dataset[self.dataset["cancer_type"] == cls]["slide_id"].unique()
                features_slides = self.load_features(slides, [cls for _ in slides])
                features_arr.extend(features_slides)
            features_df = pd.DataFrame(features_arr)
            self.dataset = self.dataset.merge(features_df, on=["slide_id", "patch_id"], how="inner")
        else:
            self.dataset = None
        
        self.features_str_to_int_map = {feat: i for i, feat in enumerate(sorted(self.dataset["cancer_type"].unique()))}

    def get_class_sizes(self):
        return np.bincount(np.array([self.features_str_to_int_map[t] for t in self.dataset["cancer_type"]]))

    def load_features(self, slide_ids, targets):
        features_slides = []
        for i, slide_id in enumerate(slide_ids):
            target = targets[i]
            pt = torch.load(os.path.join(self.feature_path, target, f"{slide_id}.pt"))
            with open(os.path.join(self.feature_path, target, f"{slide_id}.json")) as f:
                patch_mapping = json.load(f)
            # determine indices of patches to load
            patch_ids = self.dataset_original[self.dataset_original["slide_id"] == slide_id]["patch_ids"].tolist()[0]
            idxs = [patch_mapping.index(f"{pid}.jpg") for pid in patch_ids]
            feature_slide = pt[idxs]
            features_slides.extend([{"features": patch_feature, "patch_id": patch_ids[j], "slide_id": slide_id} for j, patch_feature in enumerate(feature_slide)])
        return features_slides
        

    @staticmethod
    def bag_collate_fn(batch_list):
        """
        Custom collate function for this dataset.
        """
        col_batch = {}
        for key in batch_list[0].keys():
            if key == "features":
                col_batch[key] = torch.concat([batch[key] for batch in batch_list])
            elif key == "targets":
                col_batch[key] = torch.stack([batch[key] for batch in batch_list])
            else:
                col_batch[key] = torch.tensor([batch[key] for batch in batch_list])
        return col_batch
    

    def __len__(self):
        return len(self.dataset)
    
    def get_feature_dim(self):
        if not self.preload_features:
            test_slide_features = self.load_features([self.dataset.iloc[0]["slide_id"]], [self.dataset.iloc[0]["cancer_type"]])[0]["features"]
        else:
            test_slide_features = self.dataset.iloc[0]["features"]
        return test_slide_features.shape[-1]
    
    def get_n_classes(self):
        return len(self.features_str_to_int_map)
    
    def get_int_to_class_map(self):
        return {val: key for key, val in self.features_str_to_int_map.items()}
    
    def get_int_targets(self):
        return self.dataset["cancer_type"].apply(lambda x: self.features_str_to_int_map[x]).to_numpy()

    def __getitem__(self, idx):
        """
        :return: (dict)

        """
        slide_id = self.dataset.iloc[idx]["slide_id"]
        patch_id = self.dataset.iloc[idx]["patch_id"]
        target = self.dataset.iloc[idx]["cancer_type"]
        if self.preload_features:
            slide_features = self.dataset.iloc[idx]["features"]
        else:
            slide_features = [d["features"] for d in self.load_features([slide_id], [target]) if d["patch_id"] == patch_id][0]

        return {"slide_id": slide_id, "features": slide_features.to(self.device), "target_str": target, "target": self.features_str_to_int_map[target], "patch_id": patch_id}