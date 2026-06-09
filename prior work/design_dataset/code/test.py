import os
import argparse
import json

import numpy as np
import torch
from TCGAUTDatasetImbalanced import TCGAUTDatasetImbalanced
from MLP import MLP
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from visualizations import plot_extended_confusion_matrix

def get_args():
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument('--dataset-structure-path', type=str, required=True)
    parser.add_argument('--feature-path', type=str, required=True)
    parser.add_argument('--preload-features', action='store_true',
                        help="Whether to preload all features into RAM before starting testing.")
    
    # Model args
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")

    # Visualization args
    parser.add_argument('--visualize', action='store_true')

    # Parse all args
    args = parser.parse_args()
    return args

def main():
    # Process args and create file structures
    args = get_args()
    
    # Create datasets
    device = torch.device(args.device) # put dataset on device
    dataset_val = TCGAUTDatasetImbalanced(args.dataset_structure_path, args.feature_path, None, None, args.preload_features, device)

    # Model loading
    with open(os.path.join(args.model_path, "args.json")) as f:
        args_model = json.load(f)
    torch.manual_seed(args_model["seed"])
    
    input_dim = dataset_val.get_feature_dim()
    output_dim = dataset_val.get_n_classes()
    model = MLP(input_dim, args_model["n_nodes_per_layer"], output_dim)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt")))
    
    # Model training
    model.to(device)
    print(f"Model: {model}", flush=True)

    # Model validation
    dl_val = DataLoader(
        dataset_val,
        batch_size=len(dataset_val),
        shuffle=False,
        drop_last=False
    )
    res = test_model(model, dl_val)
    val_acc = res["accuracy"]
    bal_acc = res["balanced_accuracy"]
    print(f"Finished model testing with an validation accuracy of {val_acc} and a balanced acccuracy of {bal_acc}", flush=True)
    print(f"Saving results...", flush=True)

    # Results saving
    out_file_res = os.path.join(args.model_path, "validation_results.json")
    with open(out_file_res, "w") as f:
        res_save = {(key):(val.tolist() if type(val) == np.ndarray else val) for key, val in res.items()}
        json.dump(res_save, f)

    # Visualizations
    if args.visualize:
        print("Visualizing...", flush=True)
        viz_path = os.path.join(args.model_path, "visualizations")
        os.makedirs(viz_path, exist_ok=True)
        int_to_class_map = dataset_val.get_int_to_class_map()
        class_names = [int_to_class_map[i] for i in range(dataset_val.get_n_classes())]
        fig, _ = plot_extended_confusion_matrix(res["confusion_matrix"], res["recall_per_class"], res["precision_per_class"], class_names=class_names)
        fig.savefig(os.path.join(viz_path, "extended_confusion_matrix.png"), dpi=300, bbox_inches="tight")


    print("Done.", flush=True)

def test_model(model, dl_test, model_type="mlp", class_order=None):
    if model_type == "mlp":
        with torch.no_grad():
            model.eval()
            all_preds = np.array([])
            all_labels = np.array([])
            for batch in dl_test:
                label_np = batch["target"].detach().cpu().numpy()
                output = model(batch["features"].to(model.model[0].weight.dtype)).squeeze()
                pred_np = torch.argmax(output, dim=-1).detach().cpu().numpy()
                all_preds = np.concatenate([all_preds, pred_np])
                all_labels = np.concatenate([all_labels, label_np])
            model.train()
    elif model_type=="knn" or model_type == "ncc":
        all_preds = np.array([])
        all_labels = np.array([])
        for batch in dl_test:
            label_np = batch["target"].detach().cpu().numpy()
            output = model.predict(batch["features"].squeeze().detach().cpu().numpy())
            print(f"output: {output}", flush=True)
            all_preds = np.concatenate([all_preds, output])
            all_labels = np.concatenate([all_labels, label_np])
    correct = np.sum(all_preds == all_labels)
    acc = correct/len(all_labels)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds, labels=class_order)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        labels=class_order
    )
    int_to_class_map = dl_test.dataset.get_int_to_class_map()
    if class_order is not None:
        class_names = [int_to_class_map[i] for i in class_order]
    else:
        class_names = [int_to_class_map[i] for i in range(dl_test.dataset.get_n_classes())]
    return {"accuracy": acc, "balanced_accuracy": bal_acc, "confusion_matrix": conf_mat, "precision_per_class": precision, "recall_per_class": recall, "class_order": class_order, "class_names": class_names, "preds": all_preds, "labels": all_labels}

if __name__ == '__main__':
    main()