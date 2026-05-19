import os
import argparse
import json
import time

import numpy as np
import torch
import pandas as pd
from TCGAUTDatasetImbalanced import TCGAUTDatasetImbalanced
from MLP import MLP
from torch.utils.data import DataLoader
from LossFactory import LossFactory
from test import test_model
from SKLearnModel import SKLearnModel
import pickle
from visualizations import plot_extended_confusion_matrix
from BatchBalancingSampler import BatchBalancingSampler

def positive_int(s):
    num = int(s)
    if num < 0:
        raise argparse.ArgumentTypeError(f"Number needs to be positive, but {num} is not.")
    return num

def positive_float(s):
    num = float(s)
    if num < 0:
        raise argparse.ArgumentTypeError(f"Number needs to be positive, but {num} is not.")
    return num

def get_args():
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument('--dataset-structure-path', type=str, required=True)
    parser.add_argument('--validation-dataset-structure-path', type=str, default=None)
    parser.add_argument('--feature-path', type=str, required=True)
    parser.add_argument('--args-path', type=str, default=None)
    parser.add_argument('--preload-features', action='store_true',
                        help="Whether to preload all features into RAM before starting training.")

    """
    # Split args
    parser.add_argument('--split-path', type=str, default=None)
    parser.add_argument('--train-subsets', default=['train'], nargs='+', type=str,
                        help='Split subsets that are used for training.')
    parser.add_argument('--val-subsets', default=['test'], nargs='+', type=str,
                        help='Split subsets that are used for validation.')
    parser.add_argument('--test-subsets', default=None, nargs='+', type=str,
                        help='Split subsets that are used for testing.')
    """
    
    # Model args
    parser.add_argument('--model', default="mlp", choices=["mlp", "knn", "ncc"])
    parser.add_argument('--k', type=positive_int)
    parser.add_argument('--n-nodes-per-layer', default=[100], nargs='+', type=int)
    parser.add_argument('--results-save-path', type=str, required=True)
    parser.add_argument('--store-timestamp', action='store_true')
    parser.add_argument('--loss', default="cross_entropy", choices=["cross_entropy", "focal_loss"])
    parser.add_argument('--gamma', default=2.0, type=positive_float)
    parser.add_argument('--alpha', default="uniform", type=str, choices=["uniform", "inverse_class_frequency"])
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--batch-balancing', action='store_true')
    parser.add_argument('--training-method', type=str, default="single_example",
                        choices=["single_example", "oko"],
                        help="Training objective to use for MLP models.")
    parser.add_argument('--oko-k', type=positive_int, default=1,
                        help="Number of odd classes k for OKO set construction (set size is k+2).")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=0)

    # Visualization args
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--class-names-path', type=str, default=None)

    # Parse all args
    args = parser.parse_args()
    return args

def main():
    # Process args and create file structures
    args = get_args()
    torch.manual_seed(args.seed)
    
    """
    # load splits
    if args.split_path is not None:
        with open(args.split_path) as f:
            splits = json.load(f) # {"1": {"adeno": [], "bla": []}, "2": ...}
        splits_train = {}
        for subset in args.train_subsets:
            for cls, slide_ids in splits[subset].items():
                if cls in splits_train:
                    splits_train[cls].extend(slide_ids)
                else:
                    splits_train[cls] = slide_ids
        splits_val = {}
        for subset in args.val_subsets:
            for cls, slide_ids in splits[subset].items():
                if cls in splits_val:
                    splits_val[cls].extend(slide_ids)
                else:
                    splits_val[cls] = slide_ids
        if args.test_subsets:
            splits_test = {}
            for subset in args.test_subsets:
                for cls, slide_ids in splits[subset].items():
                    if cls in splits_test:
                        splits_test[cls].extend(slide_ids)
                    else:
                        splits_test[cls] = slide_ids
    else:
        splits_train = None
        splits_val = None
        split_test = None
    """
    
    # Create datasets
    device = torch.device(args.device) # put dataset on device
    dataset_train = TCGAUTDatasetImbalanced(args.dataset_structure_path, args.feature_path, None, args.args_path, args.preload_features, device)
    if args.validation_dataset_structure_path is not None:
        dataset_val = TCGAUTDatasetImbalanced(args.validation_dataset_structure_path, args.feature_path, None, args.args_path, args.preload_features, device)
    """
    if args.test_subsets:
        dataset_test = TCGAUTDatasetImbalanced(args.dataset_structure_path, args.feature_path, splits_test, args.args_path, args.preload_features, device)
    """
    if args.batch_balancing:
        sampler = BatchBalancingSampler(dataset_train.get_int_targets(), dataset_train.get_n_classes(), seed=args.seed)
    else:
        sampler = None
    dl_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=not args.batch_balancing,
            drop_last=False
        )
    if args.validation_dataset_structure_path is not None:
        dl_val = DataLoader(
            dataset_val,
            batch_size=len(dataset_val),
            shuffle=False,
            drop_last=False
        )
    else:
        dl_val = None
    
    if args.model == "mlp":
        # Model definition
        input_dim = dataset_train.get_feature_dim()
        output_dim = dataset_train.get_n_classes()
        model = MLP(input_dim, args.n_nodes_per_layer, output_dim)
        
        # Model training
        model.to(device)
        print(f"Model: {model}", flush=True)
        if args.training_method == "single_example":
            criterion = LossFactory.build(args.loss, gamma=args.gamma, alpha=args.alpha, n_classes=output_dim, class_counts=dataset_train.get_class_sizes())
        elif args.training_method == "oko":
            criterion = LossFactory.build("oko_hard_loss")
        print(f"criterion: {criterion}", flush=True)

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        if args.training_method == "single_example":
            train(model, dl_train, args.n_epochs, criterion, optimizer, dl_val=dl_val)
        elif args.training_method == "oko":
            train_oko(model,
                      dataset_train,
                      args.n_epochs,
                      criterion,
                      optimizer,
                      dl_val=dl_val,
                      k_oko=args.oko_k,
                      seed=args.seed)

    elif args.model == "knn" or args.model == "ncc":
        required_sklearn_args = SKLearnModel.get_required_arguments_per_model(args.model)
        argument_aliases = SKLearnModel.get_argument_aliases()
        for arg in required_sklearn_args:
            if vars(args)[argument_aliases[arg]] is None: raise argparse.ArgumentError(f"Argument {arg} required but {argument_aliases[arg]} not passed!")
        sklearn_args = {
            arg: vars(args)[argument_aliases[arg]] for arg in required_sklearn_args
        }
        model = SKLearnModel(args.model, sklearn_args)
        if not args.preload_features:
            feats_df = pd.DataFrame(dataset_train.load_features(dataset_train.dataset["slide_ids"].to_list(), dataset_train.dataset["cancer_type"].to_list()))["features"]
        else:
            feats_df = dataset_train.dataset["features"]
        X = np.array(feats_df.apply(lambda x: x.squeeze().cpu().tolist()).to_list())
        y = dataset_train.get_int_targets()
        model.fit(X, y)
    print("Finished model training", flush=True)

    timestamp = time.time_ns() // 1_000_000
    if args.store_timestamp:
        base_path = os.path.join(args.results_save_path, str(timestamp))
    else:
        base_path = args.results_save_path

    # Model validation
    if args.validation_dataset_structure_path is not None:
        if args.class_names_path is None:
            class_order = np.argsort(dataset_train.get_class_sizes())
        else:
            with open(args.class_names_path) as f:
                class_names = json.load(f)
            class_order = [dataset_val.features_str_to_int_map[cls] for cls in class_names]
        res = test_model(model, dl_val, model_type=args.model, class_order=class_order)
        val_acc = res["accuracy"]
        bal_acc = res["balanced_accuracy"]
        print(f"Finished model testing with an validation accuracy of {val_acc} and a balanced acccuracy of {bal_acc}", flush=True)
        if args.visualize:
            print("Visualizing...", flush=True)
            viz_path = os.path.join(base_path, "visualizations")
            os.makedirs(viz_path, exist_ok=True)
            int_to_class_map = dataset_val.get_int_to_class_map()
            class_names = [int_to_class_map[i] for i in class_order]
            fig, _ = plot_extended_confusion_matrix(res["confusion_matrix"], res["recall_per_class"], res["precision_per_class"], class_names=class_names)
            fig.savefig(os.path.join(viz_path, "extended_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    print(f"Saving...", flush=True)

    # Model saving
    os.makedirs(base_path, exist_ok=True)
    if args.model == "mlp":
        out_file_model = os.path.join(base_path, "model.pt")
        torch.save(model.state_dict(), out_file_model)
    elif args.model == "knn" or args.model == "ncc":
        out_file_model = os.path.join(base_path, "model.pkl")
        with open(out_file_model, "wb") as f:
            pickle.dump(model.model, f)

    # Results saving
    if args.validation_dataset_structure_path is not None:
        out_file_res = os.path.join(base_path, "validation_results.json")
        res_save = {(key):(val.tolist() if type(val) == np.ndarray else val) for key, val in res.items()}
        with open(out_file_res, "w") as f:
            json.dump(res_save, f)
    out_file_args = os.path.join(base_path, "args.json")
    with open(out_file_args, "w") as f:
        json.dump(vars(args), f)

    print("Done.", flush=True)
    

def train(model, dataloader, n_epochs, criterion, optimizer, dl_val=None):

    print("Beginning training", flush=True)
    model.train()

    for epoch in range(1, n_epochs+1):
        train_single_epoch(model, dataloader, criterion, optimizer, epoch, dl_val=dl_val)


def train_oko(model: MLP,
              dataset: TCGAUTDatasetImbalanced,
              n_epochs: int,
              criterion,
              optimizer,
              dl_val=None,
              k_oko: int = 1,
              seed: int = 0):
    """
    Odd-k-out (OKO) training as described in
    'Set Learning for Accurate and Calibrated Models' (Muttenthaler et al., 2024).

    The network is applied to individual examples, logits are summed over
    k+2 examples in a set, and a cross-entropy loss is computed on the pair class.
    At test time the model is used on single examples as usual.
    """
    print("Beginning OKO training", flush=True)
    model.train()

    rng = np.random.default_rng(seed)

    for epoch in range(1, n_epochs + 1):
        train_single_epoch_oko(
            model=model,
            dataset=dataset,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            k_oko=k_oko,
            rng=rng,
        )

        # Evaluate single-example performance on validation set, if provided
        if dl_val is not None:
            res = test_model(model, dl_val)
            print(f"Validation Accuracy = {res['accuracy']:.5f}, "
                  f"Validation Balanced Accuracy = {res['balanced_accuracy']:.5f}",
                  flush=True)

def train_single_epoch(model:MLP, dataloader, criterion, optimizer, epoch, dl_val=None):
    print(f"Beginning epoch {epoch}", flush=True)

    counter = 0
    right_preds = 0
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch["features"].to(model.model[0].weight.dtype)).squeeze()
        loss = criterion(output, batch["target"])
        loss.backward()
        optimizer.step()

        pred_np = torch.argmax(output, dim=-1).detach().cpu().numpy()
        label_np = batch["target"].detach().cpu().numpy()
        right_preds += np.sum(pred_np == label_np)

        counter += len(batch["target"])

        print(f"Epoch {epoch}, Inputs encoutered {counter}, Training Accuracy = {right_preds/counter:.5f}", flush=True)
    if dl_val is not None:
        res = test_model(model, dl_val)
        print(f"Validation Accuracy = {res["accuracy"]:.5f}, Validation Balanced Accuracy = {res["balanced_accuracy"]:.5f}", flush=True)


def train_single_epoch_oko(model: MLP,
                           dataset: TCGAUTDatasetImbalanced,
                           criterion,
                           optimizer,
                           epoch: int,
                           k_oko: int,
                           rng: np.random.Generator):
    print(f"Beginning OKO epoch {epoch}", flush=True)

    device = next(model.parameters()).device
    dtype = model.model[0].weight.dtype

    n_classes = dataset.get_n_classes()
    int_targets = dataset.get_int_targets()

    # Precompute indices per class for fast set sampling
    class_indices = [
        np.where(int_targets == c)[0]
        for c in range(n_classes)
    ]

    # As in the paper, we cap the number of sets per epoch by the number of training points
    max_sets_per_epoch = len(dataset)

    right_preds = 0
    n_sets = 0

    for _ in range(max_sets_per_epoch):
        idxs, y_pair = sample_oko_set_indices(
            n_classes=n_classes,
            class_indices=class_indices,
            k_oko=k_oko,
            rng=rng,
        )
        # Not enough odd classes for a valid set
        if idxs is None:
            continue

        x_set, target = build_oko_batch(
            dataset=dataset,
            idxs=idxs,
            y_pair=y_pair,
            device=device,
            dtype=dtype,
        )

        optimizer.zero_grad()

        # Model applied to individual examples, logits summed over the set
        logits_per_example = model(x_set).squeeze()
        if logits_per_example.ndim == 1:
            logits_per_example = logits_per_example.unsqueeze(0)
        set_logits = logits_per_example.sum(dim=0, keepdim=True)
        loss = criterion(set_logits, target)

        loss.backward()
        optimizer.step()

        pred = torch.argmax(set_logits, dim=-1)
        right_preds += int((pred == target).sum().item())
        n_sets += 1

        if n_sets % 50 == 0:
            print(
                f"OKO Epoch {epoch}, Sets encountered {n_sets}, "
                f"Training Accuracy (pair class) = {right_preds / n_sets:.5f}",
                flush=True,
            )

    if n_sets > 0:
        print(
            f"Finished OKO epoch {epoch}, "
            f"Training Accuracy (pair class) = {right_preds / n_sets:.5f}",
            flush=True,
        )
    else:
        print(f"Finished OKO epoch {epoch}, no valid sets were constructed.", flush=True)


def sample_oko_set_indices(
    n_classes: int,
    class_indices,
    k_oko: int,
    rng: np.random.Generator,
):
    """
    Sample indices for a single OKO set and return (idxs, y_pair).

    If not enough odd classes with data are available for the sampled pair
    class, returns (None, None).
    """
    # Classes eligible to be the pair class must have at least two examples
    valid_pair_classes = [c for c in range(n_classes) if len(class_indices[c]) >= 2]
    if len(valid_pair_classes) == 0:
        raise RuntimeError("No class has at least two examples; cannot construct OKO sets.")

    y_pair = int(rng.choice(valid_pair_classes))

    # Sample two distinct examples from the pair class
    pair_idxs = rng.choice(class_indices[y_pair], size=2, replace=False)

    # Sample k odd classes (distinct from pair class) that have at least one example
    available_odd = [c for c in range(n_classes) if c != y_pair and len(class_indices[c]) > 0]
    if len(available_odd) < k_oko:
        return None, None

    odd_classes = rng.choice(available_odd, size=k_oko, replace=False)
    odd_indices = [rng.choice(class_indices[c]) for c in odd_classes]

    idxs = np.concatenate([pair_idxs, np.array(odd_indices, dtype=np.int64)])
    return idxs, y_pair


def build_oko_batch(
    dataset: TCGAUTDatasetImbalanced,
    idxs,
    y_pair: int,
    device,
    dtype,
):
    """
    Given a set of indices and the pair class, build the OKO batch:
    a features tensor of shape (k+2, feature_dim) and a scalar target tensor.
    """
    feats = []
    for idx in idxs:
        item = dataset[int(idx)]
        feat = item["features"].to(device=device, dtype=dtype)
        feats.append(feat.unsqueeze(0))
    x_set = torch.cat(feats, dim=0)

    target = torch.tensor([y_pair], device=device)
    return x_set, target


if __name__ == '__main__':
    main()