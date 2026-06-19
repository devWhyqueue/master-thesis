import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tcga_ut_imbalanced.cli.train_support import (
    build_mlp,
    make_dataloader,
    save_validation_plot,
    training_base_path,
    validation_class_order,
)
from tcga_ut_imbalanced.data.dataset import TCGAUTDatasetImbalanced
from tcga_ut_imbalanced.evaluation.metrics import test_model
from tcga_ut_imbalanced.evaluation.tuning_params import parse_tuning_params
from tcga_ut_imbalanced.models.sklearn import SKLearnModel
from tcga_ut_imbalanced.training.batch_sampler import BatchBalancingSampler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TrainContext:
    device: torch.device
    train_ds: TCGAUTDatasetImbalanced
    val_ds: TCGAUTDatasetImbalanced | None
    test_ds: TCGAUTDatasetImbalanced | None
    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader | None


def run_training(args: argparse.Namespace) -> None:
    """Run model training, validation, and persistence."""
    context = _build_context(args)
    model = _fit_model(args, context)
    logger.info("Finished model training")
    output_path = training_base_path(args)
    evaluate_and_save(args, context, model, output_path)
    logger.info("Done.")


def build_sklearn_model(
    args: argparse.Namespace, dataset_train: TCGAUTDatasetImbalanced
) -> SKLearnModel:
    """Build and train a scikit-learn model."""
    sklearn_args = _sklearn_args(args)
    model = SKLearnModel(args.model, sklearn_args)
    features = _sklearn_features(args, dataset_train)
    model.fit(features, dataset_train.get_int_targets())
    return model


def _build_context(args: argparse.Namespace) -> _TrainContext:
    device = torch.device(args.device)
    train_ds = _dataset(
        args.dataset_structure_path,
        args,
        device,
        args.dataset_split,
        synthetic_variant_epochs=_synthetic_variant_epochs(args, train=True),
    )
    val_ds = (
        _dataset(
            args.validation_dataset_structure_path,
            args,
            device,
            args.validation_dataset_split,
            synthetic_variant_epochs=_synthetic_variant_epochs(args, train=False),
        )
        if args.validation_dataset_structure_path
        else None
    )
    test_ds = (
        _dataset(
            args.test_dataset_structure_path,
            args,
            device,
            args.test_dataset_split,
            synthetic_variant_epochs=_synthetic_variant_epochs(args, train=False),
        )
        if args.test_dataset_structure_path
        else None
    )
    train_loader = make_dataloader(
        train_ds, args.batch_size, _sampler(args, train_ds), not args.batch_balancing
    )
    val_loader = (
        make_dataloader(val_ds, len(val_ds), shuffle=False)
        if val_ds is not None
        else None
    )
    test_loader = (
        make_dataloader(test_ds, len(test_ds), shuffle=False)
        if test_ds is not None
        else None
    )
    return _TrainContext(
        device, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    )


def _dataset(
    path: str,
    args: argparse.Namespace,
    device: torch.device,
    split_name: str | None = None,
    synthetic_variant_epochs: int | None = None,
) -> TCGAUTDatasetImbalanced:
    return TCGAUTDatasetImbalanced(
        path,
        args.feature_path,
        None,
        args.args_path,
        args.preload_features,
        device,
        args.feature_cache_path,
        split_name=split_name,
        synthetic_variant_epochs=synthetic_variant_epochs,
    )


def _sampler(
    args: argparse.Namespace, dataset: TCGAUTDatasetImbalanced
) -> BatchBalancingSampler | None:
    if not args.batch_balancing:
        return None
    tuning_params = parse_tuning_params(
        "patch", args.training_method, args.tuning_params
    )
    sampler_power = float(tuning_params.get("sampler_power", 1.0))
    return BatchBalancingSampler(
        dataset.get_int_targets(),
        dataset.get_n_classes(),
        seed=args.seed,
        sampler_power=sampler_power,
    )


def _fit_model(
    args: argparse.Namespace, context: _TrainContext
) -> nn.Module | SKLearnModel:
    if args.model == "mlp":
        return build_mlp(
            args,
            context.train_ds,
            context.device,
            context.train_loader,
            context.val_loader,
        )
    return build_sklearn_model(args, context.train_ds)


def _sklearn_args(args: argparse.Namespace) -> dict[str, int]:
    aliases = SKLearnModel.get_argument_aliases()
    return {
        arg: vars(args)[aliases[arg]]
        for arg in SKLearnModel.get_required_arguments_per_model(args.model)
    }


def _sklearn_features(
    args: argparse.Namespace, dataset: TCGAUTDatasetImbalanced
) -> np.ndarray:
    if args.preload_features:
        features = dataset.dataset["features"]
    else:
        loaded = dataset.load_features(
            dataset.dataset["slide_id"].to_list(),
            dataset.dataset["cancer_type"].to_list(),
        )
        features = pd.DataFrame(loaded)["features"]
    return np.array(
        features.apply(lambda feature: feature.squeeze().cpu().tolist()).to_list()
    )


def evaluate_and_save(
    args: argparse.Namespace,
    context: _TrainContext,
    model: nn.Module | SKLearnModel,
    output_path: str,
) -> None:
    """Evaluate on validation and test splits and persist artifacts."""
    results: dict[str, dict[str, object]] = {}
    for split, dataset, loader in _evaluation_splits(context):
        class_order = validation_class_order(args, context.train_ds, dataset)
        result = test_model(
            model, loader, model_type=args.model, class_order=class_order
        )
        logger.info(
            "%s accuracy=%s, balanced accuracy=%s, macro F1=%s",
            split.capitalize(),
            result["accuracy"],
            result["balanced_accuracy"],
            result["macro_f1"],
        )
        if split == "validation" and args.visualize:
            save_validation_plot(output_path, dataset, class_order, result)
        results[split] = result
    os.makedirs(output_path, exist_ok=True)
    _save_model(model, output_path)
    for split, result in results.items():
        with open(os.path.join(output_path, f"{split}_results.json"), "w") as file:
            json.dump(_json_ready(result), file)
    with open(os.path.join(output_path, "args.json"), "w") as file:
        json.dump(vars(args), file)


def _evaluation_splits(
    context: _TrainContext,
) -> list[tuple[str, TCGAUTDatasetImbalanced, DataLoader]]:
    splits: list[tuple[str, TCGAUTDatasetImbalanced, DataLoader]] = []
    if context.val_ds is not None and context.val_loader is not None:
        splits.append(("validation", context.val_ds, context.val_loader))
    if context.test_ds is not None and context.test_loader is not None:
        splits.append(("test", context.test_ds, context.test_loader))
    return splits


def _save_model(model: nn.Module | SKLearnModel, output_path: str) -> None:
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), os.path.join(output_path, "model.pt"))
        return
    with open(os.path.join(output_path, "model.pkl"), "wb") as file:
        pickle.dump(model.model, file)


def _json_ready(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }


def _synthetic_variant_epochs(args: argparse.Namespace, train: bool) -> int | None:
    if args.training_method != "patch_feature_progan_aug" or not train:
        return None
    params = parse_tuning_params("patch", args.training_method, args.tuning_params)
    if "final_depth_epochs" not in params:
        return None
    return int(params["final_depth_epochs"])
