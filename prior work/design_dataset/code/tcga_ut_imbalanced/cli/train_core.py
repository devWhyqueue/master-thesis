import argparse
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tcga_ut_imbalanced.data.dataset import TCGAUTDatasetImbalanced
from tcga_ut_imbalanced.evaluation.metrics import test_model
from tcga_ut_imbalanced.cli.train_support import (
    build_mlp,
    make_dataloader,
    save_validation_plot,
    validation_class_order,
)
from tcga_ut_imbalanced.models.sklearn import SKLearnModel
from tcga_ut_imbalanced.training.batch_sampler import BatchBalancingSampler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TrainContext:
    device: torch.device
    train_ds: TCGAUTDatasetImbalanced
    val_ds: TCGAUTDatasetImbalanced | None
    train_loader: DataLoader
    val_loader: DataLoader | None


def run_training(args: argparse.Namespace) -> None:
    """Run model training, validation, and persistence."""
    context = _build_context(args)
    model = _fit_model(args, context)
    logger.info("Finished model training")
    base_path = _base_path(args)
    result = _validate(args, context, model, base_path)
    _save_outputs(args, model, result, base_path)
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
    train_ds = _dataset(args.dataset_structure_path, args, device)
    val_ds = (
        _dataset(args.validation_dataset_structure_path, args, device)
        if args.validation_dataset_structure_path
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
    return _TrainContext(device, train_ds, val_ds, train_loader, val_loader)


def _dataset(
    path: str, args: argparse.Namespace, device: torch.device
) -> TCGAUTDatasetImbalanced:
    return TCGAUTDatasetImbalanced(
        path, args.feature_path, None, args.args_path, args.preload_features, device
    )


def _sampler(
    args: argparse.Namespace, dataset: TCGAUTDatasetImbalanced
) -> BatchBalancingSampler | None:
    if not args.batch_balancing:
        return None
    return BatchBalancingSampler(
        dataset.get_int_targets(), dataset.get_n_classes(), seed=args.seed
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


def _base_path(args: argparse.Namespace) -> str:
    timestamp = time.time_ns() // 1_000_000
    return (
        os.path.join(args.results_save_path, str(timestamp))
        if args.store_timestamp
        else args.results_save_path
    )


def _validate(
    args: argparse.Namespace,
    context: _TrainContext,
    model: nn.Module | SKLearnModel,
    base_path: str,
) -> dict[str, object] | None:
    if context.val_ds is None or context.val_loader is None:
        return None
    class_order = validation_class_order(args, context.train_ds, context.val_ds)
    result = test_model(
        model, context.val_loader, model_type=args.model, class_order=class_order
    )
    logger.info(
        "Validation accuracy=%s, balanced accuracy=%s",
        result["accuracy"],
        result["balanced_accuracy"],
    )
    if args.visualize:
        save_validation_plot(base_path, context.val_ds, class_order, result)
    return result


def _save_outputs(
    args: argparse.Namespace,
    model: nn.Module | SKLearnModel,
    result: dict[str, object] | None,
    base_path: str,
) -> None:
    logger.info("Saving...")
    os.makedirs(base_path, exist_ok=True)
    _save_model(args, model, base_path)
    if result is not None:
        _save_json(
            os.path.join(base_path, "validation_results.json"), _json_ready(result)
        )
    _save_json(os.path.join(base_path, "args.json"), vars(args))


def _save_model(
    args: argparse.Namespace, model: nn.Module | SKLearnModel, base_path: str
) -> None:
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), os.path.join(base_path, "model.pt"))
        return
    with open(os.path.join(base_path, "model.pkl"), "wb") as file:
        pickle.dump(model.model, file)


def _save_json(path: str, data: dict[str, object]) -> None:
    with open(path, "w") as file:
        json.dump(data, file)


def _json_ready(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }
