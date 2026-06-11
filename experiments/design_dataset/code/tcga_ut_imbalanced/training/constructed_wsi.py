import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from common_code.metrics.payload import resolve_device
from common_code.wsi import trainer as mil_trainer
from common_code.wsi.eval import _save_and_evaluate_bags
from common_code.wsi.rankmix_teacher import load_rankmix_teacher, train_rankmix_teacher
from tcga_ut_imbalanced.evaluation.tuning_params import parse_tuning_params
from tcga_ut_imbalanced.training.constructed_wsi_data import (
    ConstructedBagDataset,
    OPTIONAL_ARGS,
    write_training_outputs,
)

METHODS = (
    "mil_ce",
    "mil_weighted_ce",
    "mil_balanced_sampler_ce",
    "mil_focal",
    "rankmix_mil",
    "sc_mil",
    "mde_mil",
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--results-save-path", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--class-order-name", required=True)
    parser.add_argument("--parameter", type=float, required=True)
    parser.add_argument("--tuning-id", default=None)
    parser.add_argument("--tuning-params", default=None)
    parser.add_argument("--device", default="auto")
    for name, value_type, default in OPTIONAL_ARGS:
        parser.add_argument(name, type=value_type, default=default)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    output_dir = Path(args.results_save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_manifest(args.manifest_path)
    class_names = sorted(str(name) for name in frame["cancer_type"].unique().tolist())
    results, diagnostics = _train_method(
        args.method,
        frame,
        class_names,
        {"wsi_training": _training_config(args)},
        args.seed,
        output_dir,
    )
    write_training_outputs(output_dir, args, class_names, results, diagnostics)


def _load_manifest(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "feature_path", "cancer_type"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest requires columns {sorted(required)}.")
    frame = frame.copy()
    frame["split"] = frame["split"].replace({"validation": "val"})
    return frame


def _training_config(args: argparse.Namespace) -> dict[str, object]:
    tuning = parse_tuning_params("wsi", args.method, args.tuning_params)
    config = {
        "device": args.device,
        "epochs": args.epochs,
        "bag_batch_size": args.bag_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "focal_gamma": args.focal_gamma,
        "max_instances_per_bag": args.max_instances_per_bag,
        "rankmix_teacher_epochs": args.rankmix_teacher_epochs,
        "rankmix_alpha": args.rankmix_alpha,
        "sc_mil_temperature": args.sc_mil_temperature,
        "mde_mil_consistency_weight": args.mde_mil_consistency_weight,
        "sampler_power": args.sampler_power,
        "weight_power": args.weight_power,
        "max_bags_per_class": args.max_bags_per_class or None,
        "bag_cache_dir": args.bag_cache_dir or None,
    }
    config.update({key: tuning[key] for key in tuning})
    return config


def _train_method(
    method: str,
    frame: pd.DataFrame,
    class_names: list[str],
    config: dict[str, Any],
    seed: int,
    result_dir: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    torch.manual_seed(seed)
    training = config["wsi_training"]
    device = resolve_device(str(training["device"]))
    bag_cache_dir = cast(str, training.get("bag_cache_dir", "")) or None
    train_dataset, val_dataset, test_dataset = _split_datasets(
        frame,
        class_names,
        cast(int, training["max_instances_per_bag"]),
        cast(int | None, training["max_bags_per_class"]),
        bag_cache_dir,
    )
    labels = cast(np.ndarray, train_dataset.labels.cpu().numpy())
    build_model = (
        mil_trainer._build_mde_model
        if method == "mde_mil"
        else mil_trainer._build_model
    )
    model = build_model(train_dataset, class_names, training, device)
    optimizer = _optimizer(model, training)
    teacher = None
    if method == "rankmix_mil":
        teacher = load_rankmix_teacher(
            result_dir,
            mil_trainer._build_model,
            train_dataset,
            class_names,
            training,
            device,
            seed,
        )
        if teacher is None:
            teacher = train_rankmix_teacher(
                model,
                train_dataset,
                labels,
                training,
                optimizer,
                device,
                seed,
                result_dir,
                mil_trainer._loader,
            )
        model = mil_trainer._build_model(train_dataset, class_names, training, device)
        optimizer = _optimizer(model, training)
    diagnostics = mil_trainer._run_training(
        method,
        cast(nn.Module, model),
        train_dataset,
        labels,
        training,
        optimizer,
        device,
        seed,
        result_dir,
        teacher,
    )
    results = _save_and_evaluate_bags(
        model, val_dataset, test_dataset, class_names, device, result_dir
    )
    return results, diagnostics


def _optimizer(model: nn.Module, training: dict[str, object]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cast(float, training["learning_rate"])),
        weight_decay=float(cast(float, training["weight_decay"])),
    )


def _split_datasets(
    frame: pd.DataFrame,
    class_names: list[str],
    max_instances: int | None,
    max_bags_per_class: int | None,
    bag_cache_dir: str | None = None,
) -> tuple[ConstructedBagDataset, ConstructedBagDataset, ConstructedBagDataset]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    cache_dir = bag_cache_dir or None
    return tuple(
        ConstructedBagDataset(
            cast(pd.DataFrame, frame[frame["split"] == split]),
            class_to_idx,
            max_instances,
            max_bags_per_class,
            cache_dir,
            split,
        )
        for split in ("train", "val", "test")
    )  # type: ignore[return-value]


if __name__ == "__main__":
    main()
