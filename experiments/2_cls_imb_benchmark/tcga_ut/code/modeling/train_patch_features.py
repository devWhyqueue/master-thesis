from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from code.common import ensure_dirs, load_config, write_run_record
from code.metadata import benchmark_metadata
from code.modeling.patch.artifacts import seed_patch_run
from code.modeling.patch.losses import (
    PatchFocalLoss,
    ScholzCombinedLoss,
    inverse_frequency_weights,
)
from code.modeling.patch_feature.specialized_trainers import fit_special_patch_method
from code.modeling.patch_feature.training import (
    PatchFeatureDataset,
    build_patch_feature_model,
    evaluate_patch_feature_model,
    patch_feature_train_loader,
    select_patch_feature_rows,
    split_patch_feature_datasets,
)
from code.modeling.patch_feature.patch_feature_cache import patch_feature_cache_dir
from code.modeling.training.support import _resolve_device
from code.analysis.tuning.grid import validate_tuning_params
from code.analysis.tuning.paths import tuning_result_dir


def parse_args() -> argparse.Namespace:
    """Parse patch-feature training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tuning-id", default=None)
    parser.add_argument("--tuning-params", default=None)
    return parser.parse_args()


def main() -> None:
    """Train one patch-level method on frozen Virchow2 features."""
    args = parse_args()
    _run(args)


def _run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    tuning_params = _load_tuning_params(args.method, args.tuning_params)
    seed_patch_run(args.seed)
    paths = ensure_dirs(config)
    cache_dir = patch_feature_cache_dir(config, args.seed)
    frame = select_patch_feature_rows(
        pd.read_csv(cache_dir / "manifest.csv"),
        args.method,
        args.smoke,
        config,
        seed=args.seed,
        tuning_params=tuning_params,
    )
    class_names = sorted(frame["cancer_type"].unique().tolist())
    datasets = split_patch_feature_datasets(
        frame, cache_dir / "features.npy", class_names
    )
    model, diagnostics = _fit_model(
        args.method, datasets[0], class_names, config, args.seed, tuning_params
    )
    device = _resolve_device(str(config["patch_feature_training"]["device"]))
    result_dir = _result_dir(paths, args.method, args.seed, args.tuning_id)
    _write_outputs(
        result_dir,
        args,
        tuning_params,
        model,
        datasets,
        class_names,
        device,
        diagnostics,
    )


def _write_outputs(
    result_dir: Path,
    args: argparse.Namespace,
    tuning_params: dict[str, float],
    model: torch.nn.Module,
    datasets: tuple[PatchFeatureDataset, PatchFeatureDataset, PatchFeatureDataset],
    class_names: list[str],
    device: torch.device,
    diagnostics: dict[str, object] | None = None,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), result_dir / "model.pt")
    splits = {
        split_name: evaluate_patch_feature_model(model, dataset, class_names, device)
        for split_name, dataset in zip(("val", "test"), datasets[1:], strict=True)
    }
    write_run_record(
        result_dir,
        {
            "benchmark": "patch_feature",
            "method": args.method,
            "seed": args.seed,
            "smoke": args.smoke,
            "tuning_id": args.tuning_id,
            "tuning_params": tuning_params,
            "model_path": "model.pt",
            "method_metadata": benchmark_metadata("patch", args.method),
            "diagnostics": diagnostics,
            "splits": splits,
        },
    )


def _fit_model(
    method: str,
    train_set: PatchFeatureDataset,
    class_names: list[str],
    config: dict,
    seed: int,
    tuning_params: dict[str, float],
) -> tuple[torch.nn.Module, dict[str, object] | None]:
    settings = config["patch_feature_training"]
    device = _resolve_device(str(settings["device"]))
    special = fit_special_patch_method(
        method, train_set, class_names, settings, device, seed, tuning_params
    )
    if special is not None:
        return special
    labels = train_set.labels.cpu().numpy()
    model = build_patch_feature_model(train_set, settings, len(class_names), device)
    criterion = _criterion(
        method, labels, len(class_names), settings, device, tuning_params
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    _train(
        model,
        train_set,
        labels,
        criterion,
        optimizer,
        method,
        settings,
        device,
        seed,
        tuning_params,
    )
    return model, None


def _criterion(
    method: str,
    labels: np.ndarray,
    n_classes: int,
    settings: dict,
    device: torch.device,
    tuning_params: dict[str, float],
) -> torch.nn.Module:
    if method == "patch_feature_weighted_ce":
        return torch.nn.CrossEntropyLoss(
            weight=inverse_frequency_weights(
                labels, n_classes, float(tuning_params.get("weight_power", 1.0))
            ).to(device)
        )
    if method == "patch_feature_focal":
        gamma = _tuned_or_default(tuning_params, "focal_gamma", settings["focal_gamma"])
        return PatchFocalLoss(gamma)
    if method == "patch_feature_ce_soft_f1_balanced":
        return ScholzCombinedLoss(
            n_classes,
            "f1",
            float(tuning_params.get("metric_loss_weight", 1.0)),
        )
    if method == "patch_feature_ce_soft_mcc_balanced":
        return ScholzCombinedLoss(
            n_classes,
            "mcc",
            float(tuning_params.get("metric_loss_weight", 1.0)),
        )
    return torch.nn.CrossEntropyLoss()


def _train(
    model: torch.nn.Module,
    dataset: PatchFeatureDataset,
    labels: np.ndarray,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    method: str,
    settings: dict,
    device: torch.device,
    seed: int,
    tuning_params: dict[str, float],
) -> None:
    loader = patch_feature_train_loader(
        dataset,
        labels,
        method,
        int(settings["batch_size"]),
        seed,
        float(tuning_params.get("sampler_power", 1.0)),
    )
    for _ in range(1, int(settings["epochs"]) + 1):
        model.train()
        for features, targets in loader:
            loss = criterion(model(features.to(device)), targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _load_tuning_params(method: str, raw: str | None) -> dict[str, float]:
    if raw is None:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--tuning-params must be a JSON object")
    validate_tuning_params("patch_feature", method, payload)
    return {str(key): float(value) for key, value in payload.items()}


def _tuned_or_default(
    tuning_params: dict[str, float], key: str, default: object
) -> float:
    if key in tuning_params:
        return tuning_params[key]
    return float(cast(float | int | str, default))


def _result_dir(
    paths: dict[str, Path], method: str, seed: int, tuning_id: str | None
) -> Path:
    if tuning_id is None:
        return paths["results"] / "patch_feature" / method / f"seed={seed}"
    return tuning_result_dir(paths, "patch_feature", method, tuning_id, seed)


if __name__ == "__main__":
    main()
