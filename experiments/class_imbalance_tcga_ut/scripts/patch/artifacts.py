from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.common import write_json
from scripts.metadata import benchmark_metadata
from scripts.patch.data import PatchImageDataset
from scripts.patch.models import PatchClassifier
from scripts.training.support import _metric_payload


def seed_patch_run(seed: int) -> dict[str, Any]:
    """Seed patch benchmark randomness and return the recorded policy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch": True,
        "torch_cuda": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def write_patch_config(
    result_dir: Path,
    method: str,
    seed: int,
    class_names: list[str],
    deterministic: dict[str, Any],
) -> None:
    """Write reproducibility metadata for a patch result."""
    write_json(
        result_dir / "config.json",
        {
            "benchmark": "patch",
            "class_names": class_names,
            "deterministic": deterministic,
            "method": method,
            "method_metadata": benchmark_metadata("patch", method),
            "seed": seed,
        },
    )


def save_patch_checkpoint(
    result_dir: Path,
    method: str,
    seed: int,
    model: PatchClassifier,
    class_names: list[str],
) -> None:
    """Save enough patch state to reproduce evaluation logits."""
    payload = _checkpoint_payload(method, seed, model, class_names)
    torch.save(payload, result_dir / "checkpoint.pt")
    torch.save(model.state_dict(), result_dir / "model.pt")


def _checkpoint_payload(
    method: str,
    seed: int,
    model: PatchClassifier,
    class_names: list[str],
    epoch: int | None = None,
    epochs: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "benchmark": "patch",
        "class_names": class_names,
        "method": method,
        "method_metadata": benchmark_metadata("patch", method),
        "model_state_dict": model.state_dict(),
        "seed": seed,
    }
    if epoch is not None:
        payload["epoch"] = epoch
    if epochs is not None:
        payload["epochs"] = epochs
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def save_training_checkpoint(
    result_dir: Path,
    method: str,
    seed: int,
    model: PatchClassifier,
    optimizer: torch.optim.Optimizer,
    class_names: list[str],
    epoch: int,
    epochs: int,
) -> None:
    """Save resumable training state after each epoch."""
    payload = _checkpoint_payload(
        method, seed, model, class_names, epoch, epochs, optimizer
    )
    torch.save(payload, result_dir / "checkpoint_latest.pt")


def load_training_checkpoint(
    path: Path,
    model: PatchClassifier,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, list[str]]:
    """Restore model, optimizer, and the next epoch index from a training checkpoint."""
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    class_names = list(payload["class_names"])
    next_epoch = int(payload["epoch"]) + 1
    return next_epoch, class_names


def load_patch_checkpoint(
    path: Path,
    model: PatchClassifier,
    device: torch.device,
) -> list[str]:
    """Restore model weights for evaluation-only runs."""
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return list(payload["class_names"])


def evaluate_patch_dataset(
    model: PatchClassifier,
    dataset: PatchImageDataset,
    class_names: list[str],
    device: torch.device,
) -> dict[str, object]:
    """Evaluate a patch classifier on one split."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            logits, _ = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
    return _metric_payload(y_true, y_pred, probabilities, class_names)


def copy_synthetic_artifacts(manifest_path: Path, result_dir: Path) -> None:
    """Copy ProGAN manifest and summary into the method result directory."""
    summary_path = manifest_path.parent / "synthetic_patch_summary.json"
    shutil.copy2(manifest_path, result_dir / manifest_path.name)
    if summary_path.exists():
        shutil.copy2(summary_path, result_dir / summary_path.name)
