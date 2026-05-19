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
from scripts.patch.losses import gaussian_affinity
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
    prototypes: torch.Tensor | None,
    cfal_settings: dict[str, float] | None,
) -> None:
    """Save enough patch state to reproduce evaluation logits."""
    payload: dict[str, Any] = {
        "benchmark": "patch",
        "class_names": class_names,
        "method": method,
        "method_metadata": benchmark_metadata("patch", method),
        "model_state_dict": model.state_dict(),
        "seed": seed,
    }
    if prototypes is not None:
        payload["prototypes"] = prototypes.detach().cpu()
        payload["cfal_settings"] = cfal_settings or {}
    torch.save(payload, result_dir / "checkpoint.pt")
    torch.save(model.state_dict(), result_dir / "model.pt")


def cfal_checkpoint_settings(settings: dict[str, Any]) -> dict[str, float]:
    """Return CFAL hyperparameters stored with patch checkpoints."""
    return {
        "beta": float(settings["cfal_beta"]),
        "gamma": float(settings["cfal_gamma"]),
        "margin": float(settings["cfal_margin"]),
        "sigma": float(settings["cfal_sigma"]),
    }


def evaluate_patch_dataset(
    model: PatchClassifier,
    dataset: PatchImageDataset,
    class_names: list[str],
    device: torch.device,
    prototypes: torch.Tensor | None = None,
    cfal_sigma: float | None = None,
) -> dict[str, object]:
    """Evaluate a patch classifier on one split."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            logits, embeddings = model(images.to(device))
            if prototypes is not None and cfal_sigma is not None:
                logits = gaussian_affinity(embeddings, prototypes, cfal_sigma)
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
