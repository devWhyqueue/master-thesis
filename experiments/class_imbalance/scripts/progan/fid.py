from __future__ import annotations

from pathlib import Path
from importlib import import_module
from typing import Any, Callable
from urllib.error import URLError

import numpy as np
import torch
from PIL import Image
from scipy import linalg


def fid_for_paths(
    real_paths: list[Path], generated_paths: list[Path], device: torch.device
) -> float | None:
    """Compute Inception FID from real and generated image paths."""
    if len(real_paths) < 2 or len(generated_paths) < 2:
        return None
    models = _torchvision_models()
    if models is None:
        return None
    weights = models.Inception_V3_Weights.DEFAULT
    try:
        model = models.inception_v3(weights=weights).to(device).eval()
    except (OSError, RuntimeError, URLError, ValueError):
        return None
    setattr(model, "fc", torch.nn.Identity())
    preprocess = weights.transforms()
    real = _activations(real_paths, model, preprocess, device)
    generated = _activations(generated_paths, model, preprocess, device)
    return _frechet_distance(real, generated)


def _activations(
    paths: list[Path],
    model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.no_grad():
        for path in paths:
            image = Image.open(path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)
            values.append(model(tensor).cpu().numpy()[0])
    return np.stack(values)


def _frechet_distance(real: np.ndarray, generated: np.ndarray) -> float:
    mu_real, mu_generated = real.mean(axis=0), generated.mean(axis=0)
    cov_real, cov_generated = (
        np.cov(real, rowvar=False),
        np.cov(generated, rowvar=False),
    )
    covariance_root = np.asarray(linalg.sqrtm(cov_real @ cov_generated))
    covariance_root = np.real(covariance_root)
    mean_term = np.sum((mu_real - mu_generated) ** 2)
    trace_term = np.trace(cov_real + cov_generated - 2.0 * covariance_root)
    return float(mean_term + trace_term)


def _torchvision_models() -> Any | None:
    try:
        return import_module("torchvision.models")
    except ModuleNotFoundError:
        return None
