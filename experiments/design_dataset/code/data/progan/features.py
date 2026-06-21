from __future__ import annotations

import os
from collections.abc import Callable
from typing import cast

import pandas as pd
from PIL import Image
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers.mlp import SwiGLUPacked
import torch
from torch.utils.data import DataLoader, Dataset

from data.feature_store import load_feature_row


def build_feature_payload(
    manifest: pd.DataFrame,
    *,
    model_name: str,
    batch_size: int,
    dtype: str,
    device: torch.device,
) -> dict[str, object]:
    """Build a row-level feature cache covering real and synthetic rows."""
    real, synthetic = _split_real_and_synthetic(manifest)
    features = _real_features(real) + _synthetic_features(
        synthetic,
        model_name=model_name,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    return {
        "feature_paths": manifest["feature_path"].astype(str).tolist(),
        "feature_indices": manifest["feature_index"].astype(int).tolist(),
        "features": torch.stack(features) if features else torch.empty((0, 0)),
    }


def real_only_payload(manifest: pd.DataFrame) -> dict[str, object]:
    """Build a row-level cache payload for a real-only manifest."""
    features = _real_features(manifest)
    return {
        "feature_paths": manifest["feature_path"].astype(str).tolist(),
        "feature_indices": manifest["feature_index"].astype(int).tolist(),
        "features": torch.stack(features) if features else torch.empty((0, 0)),
    }


def _split_real_and_synthetic(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    synthetic_mask = manifest["is_synthetic"].astype(bool)
    real = manifest.loc[~synthetic_mask].copy()
    synthetic = manifest.loc[synthetic_mask].copy()
    return real, synthetic


def _real_features(real: pd.DataFrame) -> list[torch.Tensor]:
    return [
        load_feature_row(str(row["feature_path"]), int(row["feature_index"]))
        for _, row in real.iterrows()
    ]


def _synthetic_features(
    synthetic: pd.DataFrame,
    *,
    model_name: str,
    batch_size: int,
    dtype: str,
    device: torch.device,
) -> list[torch.Tensor]:
    if synthetic.empty:
        return []
    model, transforms = _load_feature_model(model_name, device)
    n_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "0"))
    loader = DataLoader(
        _ImageDataset(synthetic["image_path"].astype(str).tolist(), transforms),
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=device.type == "cuda" and n_workers > 0,
        persistent_workers=n_workers > 0,
    )
    features: list[torch.Tensor] = []
    for images in loader:
        features.extend(_embed_batch(model, images.to(device), dtype, device))
    return features


class _ImageDataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self,
        paths: list[str],
        transforms: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self._paths = paths
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._transforms(Image.open(self._paths[idx]).convert("RGB"))


def _load_feature_model(
    model_name: str, device: torch.device
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    model = timm.create_model(
        model_name,
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval().to(device)
    transforms = cast(
        Callable[[Image.Image], torch.Tensor],
        create_transform(**resolve_data_config(model.pretrained_cfg, model=model)),
    )
    return model, transforms


def _embed_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    dtype: str,
    device: torch.device,
) -> list[torch.Tensor]:
    amp_dtype = torch.float16 if dtype == "float16" else torch.float32
    with _autocast(device, amp_dtype), torch.inference_mode():
        output = model(images)
    features = _virchow2_features(output)
    return [features[index] for index in range(len(features))]


def _autocast(device: torch.device, dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type != "cpu",
    )


def _virchow2_features(output: torch.Tensor) -> torch.Tensor:
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1).float().cpu()
