from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pandas as pd
from PIL import Image
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers.mlp import SwiGLUPacked
import torch

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
    rows = synthetic.reset_index(drop=True)
    features: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start : start + batch_size]
        images = _image_batch(batch, transforms, device)
        features.extend(_embed_batch(model, images, dtype, device))
    return features


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


def _image_batch(
    batch: pd.DataFrame,
    transforms: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    images = torch.stack(
        [
            cast(torch.Tensor, transforms(Image.open(path).convert("RGB")))
            for path in batch["image_path"].astype(str)
        ]
    )
    return images.to(device)


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
