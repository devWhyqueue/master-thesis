from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
import timm
import torch
from PIL import Image
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader, Dataset

from imbalance_benchmark.datasets.features.cache import (
    load_feature_row,
    load_slide_features,
)
from imbalance_benchmark.datasets.feature_provenance import (
    FEATURE_DIM,
    VIRCHOW2_MODEL,
    VIRCHOW2_REVISION,
    VIRCHOW2_WEIGHTS_SHA256,
    patch_sort_key,
    record_cached_slide,
    resolve_feature_snapshot,
    resolve_feature_provenance,
    validate_cached_slide,
    validate_feature_cache,
)

logger = logging.getLogger(__name__)

__all__ = [
    "load_feature_model",
    "embed_image_batch",
    "extract_slide_features",
    "attach_extracted_features",
    "load_slide_features",
    "load_feature_row",
    "patch_sort_key",
]


def load_feature_model(
    model_name: str,
    device: torch.device,
    revision: str = VIRCHOW2_REVISION,
    weights_sha256: str = VIRCHOW2_WEIGHTS_SHA256,
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    """Load the frozen Virchow2 feature encoder and its input transform."""
    snapshot = resolve_feature_snapshot(
        {
            "model_name": model_name,
            "revision": revision,
            "weights_sha256": weights_sha256,
        }
    )
    model = timm.create_model(
        f"local-dir:{snapshot.as_posix()}",
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


def _virchow2_pool(output: torch.Tensor) -> torch.Tensor:
    """Concatenate the CLS token and mean patch token into a 2560-d feature."""
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1).float().cpu()


def embed_image_batch(
    model: torch.nn.Module, images: torch.Tensor, dtype: str, device: torch.device
) -> list[torch.Tensor]:
    """Embed a batch of images into per-image Virchow2 (CLS + mean-patch) features."""
    amp_dtype = torch.float16 if dtype == "float16" else torch.float32
    with (
        torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=device.type != "cpu"
        ),
        torch.inference_mode(),
    ):
        output = model(images)
    features = _virchow2_pool(output)
    return [features[index] for index in range(len(features))]


class _ImagePathDataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self, paths: list[str], transforms: Callable[[Image.Image], torch.Tensor]
    ) -> None:
        self._paths = paths
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._transforms(Image.open(self._paths[index]).convert("RGB"))


def extract_slide_features(
    image_paths: list[str],
    model_name: str = VIRCHOW2_MODEL,
    batch_size: int = 64,
    dtype: str = "float16",
    device: torch.device | None = None,
    revision: str = VIRCHOW2_REVISION,
    weights_sha256: str = VIRCHOW2_WEIGHTS_SHA256,
) -> torch.Tensor:
    """Embed a slide's ordered patch images into a stacked (n_patches, 2560) tensor."""
    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if not image_paths:
        return torch.empty((0, FEATURE_DIM))
    model, transforms = load_feature_model(
        model_name, resolved_device, revision, weights_sha256
    )
    loader = DataLoader(
        _ImagePathDataset(image_paths, transforms),
        batch_size=batch_size,
        num_workers=0,
    )
    rows: list[torch.Tensor] = []
    for images in loader:
        rows.extend(
            embed_image_batch(model, images.to(resolved_device), dtype, resolved_device)
        )
    return torch.stack(rows)


def attach_extracted_features(
    frame: pd.DataFrame,
    feature_root: Path,
    model_name: str = "hf-hub:paige-ai/Virchow2",
    batch_size: int = 64,
    dtype: str = "float16",
    device: torch.device | None = None,
    revision: str = VIRCHOW2_REVISION,
    weights_sha256: str = VIRCHOW2_WEIGHTS_SHA256,
) -> pd.DataFrame:
    """Extract one stacked per-slide feature tensor per slide and attach references.

    Rows must already be in the deterministic per-slide patch order fixed by the
    dataset adapter. Existing ``<feature_root>/<slide_id>.pt`` files are reused
    rather than re-extracted.
    """
    options = {
        "model_name": model_name,
        "batch_size": batch_size,
        "dtype": dtype,
        "device": device,
        "revision": revision,
        "weights_sha256": weights_sha256,
    }
    _prepare_feature_cache(feature_root, options)
    enriched = frame.copy()
    feature_paths, feature_indices = _feature_references(
        enriched, feature_root, options
    )
    enriched["feature_path"] = feature_paths
    enriched["feature_index"] = feature_indices.astype(int)
    return enriched


def _prepare_feature_cache(feature_root: Path, options: dict[str, Any]) -> None:
    feature_root.mkdir(parents=True, exist_ok=True)
    provenance = resolve_feature_provenance(options)
    validate_feature_cache(feature_root, provenance)


def _feature_references(
    frame: pd.DataFrame, feature_root: Path, options: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    feature_paths = pd.Series(index=frame.index, dtype=object)
    feature_indices = pd.Series(index=frame.index, dtype=object)
    for slide_id, group in frame.groupby("slide_id", sort=False):
        slide_path = _ensure_slide_features(group, feature_root, str(slide_id), options)
        feature_paths.loc[group.index] = str(slide_path)
        feature_indices.loc[group.index] = range(len(group))
    return feature_paths, feature_indices


def _ensure_slide_features(
    group: pd.DataFrame,
    feature_root: Path,
    slide_id: str,
    options: dict[str, Any],
) -> Path:
    slide_path = feature_root / f"{slide_id}.pt"
    identities = _ordered_patch_identity(group)
    if slide_path.exists():
        validate_cached_slide(
            feature_root,
            slide_id,
            slide_path,
            identities,
            len(load_slide_features(str(slide_path))),
        )
        return slide_path
    image_paths = group["image_path"].astype(str).tolist()
    tensor = extract_slide_features(
        image_paths,
        str(options["model_name"]),
        int(options["batch_size"]),
        str(options["dtype"]),
        cast(torch.device | None, options["device"]),
        str(options["revision"]),
        str(options["weights_sha256"]),
    )
    torch.save(tensor, slide_path)
    record_cached_slide(feature_root, slide_id, slide_path, identities, len(tensor))
    return slide_path


def _ordered_patch_identity(group: pd.DataFrame) -> list[str]:
    """Return the exact patch identity sequence tied to cached tensor rows."""
    patch_ids = (
        group["patch_id"].astype(str)
        if "patch_id" in group
        else group["image_path"].astype(str)
    )
    return [
        f"{patch_id}\0{image_path}"
        for patch_id, image_path in zip(
            patch_ids, group["image_path"].astype(str), strict=True
        )
    ]
