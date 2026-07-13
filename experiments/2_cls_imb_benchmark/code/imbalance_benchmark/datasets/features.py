from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import cast

import pandas as pd
import timm
import torch
from PIL import Image
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

__all__ = [
    "load_feature_model",
    "embed_image_batch",
    "extract_slide_features",
    "attach_extracted_features",
    "SlideFeatureStore",
    "load_slide_features",
    "load_feature_row",
    "patch_sort_key",
]

FEATURE_DIM = 2560


def patch_sort_key(item: str) -> tuple[int, int]:
    """Sort patch identifiers by region and patch index."""
    region, index = item.split("_")[:2]
    return int(region), int(index)


def load_feature_model(
    model_name: str, device: torch.device
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor]]:
    """Load the frozen Virchow2 feature encoder and its input transform."""
    model = timm.create_model(
        model_name, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
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
    model_name: str = "hf-hub:paige-ai/Virchow2",
    batch_size: int = 64,
    dtype: str = "float16",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Embed a slide's ordered patch images into a stacked (n_patches, 2560) tensor."""
    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if not image_paths:
        return torch.empty((0, FEATURE_DIM))
    model, transforms = load_feature_model(model_name, resolved_device)
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
) -> pd.DataFrame:
    """Extract one stacked per-slide feature tensor per slide and attach references.

    Rows must already be in the deterministic per-slide patch order fixed by the
    dataset adapter. Existing ``<feature_root>/<slide_id>.pt`` files are reused
    rather than re-extracted.
    """
    feature_root.mkdir(parents=True, exist_ok=True)
    enriched = frame.copy()
    feature_paths = pd.Series(index=enriched.index, dtype=object)
    feature_indices = pd.Series(index=enriched.index, dtype=object)
    for slide_id, group in enriched.groupby("slide_id", sort=False):
        slide_path = feature_root / f"{slide_id}.pt"
        if not slide_path.exists():
            image_paths = group["image_path"].astype(str).tolist()
            tensor = extract_slide_features(
                image_paths, model_name, batch_size, dtype, device
            )
            torch.save(tensor, slide_path)
        feature_paths.loc[group.index] = str(slide_path)
        feature_indices.loc[group.index] = range(len(group))
    enriched["feature_path"] = feature_paths
    enriched["feature_index"] = feature_indices.astype(int)
    return enriched


@lru_cache(maxsize=512)
def load_slide_features(path: str) -> torch.Tensor:
    """Load a feature tensor and normalize to (n_instances, dim)."""
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(tensor, dict):
        cls_tok, mean_tok = tensor.get("cls"), tensor.get("mean_patch")
        features = (
            torch.cat([cls_tok, mean_tok], dim=-1).float()
            if cls_tok is not None and mean_tok is not None
            else next(
                value for value in tensor.values() if torch.is_tensor(value)
            ).float()
        )
    else:
        features = tensor.float()
    if features.ndim == 1:
        return features.unsqueeze(0)
    if features.ndim > 2:
        return features.reshape(-1, features.shape[-1])
    return features


def load_feature_row(path: str, index: int | None = None) -> torch.Tensor:
    """Load one feature vector; a multi-row tensor requires an explicit index."""
    features = load_slide_features(path)
    if index is not None:
        return features[int(index)].squeeze()
    if features.shape[0] == 1:
        return features[0].squeeze()
    raise ValueError(
        f"Feature file {path} has {features.shape[0]} rows; "
        "provide feature_index for multi-row tensors."
    )


class SlideFeatureStore:
    """Index chunked per-slide feature tensors and resolve manifest patches to rows."""

    def __init__(
        self,
        feature_dir: str,
        suffix_pattern: str = r"_[0-9]+$",
        patches_per_chunk: int = 30,
    ) -> None:
        self.feature_dir = Path(feature_dir)
        if not self.feature_dir.is_dir():
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
        self.patches_per_chunk = patches_per_chunk
        self._chunks_by_slide = _index_slide_chunks(self.feature_dir, suffix_pattern)

    def patch_index(self, patch_ids: list[str], patch_id: str) -> int:
        """Return the row index for one patch after deterministic sorting."""
        return _ordered_patch_indices(tuple(patch_ids))[patch_id]

    def resolve_patch(
        self, slide_id: str, patch_ids: list[str], patch_id: str
    ) -> tuple[str, int]:
        """Return the chunk path and row index for one manifest patch."""
        patch_index = self.patch_index(patch_ids, patch_id)
        chunk_index = patch_index // self.patches_per_chunk
        row_index = patch_index % self.patches_per_chunk
        chunk_path = self._chunks_by_slide[slide_id][chunk_index]
        return str(chunk_path), row_index

    def load_patch_feature(
        self, slide_id: str, patch_ids: list[str], patch_id: str
    ) -> torch.Tensor:
        """Load one patch embedding from the matching chunk tensor."""
        path, index = self.resolve_patch(slide_id, patch_ids, patch_id)
        return load_feature_row(path, index)


@lru_cache(maxsize=512)
def _ordered_patch_indices(patch_ids: tuple[str, ...]) -> dict[str, int]:
    ordered = sorted(patch_ids, key=patch_sort_key)
    return {patch: index for index, patch in enumerate(ordered)}


def _index_slide_chunks(
    feature_dir: Path, suffix_pattern: str
) -> dict[str, list[Path]]:
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(feature_dir.glob("*.pt")):
        stem = path.stem
        slide_id = re.sub(f"{suffix_pattern}$", "", stem)
        suffix = stem[len(slide_id) :]
        chunk_index = int(suffix.lstrip("_")) if suffix else 0
        grouped[slide_id].append((chunk_index, path))
    if not grouped:
        raise RuntimeError(f"No .pt features found under {feature_dir}")
    return {
        slide_id: [path for _, path in sorted(items, key=lambda item: item[0])]
        for slide_id, items in grouped.items()
    }
