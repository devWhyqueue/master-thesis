from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers.mlp import SwiGLUPacked

from code.common import ensure_dirs, load_config, write_json
from code.data.staging.patch import stage_patch_manifest
from code.modeling.training.support import _resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse patch-feature extraction arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def patch_feature_cache_dir(config: dict, seed: int) -> Path:
    """Return the feature-cache directory for one patch seed."""
    return ensure_dirs(config)["data"] / "patch_feature_cache" / f"seed={seed}"


def main() -> None:
    """Extract frozen Virchow2 embeddings for controlled patch manifests."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    _load_env_files(*_env_candidates(paths["root"]))
    manifest_path = stage_patch_manifest(
        config, args.seed, include_synthetic=args.include_synthetic
    )
    frame = pd.read_csv(manifest_path)
    if args.smoke:
        frame = _smoke_frame(frame)
    _extract_features(
        config,
        frame,
        patch_feature_cache_dir(config, args.seed),
        args.seed,
        args.include_synthetic,
    )


def _load_env_files(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            _load_env_file(path)


def _env_candidates(root: Path) -> list[Path]:
    candidates = [Path.home() / ".env"]
    candidates.extend(parent / ".env" for parent in [root, *root.parents])
    return candidates


def _load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            token = value.strip().strip("\"'")
            os.environ.setdefault(key.strip(), token)
            os.environ.setdefault("HF_TOKEN", token)


def _smoke_frame(frame: pd.DataFrame) -> pd.DataFrame:
    parts = [part.head(8) for _, part in frame.groupby("split", sort=False)]
    return pd.concat(parts, ignore_index=True)


def _extract_features(
    config: dict,
    frame: pd.DataFrame,
    output_dir: Path,
    seed: int,
    include_synthetic: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = config["patch_feature_extraction"]
    device = _resolve_device(str(settings["device"]))
    model, transforms = _load_model(str(settings["model_name"]), device)
    features = _write_feature_array(
        frame, output_dir, settings, model, transforms, device, seed
    )
    _write_feature_manifest(frame, output_dir, features, seed, include_synthetic)


def _write_feature_array(
    frame: pd.DataFrame,
    output_dir: Path,
    settings: dict,
    model: torch.nn.Module,
    transforms: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    seed: int,
) -> np.memmap:
    batch_size = int(settings["batch_size"])
    dtype = _feature_dtype(settings)
    features = None
    for start in range(0, len(frame), batch_size):
        rows = frame.iloc[start : start + batch_size]
        images = torch.stack(
            [
                cast(torch.Tensor, transforms(_load_image(path)))
                for path in rows["image_path"]
            ]
        )
        embeddings = _embed_batch(model, images.to(device), device).cpu().numpy()
        features = _ensure_feature_array(
            features, output_dir, len(frame), embeddings, dtype
        )
        features[start : start + len(rows)] = embeddings.astype(dtype, copy=False)
        _log_progress(seed, start + len(rows), len(frame), batch_size)
    if features is None:
        raise RuntimeError("Patch feature extraction received an empty manifest.")
    return features


def _feature_dtype(settings: dict) -> type[np.float16] | type[np.float32]:
    return (
        np.float16 if str(settings.get("dtype", "float16")) == "float16" else np.float32
    )


def _ensure_feature_array(
    features: np.memmap | None,
    output_dir: Path,
    row_count: int,
    embeddings: np.ndarray,
    dtype: type[np.float16] | type[np.float32],
) -> np.memmap:
    if features is not None:
        return features
    return np.lib.format.open_memmap(
        output_dir / "features.npy",
        mode="w+",
        dtype=dtype,
        shape=(row_count, embeddings.shape[1]),
    )


def _log_progress(seed: int, written: int, total: int, batch_size: int) -> None:
    if written % max(batch_size * 10, 1) == 0:
        logger.info("extracted seed=%s rows=%s/%s", seed, written, total)


def _write_feature_manifest(
    frame: pd.DataFrame,
    output_dir: Path,
    features: np.memmap,
    seed: int,
    include_synthetic: bool,
) -> None:
    manifest = frame.copy()
    manifest["feature_index"] = np.arange(len(manifest), dtype=np.int64)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    write_json(
        output_dir / "metadata.json",
        {
            "dtype": str(features.dtype),
            "feature_dim": int(features.shape[1]),
            "features_path": str(output_dir / "features.npy"),
            "include_synthetic": include_synthetic,
            "n_rows": int(len(frame)),
            "seed": seed,
        },
    )


def _load_model(
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


def _load_image(path: str) -> Image.Image:
    return Image.open(Path(path)).convert("RGB")


def _embed_batch(
    model: torch.nn.Module, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.float16),
    ):
        output = model(images)
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1).float()


if __name__ == "__main__":
    main()
