from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, cast

import timm  # type: ignore[import-untyped]
import pandas as pd
import torch
from huggingface_hub import login  # type: ignore[import-untyped]
from huggingface_hub.errors import GatedRepoError  # type: ignore[import-not-found]
from PIL import Image
from timm.data import resolve_data_config  # type: ignore[import-untyped]
from timm.data.transforms_factory import create_transform  # type: ignore[import-untyped]
from timm.layers import SwiGLUPacked  # type: ignore[import-untyped]

from scripts.common import (
    ensure_dirs,
    load_config,
    output_root,
    write_json,
    write_progress,
)
from scripts.training.support import _resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Virchow2 feature extraction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _input_manifest(config: dict, seed: int) -> Path:
    return (
        output_root(config)
        / "outputs"
        / "synthetic_images"
        / f"seed={seed}"
        / ("synthetic_image_manifest.csv")
    )


def _output_root(config: dict, seed: int) -> Path:
    return output_root(config) / "outputs" / "synthetic_features" / f"seed={seed}"


def _login_if_token_available() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return

    login(token=token, add_to_git_credential=False)


def _load_model(device: torch.device) -> tuple[Any, Any]:
    _login_if_token_available()
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval().to(device)
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    return model, transform


def _read_input_frame(
    config: dict, seed: int, batch_size: int, smoke: bool
) -> pd.DataFrame:
    input_manifest = _input_manifest(config, seed)
    frame = pd.read_csv(input_manifest)
    if smoke:
        return frame.head(min(len(frame), batch_size))
    return frame


def _write_outputs(
    output_dir: Path,
    seed: int,
    input_manifest: Path,
    rows: list[dict[str, object]],
) -> None:
    manifest_path = output_dir / "synthetic_feature_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    write_json(
        output_dir / "synthetic_feature_config.json",
        {
            "seed": seed,
            "n_features": len(rows),
            "input_manifest": str(input_manifest),
            "output_manifest": str(manifest_path),
            "model": "paige-ai/Virchow2",
        },
    )
    write_progress(
        output_dir / "progress.json",
        {"seed": seed, "status": "completed", "n_features": len(rows)},
    )


def _write_gated_repo_failure(
    output_dir: Path, seed: int, error: GatedRepoError
) -> None:
    write_progress(
        output_dir / "progress.json",
        {
            "seed": seed,
            "status": "failed",
            "reason": "huggingface_gated_repo",
            "model": "paige-ai/Virchow2",
            "message": str(error).splitlines()[0],
        },
    )


def _load_encoder(output_dir: Path, seed: int, device: torch.device) -> tuple[Any, Any]:
    try:
        return _load_model(device)
    except GatedRepoError as error:
        _write_gated_repo_failure(output_dir, seed, error)
        raise


def _run_encoding(args: argparse.Namespace) -> tuple[Path, int]:
    config = load_config(args.config)
    ensure_dirs(config)
    input_manifest = _input_manifest(config, args.seed)
    output_dir = _output_root(config, args.seed)
    frame = _read_input_frame(config, args.seed, args.batch_size, args.smoke)
    write_progress(
        output_dir / "progress.json", {"seed": args.seed, "status": "started"}
    )
    device = _resolve_device(config["training"]["device"])
    model, transform = _load_encoder(output_dir, args.seed, device)
    rows = _write_feature_rows(
        frame, model, transform, output_dir, args.batch_size, device
    )
    _write_outputs(output_dir, args.seed, input_manifest, rows)
    return output_dir, len(rows)


def _encode_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch = batch.to(device)
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, enabled=device.type == "cuda"),
    ):
        output = model(batch)
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1).float().cpu()


def _load_images(paths: list[Path], transform: Any) -> torch.Tensor:
    images = [transform(Image.open(path).convert("RGB")) for path in paths]
    return torch.stack(cast(list[torch.Tensor], images))


def _write_feature_rows(
    frame: pd.DataFrame,
    model: torch.nn.Module,
    transform: Any,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    records = frame.to_dict("records")
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        image_paths = [Path(str(row["image_path"])) for row in chunk]
        embeddings = _encode_batch(model, _load_images(image_paths, transform), device)
        for row, embedding in zip(chunk, embeddings, strict=True):
            feature_id = Path(str(row["image_path"])).stem
            feature_path = feature_dir / f"{feature_id}.pt"
            torch.save(embedding.unsqueeze(0), feature_path)
            rows.append(
                {
                    "feature_id": feature_id,
                    "slide_id": feature_id,
                    "cancer_type": str(row["cancer_type"]),
                    "feature_path": str(feature_path),
                    "source_image_path": str(row["image_path"]),
                }
            )
    return rows


def main() -> None:
    """Encode generated patch images with Virchow2 into feature tensors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    output_dir, n_rows = _run_encoding(parse_args())
    logger.info("Wrote %s synthetic features to %s", n_rows, output_dir)


if __name__ == "__main__":
    main()
