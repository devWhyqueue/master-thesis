"""Extract frozen Virchow2 features for one CAMELYON16 slide (array task).

Features are seed-independent: the capped patch set and its ordering are the
same across split seeds, so extraction reads a single reference manifest and
writes one ``<slide>.pt`` file reused by every seed. One SLURM array task
handles one slide.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from data.bracs.features import extract_features

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse per-slide feature extraction arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--array-task-id", type=int, default=None)
    parser.add_argument("--model-name", default="hf-hub:paige-ai/Virchow2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Extract Virchow2 features for the slide at this array index."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    manifest = pd.read_csv(args.manifest_path)
    slides = sorted(manifest["slide_id"].unique().tolist())
    task_id = _resolve_task_id(args)
    if task_id >= len(slides):
        logger.info(
            "Array task %d exceeds %d slides; nothing to do.", task_id, len(slides)
        )
        return
    slide_id = str(slides[task_id])
    feature_path = Path(args.feature_dir) / f"{slide_id}.pt"
    if feature_path.exists():
        logger.info("Skipping existing slide features: %s", feature_path)
        return
    rows = manifest[manifest["slide_id"] == slide_id].sort_values("feature_index")
    _extract_slide(rows, feature_path, args, slide_id)


def _extract_slide(
    rows: pd.DataFrame, feature_path: Path, args: argparse.Namespace, slide_id: str
) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features = extract_features(
        rows["image_path"].astype(str).tolist(),
        str(args.model_name),
        int(args.batch_size),
        str(args.dtype),
        torch.device(str(args.device)),
    )
    torch.save(features, feature_path)
    logger.info("Wrote %d features for %s to %s", len(features), slide_id, feature_path)


def _resolve_task_id(args: argparse.Namespace) -> int:
    if args.array_task_id is not None:
        return args.array_task_id
    env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_value is None:
        raise ValueError("Pass --array-task-id or run in a SLURM array.")
    return int(env_value)


if __name__ == "__main__":
    main()
