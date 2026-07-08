"""Prepare native CAMELYON16 manifests from pre-tiled 20x patches and masks.

Two regimes share the same per-slide Virchow2 feature files but use different
labels: the WSI-bag regime labels each slide bag with its slide-level
tumor/normal label; the patch regime labels each tile from the pixel mask
(severe native imbalance). The WSI manifest caps each slide at the median
available-tiles-per-slide (the data-driven bag budget M); the patch pool is
uniformly subsampled per split to a bounded budget, preserving the native
tumor/normal ratio.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.camelyon16.masks import load_mask, patch_labels
from data.camelyon16.metadata import (
    LABELS,
    NON_EXHAUSTIVE_TUMOR,
    list_slide_patches,
    load_slide_labels,
    slides_with_patches,
)
from data.camelyon16.splitting import SPLITS, assert_slide_disjoint, split_cases

MIN_NATIVE_IMBALANCE_RATIO = 5.0

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CAMELYON16 preparation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-patch-budget", type=int, default=100000)
    parser.add_argument("--eval-patch-budget", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    """Build CAMELYON16 native benchmark manifests."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    data_root = Path(args.data_root)
    output = Path(args.output_root)
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    slide_labels = load_slide_labels(data_root)
    slides = _usable_slides(data_root, slide_labels)
    bag_size = _median_available(data_root, slides)
    logger.info("CAMELYON16 bag size (median available tiles/slide) = %d", bag_size)
    frame = _build_patch_frame(data_root, slides, slide_labels, bag_size, output)
    for seed in args.seeds:
        _write_seed(frame, output / "manifests", seed, args)
    _write_report(output, frame, bag_size)


def _usable_slides(data_root: Path, slide_labels: dict[str, str]) -> list[str]:
    mask_dir = data_root / "masks"
    slides = [
        slide
        for slide in slides_with_patches(data_root)
        if slide in slide_labels and (mask_dir / f"{slide}_mask.npy").is_file()
    ]
    if not slides:
        raise RuntimeError("No usable CAMELYON16 slides found.")
    return slides


def _median_available(data_root: Path, slides: list[str]) -> int:
    counts = [len(list_slide_patches(data_root, slide)) for slide in slides]
    return int(np.median(counts))


def _build_patch_frame(
    data_root: Path,
    slides: list[str],
    slide_labels: dict[str, str],
    bag_size: int,
    output: Path,
) -> pd.DataFrame:
    feature_dir = output / "features" / "virchow2"
    parts = [
        _slide_rows(data_root, slide, slide_labels[slide], bag_size, feature_dir)
        for slide in slides
    ]
    return pd.concat(parts, ignore_index=True)


def _slide_rows(
    data_root: Path,
    slide_id: str,
    slide_label: str,
    bag_size: int,
    feature_dir: Path,
) -> pd.DataFrame:
    patches = list_slide_patches(data_root, slide_id)
    if len(patches) > bag_size:
        keep = np.linspace(0, len(patches) - 1, bag_size).astype(int)
        patches = [patches[index] for index in keep]
    patch_ids = [pid for pid, _ in patches]
    labels = patch_labels(load_mask(data_root, slide_id), patch_ids)
    exhaustive = slide_id not in NON_EXHAUSTIVE_TUMOR
    return pd.DataFrame(
        {
            "dataset": "camelyon16",
            "case_id": slide_id,
            "slide_id": slide_id,
            "patch_id": patch_ids,
            "slide_label": slide_label,
            "patch_label": labels,
            "image_path": [str(path) for _, path in patches],
            "feature_path": str(feature_dir / f"{slide_id}.pt"),
            "feature_index": range(len(patches)),
            "exhaustive": exhaustive,
        }
    )


def _write_seed(
    frame: pd.DataFrame, root: Path, seed: int, args: argparse.Namespace
) -> None:
    slide_frame = frame.drop_duplicates("case_id")[["case_id", "slide_label"]]
    assignment = split_cases(slide_frame, seed)
    tagged = frame.merge(assignment, on="case_id", how="inner")
    assert_slide_disjoint(tagged)
    seed_dir = root / f"native_seed={seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    _write_wsi_manifest(tagged, seed_dir)
    _write_patch_manifests(tagged, seed_dir, seed, args)
    _write_json(seed_dir / "class_order.json", list(LABELS))
    _write_json(seed_dir / "args.json", {"dataset": "camelyon16", "seed": seed})


def _write_wsi_manifest(tagged: pd.DataFrame, seed_dir: Path) -> None:
    wsi = tagged.copy()
    wsi["cancer_type"] = wsi["slide_label"]
    wsi.to_csv(seed_dir / "manifest_splits.csv", index=False)


def _write_patch_manifests(
    tagged: pd.DataFrame, seed_dir: Path, seed: int, args: argparse.Namespace
) -> None:
    patch_pool = tagged[tagged["exhaustive"]].copy()
    patch_pool["cancer_type"] = patch_pool["patch_label"]
    rng = np.random.default_rng(seed)
    budgets = {
        "train": args.train_patch_budget,
        "validation": args.eval_patch_budget,
        "test": args.eval_patch_budget,
    }
    parts = []
    for split in SPLITS:
        rows = patch_pool[patch_pool["split"] == split]
        rows = _subsample(rows, budgets[split], rng)
        rows.to_csv(seed_dir / f"{split}.csv", index=False)
        parts.append(rows)
    pd.concat(parts, ignore_index=True).to_csv(
        seed_dir / "patch_manifest.csv", index=False
    )
    counts = _label_counts(parts[0])
    _write_json(seed_dir / "target_counts.json", counts)
    _write_json(seed_dir / "available_counts.json", counts)


def _subsample(
    rows: pd.DataFrame, budget: int, rng: np.random.Generator
) -> pd.DataFrame:
    if len(rows) <= budget:
        return rows
    chosen = rng.choice(len(rows), size=budget, replace=False)
    return rows.iloc[np.sort(chosen)]


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["patch_label"].value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}


def _write_report(output: Path, frame: pd.DataFrame, bag_size: int) -> None:
    patch_frame = frame[frame["exhaustive"]]
    counts = {
        label: int((patch_frame["patch_label"] == label).sum()) for label in LABELS
    }
    ratio = float(counts["normal"] / max(1, counts["tumor"]))
    slide_counts = frame.drop_duplicates("slide_id")["slide_label"].value_counts()
    report = {
        "dataset": "CAMELYON16",
        "n_slides": int(frame["slide_id"].nunique()),
        "n_patch_slides": int(patch_frame["slide_id"].nunique()),
        "wsi_bag_size": bag_size,
        "n_patches_capped": int(len(frame)),
        "class_counts_patch": counts,
        "imbalance_ratio_patch": ratio,
        "slide_counts": {label: int(slide_counts.get(label, 0)) for label in LABELS},
        "min_native_imbalance_ratio": MIN_NATIVE_IMBALANCE_RATIO,
        "recommended_benchmark_mode": (
            "native" if ratio >= MIN_NATIVE_IMBALANCE_RATIO else "power_law"
        ),
        "labels": list(LABELS),
    }
    _write_json(output / "camelyon16_prepare_report.json", report)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
