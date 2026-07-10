"""Prepare native PANDA manifests from tiled 20x patches.

Two regimes share the per-slide Virchow2 features but use different labels: the
WSI-bag regime grades each biopsy by its 6-class ISUP label; the patch regime
uses the mask-derived binary cancer/benign label and only masked ("exhaustive")
slides. The WSI manifest caps each slide at the median available-tiles-per-slide
(the bag budget M); the patch pool is uniformly subsampled per split to a bounded
budget, preserving the native cancer/benign ratio.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.panda.metadata import PATCH_LABELS, WSI_LABELS
from data.panda.splitting import SPLITS, assert_slide_disjoint, split_cases

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse PANDA preparation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selection-path", required=True)
    parser.add_argument("--tiles-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-patch-budget", type=int, default=100000)
    parser.add_argument("--eval-patch-budget", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    """Build PANDA native benchmark manifests."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    output = Path(args.output_root)
    (output / "manifests").mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(args.selection_path)
    tiles = _load_tiles(selection, Path(args.tiles_dir))
    if not tiles:
        raise RuntimeError("No tiled PANDA slides found.")
    bag_size = int(np.median([len(frame) for frame in tiles.values()]))
    logger.info("PANDA bag size (median available tiles/slide) = %d", bag_size)
    frame = _build_patch_frame(selection, tiles, bag_size, output)
    for seed in args.seeds:
        _write_seed(frame, output / "manifests", seed, args)
    _write_report(output, frame, bag_size)


def _load_tiles(selection: pd.DataFrame, tiles_dir: Path) -> dict[str, pd.DataFrame]:
    tiles: dict[str, pd.DataFrame] = {}
    for slide_id in selection["slide_id"]:
        path = tiles_dir / f"{slide_id}.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            tiles[str(slide_id)] = frame
    return tiles


def _build_patch_frame(
    selection: pd.DataFrame,
    tiles: dict[str, pd.DataFrame],
    bag_size: int,
    output: Path,
) -> pd.DataFrame:
    feature_dir = output / "features" / "virchow2"
    parts = [
        _slide_rows(row, tiles[str(row["slide_id"])], bag_size, feature_dir)
        for _, row in selection.iterrows()
        if str(row["slide_id"]) in tiles
    ]
    return pd.concat(parts, ignore_index=True)


def _slide_rows(
    row: pd.Series, tiles: pd.DataFrame, bag_size: int, feature_dir: Path
) -> pd.DataFrame:
    if len(tiles) > bag_size:
        keep = np.linspace(0, len(tiles) - 1, bag_size).astype(int)
        tiles = tiles.iloc[keep]
    slide_id = str(row["slide_id"])
    return pd.DataFrame(
        {
            "dataset": "panda",
            "case_id": slide_id,
            "slide_id": slide_id,
            "patch_id": tiles["patch_id"].to_numpy(),
            "slide_label": row["slide_label"],
            "patch_label": tiles["patch_label"].to_numpy(),
            "provider": row["provider"],
            "image_path": tiles["image_path"].to_numpy(),
            "feature_path": str(feature_dir / f"{slide_id}.pt"),
            "feature_index": range(len(tiles)),
            "exhaustive": bool(row["has_mask"]),
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
    _write_json(seed_dir / "class_order.json", list(PATCH_LABELS))
    _write_json(seed_dir / "args.json", {"dataset": "panda", "seed": seed})


def _write_wsi_manifest(tagged: pd.DataFrame, seed_dir: Path) -> None:
    wsi = tagged.copy()
    wsi["cancer_type"] = wsi["slide_label"]
    wsi.to_csv(seed_dir / "manifest_splits.csv", index=False)


def _write_patch_manifests(
    tagged: pd.DataFrame, seed_dir: Path, seed: int, args: argparse.Namespace
) -> None:
    patch_pool = tagged[
        tagged["exhaustive"] & tagged["patch_label"].isin(PATCH_LABELS)
    ].copy()
    patch_pool["cancer_type"] = patch_pool["patch_label"]
    rng = np.random.default_rng(seed)
    budgets = {
        "train": args.train_patch_budget,
        "validation": args.eval_patch_budget,
        "test": args.eval_patch_budget,
    }
    parts = []
    for split in SPLITS:
        rows = _subsample(patch_pool[patch_pool["split"] == split], budgets[split], rng)
        rows.to_csv(seed_dir / f"{split}.csv", index=False)
        parts.append(rows)
    pd.concat(parts, ignore_index=True).to_csv(
        seed_dir / "patch_manifest.csv", index=False
    )
    counts = _patch_counts(parts[0])
    _write_json(seed_dir / "target_counts.json", counts)
    _write_json(seed_dir / "available_counts.json", counts)


def _subsample(
    rows: pd.DataFrame, budget: int, rng: np.random.Generator
) -> pd.DataFrame:
    if len(rows) <= budget:
        return rows
    chosen = rng.choice(len(rows), size=budget, replace=False)
    return rows.iloc[np.sort(chosen)]


def _patch_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["patch_label"].value_counts()
    return {label: int(counts.get(label, 0)) for label in PATCH_LABELS}


def _write_report(output: Path, frame: pd.DataFrame, bag_size: int) -> None:
    patch_frame = frame[frame["exhaustive"] & frame["patch_label"].isin(PATCH_LABELS)]
    patch_counts = {
        label: int((patch_frame["patch_label"] == label).sum())
        for label in PATCH_LABELS
    }
    slides = frame.drop_duplicates("slide_id")
    slide_counts = {
        label: int((slides["slide_label"] == label).sum()) for label in WSI_LABELS
    }
    provider_counts = slides["provider"].value_counts().to_dict()
    report = {
        "dataset": "PANDA",
        "n_slides": int(frame["slide_id"].nunique()),
        "n_patch_slides": int(patch_frame["slide_id"].nunique()),
        "wsi_bag_size": bag_size,
        "n_patches_capped": int(len(frame)),
        "class_counts_wsi": slide_counts,
        "imbalance_ratio_wsi": _ratio(list(slide_counts.values())),
        "class_counts_patch": patch_counts,
        "imbalance_ratio_patch": _ratio(list(patch_counts.values())),
        "slide_counts": slide_counts,
        "provider_counts": {str(k): int(v) for k, v in provider_counts.items()},
        "recommended_benchmark_mode": "native",
        "labels_wsi": list(WSI_LABELS),
        "labels_patch": list(PATCH_LABELS),
    }
    _write_json(output / "panda_prepare_report.json", report)


def _ratio(counts: list[int]) -> float:
    positive = [count for count in counts if count > 0]
    if not positive:
        return 0.0
    return float(max(positive) / min(positive))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
