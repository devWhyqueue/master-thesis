"""Dataset adapters producing a unified manifest schema.

Every adapter returns a frame with ``case_id``, ``slide_id``, ``cancer_type``,
and ``split`` columns. Patch-regime rows from BRACS/CAMELYON16/PANDA also carry
``image_path`` for downstream Virchow2 feature extraction (see
``imbalance_benchmark.datasets.features``); TCGA-UT rows already carry
``feature_path``/``feature_index`` because its features are pre-extracted on
the cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.datasets import bracs, bracs_tiling, camelyon16, panda, tcga_ut
from imbalance_benchmark.datasets import features as feature_lib

__all__ = ["DATASET_NAMES", "build_manifest"]

DATASET_NAMES = ("bracs", "camelyon16", "panda", "tcga_ut")


def build_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Dispatch to the configured dataset adapter and return a unified manifest."""
    dataset_cfg = config["dataset"]
    name = dataset_cfg["name"]
    builders: dict[str, Callable[[dict[str, Any]], pd.DataFrame]] = {
        "bracs": _build_bracs,
        "camelyon16": _build_camelyon16,
        "panda": _build_panda,
        "tcga_ut": _build_tcga_ut,
    }
    if name not in builders:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASET_NAMES}")
    return builders[name](config)


def _build_bracs(config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    root = Path(dataset_cfg["root"])
    regime = dataset_cfg.get("regime", "patch")
    metadata_csv = dataset_cfg.get("metadata_csv")
    tile_root = Path(dataset_cfg.get("tile_root", root / "tiles"))
    tile_size = int(dataset_cfg.get("tile_size", 256))
    roi_frame = bracs.load_roi_metadata(
        root, Path(metadata_csv) if metadata_csv else None
    )
    image_index = bracs.index_roi_images(root)
    tiled, _ = bracs_tiling.tile_rois(roi_frame, image_index, tile_root, tile_size)
    if tiled.empty:
        raise RuntimeError("No BRACS ROI tiles were generated.")
    assignment = bracs.split_cases(tiled, int(dataset_cfg.get("seed", 0)))
    tagged = tiled.merge(assignment, on="case_id", how="inner")
    bracs.assert_patient_disjoint(tagged)
    tagged["patch_label"] = tagged["cancer_type"]
    tagged["slide_label"] = tagged["cancer_type"]
    if regime == "wsi":
        tagged["cancer_type"] = tagged["slide_label"]
    return tagged


def _build_camelyon16(config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    data_root = Path(dataset_cfg["root"])
    regime = dataset_cfg.get("regime", "patch")
    slide_labels = camelyon16.load_slide_labels(data_root)
    slides = [
        slide
        for slide in camelyon16.slides_with_patches(data_root)
        if slide in slide_labels
        and (data_root / "masks" / f"{slide}_mask.npy").is_file()
    ]
    if not slides:
        raise RuntimeError("No usable CAMELYON16 slides found.")
    bag_size = int(
        np.median([len(camelyon16.list_slide_patches(data_root, s)) for s in slides])
    )
    parts = [
        _camelyon16_slide_rows(data_root, slide, slide_labels[slide], bag_size)
        for slide in slides
    ]
    frame = pd.concat(parts, ignore_index=True)
    slide_frame = cast(
        pd.DataFrame, frame.drop_duplicates("case_id")[["case_id", "slide_label"]]
    )
    assignment = camelyon16.split_cases(slide_frame, int(dataset_cfg.get("seed", 0)))
    tagged = frame.merge(assignment, on="case_id", how="inner")
    camelyon16.assert_slide_disjoint(tagged)
    tagged["cancer_type"] = (
        tagged["slide_label"] if regime == "wsi" else tagged["patch_label"]
    )
    if regime == "patch":
        tagged = cast(pd.DataFrame, tagged[tagged["exhaustive"]]).reset_index(drop=True)
    return tagged


def _camelyon16_slide_rows(
    data_root: Path, slide_id: str, slide_label: str, bag_size: int
) -> pd.DataFrame:
    patches = camelyon16.list_slide_patches(data_root, slide_id)
    if len(patches) > bag_size:
        keep = np.linspace(0, len(patches) - 1, bag_size).astype(int)
        patches = [patches[index] for index in keep]
    patch_ids = [pid for pid, _ in patches]
    labels = camelyon16.patch_labels(
        camelyon16.load_mask(data_root, slide_id), patch_ids
    )
    exhaustive = slide_id not in camelyon16.NON_EXHAUSTIVE_TUMOR
    return pd.DataFrame(
        {
            "dataset": "camelyon16",
            "case_id": slide_id,
            "slide_id": slide_id,
            "patch_id": patch_ids,
            "slide_label": slide_label,
            "patch_label": labels,
            "image_path": [str(path) for _, path in patches],
            "exhaustive": exhaustive,
        }
    )


def _build_panda(config: dict[str, Any]) -> pd.DataFrame:
    """Build the PANDA manifest from a pre-selected slide list and per-slide tile CSVs.

    ``tiles_dir`` is produced by an upstream on-cluster tiling stage (patch_id,
    image_path, patch_label already mask-decoded via ``panda.cell_label``); this
    adapter only assembles the manifest and applies patient-disjoint splitting.
    """
    dataset_cfg = config["dataset"]
    regime = dataset_cfg.get("regime", "patch")
    selection = pd.read_csv(dataset_cfg["selection_path"])
    tiles_dir = Path(dataset_cfg["tiles_dir"])
    tiles = {
        str(slide_id): pd.read_csv(path)
        for slide_id in selection["slide_id"]
        if (path := tiles_dir / f"{slide_id}.csv").is_file()
        and not pd.read_csv(path).empty
    }
    if not tiles:
        raise RuntimeError("No tiled PANDA slides found.")
    bag_size = int(np.median([len(frame) for frame in tiles.values()]))
    parts = [
        _panda_slide_rows(row, tiles[str(row["slide_id"])], bag_size)
        for _, row in selection.iterrows()
        if str(row["slide_id"]) in tiles
    ]
    frame = pd.concat(parts, ignore_index=True)
    slide_frame = cast(
        pd.DataFrame, frame.drop_duplicates("case_id")[["case_id", "slide_label"]]
    )
    assignment = panda.split_cases(slide_frame, int(dataset_cfg.get("seed", 0)))
    tagged = frame.merge(assignment, on="case_id", how="inner")
    panda.assert_slide_disjoint(tagged)
    tagged["cancer_type"] = (
        tagged["slide_label"] if regime == "wsi" else tagged["patch_label"]
    )
    if regime == "patch":
        tagged = cast(
            pd.DataFrame,
            tagged[
                tagged["exhaustive"] & tagged["patch_label"].isin(panda.PATCH_LABELS)
            ],
        ).reset_index(drop=True)
    return tagged


def _panda_slide_rows(
    row: pd.Series, tiles: pd.DataFrame, bag_size: int
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
            "exhaustive": bool(row["has_mask"]),
        }
    )


def _build_tcga_ut(config: dict[str, Any]) -> pd.DataFrame:
    """Build the TCGA-UT manifest from pre-extracted chunked feature tensors."""
    dataset_cfg = config["dataset"]
    raw_root = Path(dataset_cfg["raw_root"])
    feature_dir = Path(dataset_cfg["feature_dir"])
    feature_glob = str(dataset_cfg.get("feature_glob", "*.pt"))
    suffix_pattern = str(dataset_cfg.get("feature_suffix_pattern", "_[0-9]+"))
    labels, _ = tcga_ut.collect_slide_labels(raw_root)
    chunk_manifest, slide_manifest, _ = tcga_ut.build_feature_manifest(
        feature_dir, labels, feature_glob, suffix_pattern
    )
    patch_rows = [
        row
        for _, chunk_row in chunk_manifest.iterrows()
        for row in _expand_chunk(chunk_row)
    ]
    frame = pd.DataFrame(patch_rows)
    assignment = tcga_ut.split_cases(slide_manifest, int(dataset_cfg.get("seed", 0)))
    tagged = frame.merge(assignment, on="case_id", how="inner")
    tcga_ut.assert_case_disjoint(tagged)
    return tagged


def _expand_chunk(chunk_row: pd.Series) -> list[dict[str, Any]]:
    """Expand one chunk tensor into one manifest row per patch it contains."""
    n_rows = feature_lib.load_slide_features(str(chunk_row["feature_path"])).shape[0]
    return [
        {
            "dataset": "tcga_ut",
            "case_id": chunk_row["case_id"],
            "slide_id": chunk_row["slide_id"],
            "cancer_type": chunk_row["cancer_type"],
            "feature_path": chunk_row["feature_path"],
            "feature_index": index,
        }
        for index in range(n_rows)
    ]
