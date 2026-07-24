"""Dataset adapters producing unified case, slide, and target manifests.

Image-backed rows carry paths; pre-extracted TCGA rows carry feature references.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd

from imbalance_benchmark.datasets import (
    bracs,
    camelyon16,
    panda,
    panda_audit,
    tcga_ut,
)

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
        "tcga_ut": tcga_ut.build_manifest,
    }
    if name not in builders:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASET_NAMES}")
    return builders[name](config)


def _build_bracs(config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    root = Path(dataset_cfg["root"])
    regime = dataset_cfg.get("regime", "patch")
    if regime == "wsi":
        metadata_csv = dataset_cfg.get("wsi_metadata_csv")
        return bracs.build_wsi_manifest(
            root,
            Path(dataset_cfg.get("wsi_tile_root", root / "tiles" / "wsi")),
            int(dataset_cfg.get("seed", 0)),
            Path(metadata_csv) if metadata_csv else None,
            Path(
                dataset_cfg.get(
                    "wsi_tile_manifest",
                    Path(dataset_cfg.get("wsi_tile_root", root / "tiles" / "wsi"))
                    / "tile_manifest.csv",
                )
            ),
            int(dataset_cfg.get("expected_wsi_count", 547)),
        )
    metadata_csv = dataset_cfg.get("metadata_csv")
    tile_root = Path(dataset_cfg.get("tile_root", root / "tiles"))
    tile_size = int(dataset_cfg.get("tile_size", 256))
    roi_frame = bracs.load_roi_metadata(
        root, Path(metadata_csv) if metadata_csv else None
    )
    image_index = bracs.index_roi_images(root)
    tiled = bracs.tile_rois(roi_frame, image_index, tile_root, tile_size)
    if tiled.empty:
        raise RuntimeError("No BRACS ROI tiles were generated.")
    assignment = bracs.split_cases(tiled, int(dataset_cfg.get("seed", 0)))
    tagged = tiled.merge(assignment, on="case_id", how="inner")
    bracs.assert_patient_disjoint(tagged)
    tagged["patch_label"] = tagged["cancer_type"]
    tagged["slide_label"] = tagged["cancer_type"]
    return tagged


def _build_camelyon16(config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    data_root = Path(dataset_cfg["root"])
    patches_root = Path(dataset_cfg.get("patches_root", data_root / "patches" / "20x"))
    regime = dataset_cfg.get("regime", "patch")
    slide_labels = camelyon16.load_slide_labels(data_root)
    slides = [
        slide
        for slide in camelyon16.slides_with_patches(patches_root)
        if slide in slide_labels
    ]
    if regime == "patch":
        slides = [
            slide
            for slide in slides
            if (data_root / "masks" / f"{slide}_mask.npy").is_file()
        ]
    if not slides:
        raise RuntimeError("No usable CAMELYON16 slides found.")
    parts = [
        _camelyon16_slide_rows(
            data_root,
            patches_root,
            slide,
            slide_labels[slide],
            include_patch_labels=regime == "patch",
        )
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
    data_root: Path,
    patches_root: Path,
    slide_id: str,
    slide_label: str,
    *,
    include_patch_labels: bool,
) -> pd.DataFrame:
    patches = camelyon16.list_slide_patches(patches_root, slide_id)
    patch_ids = [pid for pid, _ in patches]
    rows: dict[str, Any] = {
        "dataset": "camelyon16",
        "case_id": slide_id,
        "slide_id": slide_id,
        "patch_id": patch_ids,
        "slide_label": slide_label,
        "image_path": [str(path) for _, path in patches],
    }
    if include_patch_labels:
        rows["patch_label"] = camelyon16.patch_labels(
            camelyon16.load_mask(data_root, slide_id), patch_ids
        )
        rows["exhaustive"] = slide_id not in camelyon16.NON_EXHAUSTIVE_TUMOR
    return pd.DataFrame(rows)


def _build_panda(config: dict[str, Any]) -> pd.DataFrame:
    """Build PANDA rows from a validated full-cohort level-0 tile inventory."""
    dataset_cfg = config["dataset"]
    regime = dataset_cfg.get("regime", "patch")
    official = panda.load_slide_frame(Path(dataset_cfg["root"]))
    selection = pd.read_csv(dataset_cfg["selection_path"])
    expected_slides = int(dataset_cfg.get("expected_slide_count", 10_616))
    panda_audit.validate_selection(
        selection,
        official,
        expected_slides,
    )
    tiles_dir = Path(dataset_cfg["tiles_dir"])
    tiles = panda.load_tile_inventory(selection, tiles_dir)
    panda_audit.validate_tile_inventory(selection, tiles, official, expected_slides)
    parts = [
        _panda_slide_rows(row, tiles[str(row["slide_id"])])
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


def _panda_slide_rows(row: pd.Series, tiles: pd.DataFrame) -> pd.DataFrame:
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
