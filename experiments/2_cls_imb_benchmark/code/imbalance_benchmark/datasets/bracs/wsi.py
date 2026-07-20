"""Official BRACS WSI metadata and deterministic tissue-tile manifests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pandas as pd

from imbalance_benchmark.datasets.bracs import metadata
from imbalance_benchmark.datasets.bracs.audit import (
    load_tile_manifest,
)

_ROI_ID_COLUMNS = {"roi", "roi_id", "roi_filename", "roi_name"}
_SLIDE_COLUMNS = (
    "wsi_filename",
    "wsi_id",
    "slide_id",
    "wsi",
    "slide",
    "filename",
)
_CASE_COLUMNS = ("patient_id", "case_id", "patient", "case")
_LABEL_COLUMNS = (
    "wsi_label",
    "slide_label",
    "lesion_subtype",
    "subtype",
    "label",
    "class",
    "diagnosis",
    "category",
)
_SPLIT_COLUMNS = ("reference_set", "set", "split", "partition")


def load_wsi_metadata(root: Path, metadata_csv: Path | None = None) -> pd.DataFrame:
    """Load official BRACS WSI labels without deriving them from ROI annotations."""
    sheets = _read_metadata_sheets(root, metadata_csv)
    frames = [_normalize_wsi_sheet(sheet) for sheet in sheets]
    candidates = [frame for frame in frames if not frame.empty]
    if not candidates:
        columns = [list(sheet.columns) for sheet in sheets]
        raise ValueError(f"Could not identify BRACS WSI metadata: {columns}")
    frame = max(candidates, key=len)
    _validate_unique_slide_annotations(frame)
    return cast(pd.DataFrame, frame.drop_duplicates("slide_id").reset_index(drop=True))


def list_slide_tiles(tile_root: Path, slide_id: str) -> list[Path]:
    """Return eligible tissue tiles for one BRACS WSI in deterministic order."""
    for slide_dir in _slide_directories(tile_root, slide_id):
        if slide_dir.is_dir():
            tiles = [
                path
                for path in slide_dir.rglob("*")
                if path.suffix.lower() in metadata.IMAGE_EXTENSIONS
            ]
            return sorted(tiles, key=lambda path: path.as_posix().lower())
    return []


def build_manifest(
    root: Path,
    tile_root: Path,
    seed: int,
    metadata_csv: Path | None = None,
    tile_manifest_csv: Path | None = None,
    expected_slides: int = 547,
) -> pd.DataFrame:
    """Build patient-disjoint BRACS WSI-bag rows from official slide labels."""
    wsi_frame = load_wsi_metadata(root, metadata_csv)
    if len(wsi_frame) != expected_slides:
        raise ValueError(
            f"BRACS WSI metadata has {len(wsi_frame)} slides; expected {expected_slides}"
        )
    tile_manifest = load_tile_manifest(
        tile_manifest_csv or tile_root / "tile_manifest.csv",
        tile_root,
        expected_slides,
    )
    parts = _tile_manifest_parts(wsi_frame, tile_manifest)
    populated = [part for part in parts if not part.empty]
    if not populated:
        raise RuntimeError(f"No BRACS WSI tissue tiles found under {tile_root}")
    frame = pd.concat(populated, ignore_index=True)
    assignment = metadata.split_cases(frame, seed)
    tagged = frame.merge(assignment, on="case_id", how="inner")
    metadata.assert_patient_disjoint(tagged)
    return tagged


def _tile_manifest_parts(
    wsi_frame: pd.DataFrame, tile_manifest: pd.DataFrame
) -> list[pd.DataFrame]:
    slide_ids = tile_manifest["slide_id"].astype(str)
    return [
        _slide_rows(
            row,
            [
                Path(path)
                for path in tile_manifest.loc[
                    slide_ids == str(row["slide_id"]), "image_path"
                ]
            ],
        )
        for _, row in wsi_frame.iterrows()
    ]


def _read_metadata_sheets(root: Path, metadata_csv: Path | None) -> list[pd.DataFrame]:
    if metadata_csv is not None:
        return [pd.read_csv(metadata_csv, dtype=str)]
    workbook = metadata.find_summary_file(root)
    return list(pd.read_excel(workbook, sheet_name=None, dtype=str).values())


def _normalize_wsi_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    raw = sheet.dropna(how="all").copy()
    columns = {_canonical(column): column for column in raw.columns}
    if raw.empty or _ROI_ID_COLUMNS.intersection(columns):
        return pd.DataFrame()
    slide_col = _column(columns, _SLIDE_COLUMNS)
    case_col = _column(columns, _CASE_COLUMNS)
    label_col = _column(columns, _LABEL_COLUMNS)
    if slide_col is None or case_col is None or label_col is None:
        return pd.DataFrame()
    split_col = _column(columns, _SPLIT_COLUMNS)
    rows = [
        row
        for _, raw_row in raw.iterrows()
        if (row := _metadata_row(raw_row, slide_col, case_col, label_col, split_col))
    ]
    return pd.DataFrame(rows)


def _metadata_row(
    row: pd.Series,
    slide_col: object,
    case_col: object,
    label_col: object,
    split_col: object | None,
) -> dict[str, str] | None:
    slide_label = metadata.normalize_label(row[label_col])
    slide_id, case_id = _clean_id(row[slide_col]), _clean_id(row[case_col])
    if slide_label is None or not slide_id or not case_id:
        return None
    result = {
        "dataset": "bracs",
        "case_id": case_id,
        "slide_id": slide_id,
        "slide_label": slide_label,
        "cancer_type": slide_label,
    }
    if split_col is not None and (official_split := _clean_id(row[split_col])):
        result["official_split"] = official_split
    return result


def _validate_unique_slide_annotations(frame: pd.DataFrame) -> None:
    conflicts = frame.groupby("slide_id")[["case_id", "slide_label"]].nunique()
    conflicting = cast(pd.DataFrame, conflicts.loc[(conflicts > 1).any(axis=1)])
    if not conflicting.empty:
        raise ValueError(
            f"Conflicting official BRACS WSI metadata: {conflicting.index.tolist()[:5]}"
        )


def _slide_rows(row: pd.Series, tiles: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": "bracs",
            "case_id": str(row["case_id"]),
            "slide_id": str(row["slide_id"]),
            "patch_id": [path.stem for path in tiles],
            "slide_label": str(row["slide_label"]),
            "cancer_type": str(row["slide_label"]),
            "image_path": [str(path) for path in tiles],
        }
    )


def _slide_directories(tile_root: Path, slide_id: str) -> list[Path]:
    unprefixed = slide_id.removeprefix("BRACS_")
    aliases = dict.fromkeys((slide_id, unprefixed, f"BRACS_{unprefixed}"))
    return [tile_root / alias for alias in aliases]


def _canonical(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _column(columns: dict[str, object], names: tuple[str, ...]) -> object | None:
    return next((columns[name] for name in names if name in columns), None)


def _clean_id(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() == "nan" else Path(text).stem
