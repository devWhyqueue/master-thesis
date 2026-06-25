"""BRACS workbook parsing and label normalization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pandas as pd

LABELS = ("N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC")
LESION_TYPES = {
    "N": "benign",
    "PB": "benign",
    "UDH": "benign",
    "FEA": "atypical",
    "ADH": "atypical",
    "DCIS": "malignant",
    "IC": "malignant",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_roi_metadata(root: Path, metadata_csv: Path | None = None) -> pd.DataFrame:
    """Load and normalize ROI metadata from a pre-converted CSV or BRACS.xlsx."""
    if metadata_csv is not None and metadata_csv.exists():
        return cast(pd.DataFrame, pd.read_csv(metadata_csv))
    workbook = find_summary_file(root)
    sheets = pd.read_excel(workbook, sheet_name=None)
    candidates = [_normalize_sheet(sheet) for sheet in sheets.values()]
    frames = [frame for frame in candidates if not frame.empty]
    if not frames:
        columns = {name: list(sheet.columns) for name, sheet in sheets.items()}
        raise ValueError(f"Could not identify a BRACS ROI metadata sheet: {columns}")
    frame = (
        max(frames, key=len).drop_duplicates(subset=["roi_id"]).reset_index(drop=True)
    )
    frame["dataset"] = "bracs"
    frame["lesion_type"] = frame["cancer_type"].map(lambda label: LESION_TYPES[label])
    return cast(pd.DataFrame, frame)


def find_summary_file(root: Path) -> Path:
    """Return the BRACS summary spreadsheet path."""
    matches = list(root.rglob("BRACS.xlsx"))
    if not matches:
        raise FileNotFoundError(f"BRACS.xlsx not found under {root}")
    return matches[0]


def index_roi_images(root: Path) -> dict[str, Path]:
    """Index ROI images by filename stem."""
    roi_roots = [path for path in root.rglob("BRACS_RoI") if path.is_dir()]
    search_roots = roi_roots or [root]
    index: dict[str, Path] = {}
    for search_root in search_roots:
        for path in search_root.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.stem, path)
    if not index:
        raise FileNotFoundError(f"No ROI images found under {search_roots}")
    return index


def normalize_label(value: object) -> str | None:
    """Normalize BRACS subtype labels to the seven report classes."""
    text = str(value).strip()
    upper = re.sub(r"[^A-Z0-9]+", "", text.upper())
    aliases = {
        "NORMAL": "N",
        "N": "N",
        "PB": "PB",
        "PATHOLOGICALBENIGN": "PB",
        "UDH": "UDH",
        "USUALDUCTALHYPERPLASIA": "UDH",
        "FEA": "FEA",
        "FLATEPITHELIALATYPIA": "FEA",
        "ADH": "ADH",
        "ATYPICALDUCTALHYPERPLASIA": "ADH",
        "DCIS": "DCIS",
        "DUCTALCARCINOMAINSITU": "DCIS",
        "IC": "IC",
        "INVASIVECARCINOMA": "IC",
    }
    return aliases.get(upper)


def _normalize_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    raw = sheet.dropna(how="all").copy()
    if raw.empty:
        return pd.DataFrame()
    columns = {_canonical(column): column for column in raw.columns}
    roi_col = _first(columns, ("roi", "roi_id", "roi_filename", "roi_name", "image"))
    slide_col = _first(columns, ("wsi", "slide", "slide_id", "wsi_filename"))
    case_col = _first(columns, ("patient", "patient_id", "case", "case_id"))
    label_col = _first(
        columns,
        ("subtype", "lesion_subtype", "label", "class", "diagnosis", "category"),
    )
    if roi_col is None or slide_col is None or case_col is None or label_col is None:
        return pd.DataFrame()
    return pd.DataFrame(
        row
        for _, raw_row in raw.iterrows()
        if (row := _metadata_row(raw_row, roi_col, slide_col, case_col, label_col))
    )


def _metadata_row(
    row: pd.Series,
    roi_col: object,
    slide_col: object,
    case_col: object,
    label_col: object,
) -> dict[str, str] | None:
    label = normalize_label(row[label_col])
    roi_id = _clean_id(row[roi_col])
    slide_id = _clean_id(row[slide_col])
    case_id = _clean_id(row[case_col])
    if label is None or not roi_id or not slide_id or not case_id:
        return None
    return {
        "case_id": case_id,
        "slide_id": slide_id,
        "roi_id": Path(roi_id).stem,
        "cancer_type": label,
    }


def _canonical(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _first(columns: dict[str, object], names: tuple[str, ...]) -> object | None:
    for name in names:
        if name in columns:
            return columns[name]
    for key, column in columns.items():
        if any(name in key for name in names):
            return column
    return None


def _clean_id(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() == "nan" else Path(text).stem
