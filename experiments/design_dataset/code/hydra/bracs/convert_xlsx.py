"""Build per-ROI metadata CSV from the BRACS filesystem and xlsx patient map.

BRACS.xlsx contains WSI-level rows only; ROI filenames and labels come from
the BRACS_RoI directory tree (BRACS_<wsi>_<subtype>_<seq>.ext).
"""

from __future__ import annotations

import csv
import logging
import pathlib
import sys
from openpyxl import load_workbook  # available on login/compute nodes via system Python

logger = logging.getLogger(__name__)

_FIELDS = ["case_id", "slide_id", "roi_id", "cancer_type", "dataset", "lesion_type"]
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_SUBTYPE = {"N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"}
_LESION = {
    "N": "benign",
    "PB": "benign",
    "UDH": "benign",
    "FEA": "atypical",
    "ADH": "atypical",
    "DCIS": "malignant",
    "IC": "malignant",
}


def _wsi_patient_map(data_root: pathlib.Path) -> dict[str, str]:
    """Return {wsi_filename: patient_id} from the WSI_Information xlsx sheet."""
    xlsx = next(data_root.rglob("BRACS.xlsx"))
    wb = load_workbook(xlsx, read_only=True)
    for sheet in wb.worksheets:
        rows = list(sheet.rows)
        if not rows:
            continue
        headers = [str(c.value or "").strip().lower() for c in rows[0]]
        wsi_col = next(
            (i for i, h in enumerate(headers) if "wsi" in h and "filename" in h), None
        )
        pat_col = next((i for i, h in enumerate(headers) if "patient" in h), None)
        if wsi_col is None or pat_col is None:
            continue
        return {
            str(r[wsi_col].value).strip(): str(r[pat_col].value).split(".")[0]
            for r in rows[1:]
            if r[wsi_col].value and r[pat_col].value
        }
    raise ValueError("Could not find WSI filename / patient columns in BRACS.xlsx")


def _roi_rows(data_root: pathlib.Path, wsi_map: dict[str, str]) -> list[dict]:
    """Build per-ROI rows by walking the BRACS_RoI/latest_version directory."""
    roi_dir = next((p for p in data_root.rglob("BRACS_RoI") if p.is_dir()), None)
    if roi_dir is None:
        raise FileNotFoundError(f"BRACS_RoI not found under {data_root}")
    scan_root = (
        roi_dir / "latest_version" if (roi_dir / "latest_version").is_dir() else roi_dir
    )
    rows = []
    for img in sorted(scan_root.rglob("*")):
        if img.suffix.lower() not in _IMAGE_EXT:
            continue
        parts = img.stem.split("_")
        if len(parts) < 4 or parts[0] != "BRACS" or parts[2].upper() not in _SUBTYPE:
            continue
        wsi_id = f"{parts[0]}_{parts[1]}"
        label = parts[2].upper()
        patient_id = wsi_map.get(wsi_id, "")
        if not patient_id:
            continue
        rows.append(
            {
                "case_id": patient_id,
                "slide_id": wsi_id,
                "roi_id": img.stem,
                "cancer_type": label,
                "dataset": "bracs",
                "lesion_type": _LESION[label],
            }
        )
    return rows


def _write_csv(output_csv: pathlib.Path, rows: list) -> None:
    """Write normalised rows to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def convert(data_root: pathlib.Path, output_csv: pathlib.Path) -> int:
    """Build per-ROI metadata CSV from BRACS filesystem and return the row count."""
    wsi_map = _wsi_patient_map(data_root)
    rows = _roi_rows(data_root, wsi_map)
    _write_csv(output_csv, rows)
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data_root, output_csv = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    logger.info("wrote %d rows to %s", convert(data_root, output_csv), output_csv)
