"""Images stage: copy the selected example tiles out of the mounted SquashFS image archives.

The job mounts the archive at the manifest's own image root (``slurm.shared_squashfs``), so each
example's stored ``image_path`` is read unchanged; copies keep their bytes and colours exactly and
are listed with their source path and sha256.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from imbalance_benchmark.common import ensure_dirs

__all__ = ["copy_tiles", "run_images"]


def copy_tiles(rows: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Copy each distinct patch once to ``out_dir`` under a flat name; return the provenance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for patch_id, source in rows.drop_duplicates("patch_id")[
        ["patch_id", "image_path"]
    ].itertuples(index=False):
        name = str(patch_id).replace("/", "__") + Path(source).suffix
        shutil.copyfile(source, out_dir / name)
        digest = hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        records.append(
            {"patch_id": patch_id, "image_path": source, "file": name, "sha256": digest}
        )
    return pd.DataFrame(records)


def run_images(config: dict[str, Any]) -> Path:
    """Copy every example tile of this dataset and write images/provenance.csv."""
    data = ensure_dirs(config)["data"]
    rows = pd.read_csv(data / "examples_annotated.csv", dtype=str)
    provenance = copy_tiles(rows, data / "images")
    provenance["archive"] = ";".join(
        i["source"] for i in config["slurm"]["shared_squashfs"]
    )
    path = data / "images" / "provenance.csv"
    provenance.to_csv(path, index=False)
    return path
