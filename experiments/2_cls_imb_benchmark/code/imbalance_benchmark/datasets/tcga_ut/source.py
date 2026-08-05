"""TCGA-UT Zenodo authentication: locked metadata, archive and MD5 validation.

Fetches and locks the immutable Zenodo v1.0 file metadata for record 5889558,
matches the shared ``data_raw.zip``'s central directory against it, and
extracts + MD5-verifies one nested class archive at a time. Packing the
verified images into a SqFS is :mod:`imbalance_benchmark.datasets.tcga_ut.pack`.
Stdlib-only by design: this never runs inside the training container and must
not gain a dependency on it.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    compute_sha256,
    sign_file,
    verify_signed_file,
    write_json,
)

__all__ = [
    "ZENODO_RECORD_ID",
    "ZENODO_VERSION",
    "EXPECTED_FILE_COUNT",
    "parse_zenodo_metadata",
    "fetch_zenodo_metadata",
    "load_or_fetch_metadata",
    "class_name",
    "class_files",
    "validate_shared_archive",
    "stage_nested_archive",
    "redownload_member",
    "extract_class_images",
]

ZENODO_RECORD_ID = "5889558"
ZENODO_VERSION = "1.0"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
LICENSE_KEY = "LICENSE"
EXPECTED_FILE_COUNT = 33  # 32 class archives + LICENSE
IMAGE_SUFFIXES = (".jpg", ".jpeg")


def _parse_zenodo_file(entry: dict[str, Any]) -> dict[str, Any]:
    checksum = str(entry.get("checksum", ""))
    if not checksum.startswith("md5:"):
        raise ValueError(f"Zenodo file {entry.get('key')!r} has no md5 checksum")
    return {
        "key": str(entry["key"]),
        "size": int(entry["size"]),
        "md5": checksum.removeprefix("md5:"),
        "url": str(entry.get("links", {}).get("self", "")),
    }


def parse_zenodo_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a Zenodo record API response into a locked file list."""
    version = str(payload.get("metadata", {}).get("version", ""))
    if version != ZENODO_VERSION:
        raise ValueError(
            f"Zenodo record {ZENODO_RECORD_ID} is not version {ZENODO_VERSION}: got {version!r}"
        )
    files = [_parse_zenodo_file(entry) for entry in payload.get("files", [])]
    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FILE_COUNT} Zenodo files, found {len(files)}"
        )
    return {"record_id": ZENODO_RECORD_ID, "version": version, "files": files}


def fetch_zenodo_metadata(timeout: float = 60.0) -> dict[str, Any]:
    """Fetch the immutable Zenodo v1.0 file metadata (record, version, checksums)."""
    with urllib.request.urlopen(ZENODO_API_URL, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_zenodo_metadata(payload)


def load_or_fetch_metadata(metadata_path: Path) -> dict[str, Any]:
    """Reuse a previously fetched and signed metadata sidecar, or fetch it fresh."""
    if (
        metadata_path.is_file()
        and metadata_path.with_suffix(metadata_path.suffix + ".sha256").is_file()
    ):
        verify_signed_file(metadata_path)
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = fetch_zenodo_metadata()
    write_json(metadata_path, metadata)
    sign_file(metadata_path)
    return metadata


def class_name(key: str) -> str:
    """Return the class name for one Zenodo archive key (its stem)."""
    return Path(key).stem


def class_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the 32 class archives, excluding LICENSE, sorted by class name."""
    return sorted(
        (f for f in files if f["key"] != LICENSE_KEY),
        key=lambda f: class_name(f["key"]),
    )


def validate_shared_archive(
    shared_zip: Path, files: list[dict[str, Any]]
) -> dict[str, zipfile.ZipInfo]:
    """Match the shared archive's central directory to the official Zenodo file list."""
    with zipfile.ZipFile(shared_zip) as archive:
        members = {info.filename: info for info in archive.infolist()}
    mismatched = [
        f["key"]
        for f in files
        if f["key"] not in members or members[f["key"]].file_size != f["size"]
    ]
    if mismatched:
        raise ValueError(
            f"Shared archive members differ from Zenodo metadata: {sorted(mismatched)}"
        )
    return {f["key"]: members[f["key"]] for f in files}


def _md5_of(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def stage_nested_archive(
    shared_zip: Path, member: zipfile.ZipInfo, expected_md5: str, scratch_dir: Path
) -> Path:
    """Extract one nested class archive from the shared zip and verify its MD5."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(shared_zip) as archive:
        extracted = Path(archive.extract(member, scratch_dir))
    if _md5_of(extracted) != expected_md5:
        raise ValueError(
            f"{member.filename} MD5 differs from Zenodo metadata; "
            "re-download this member from Zenodo, not the full archive"
        )
    return extracted


def redownload_member(file: dict[str, Any], scratch_dir: Path) -> Path:
    """Replace one corrupted nested archive by downloading it directly from Zenodo."""
    if not file.get("url"):
        raise ValueError(f"No Zenodo download URL recorded for {file['key']}")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    destination = scratch_dir / Path(str(file["key"])).name
    urllib.request.urlretrieve(str(file["url"]), destination)
    if (
        destination.stat().st_size != file["size"]
        or _md5_of(destination) != file["md5"]
    ):
        raise ValueError(
            f"Redownloaded {file['key']} still does not match Zenodo metadata"
        )
    return destination


def extract_class_images(
    nested_zip: Path, class_dir_name: str, dest_root: Path
) -> list[tuple[str, int, str]]:
    """Extract one class archive's images with CRC validation and hash each one.

    ``zipfile.ZipFile.extract`` validates each member's CRC-32 while
    decompressing it and raises ``BadZipFile`` on mismatch. Returns
    ``(relative_path, size, sha256)`` rows sorted by relative path.
    """
    class_root = dest_root / class_dir_name
    rows = []
    with zipfile.ZipFile(nested_zip) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(IMAGE_SUFFIXES):
                continue
            extracted = Path(archive.extract(info, class_root))
            rel = f"{class_dir_name}/{info.filename}".replace("\\", "/")
            rows.append((rel, extracted.stat().st_size, compute_sha256(extracted)))
    return sorted(rows, key=lambda row: row[0])
