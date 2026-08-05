"""TCGA-UT source packing: canonical manifest and materialize orchestration.

Combines per-class partial manifests (see
:mod:`imbalance_benchmark.datasets.tcga_ut.source`) into one canonical,
deterministic image-tree manifest, then hands off to
:mod:`imbalance_benchmark.datasets.tcga_ut.squashfs` to pack, extract-verify,
and publish it. ``materialize`` is the entry point run by the standalone
``materialize-tcga-ut`` command.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import sign_file, verify_signed_file, write_json
from imbalance_benchmark.datasets.tcga_ut import source, squashfs
from imbalance_benchmark.datasets.tcga_ut.squashfs import tree_hash

logger = logging.getLogger(__name__)

__all__ = [
    "class_partial_done",
    "write_class_partial",
    "read_class_partial",
    "combine_partials",
    "tree_hash",
    "validate_manifest_cohort",
    "materialize",
]

EXPECTED_CLASS_COUNT = 32
EXPECTED_IMAGE_COUNT = 1_608_060


def _partial_path(partials_dir: Path, class_dir_name: str) -> Path:
    return partials_dir / f"{class_dir_name}.jsonl"


def class_partial_done(partials_dir: Path, class_dir_name: str) -> bool:
    """Return whether a class's partial manifest is complete and unaltered."""
    path = _partial_path(partials_dir, class_dir_name)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False
    verify_signed_file(path)
    return True


def write_class_partial(
    partials_dir: Path, class_dir_name: str, rows: list[tuple[str, int, str]]
) -> None:
    """Persist one completed class's signed manifest rows for safe resume."""
    partials_dir.mkdir(parents=True, exist_ok=True)
    path = _partial_path(partials_dir, class_dir_name)
    with path.open("w", encoding="utf-8") as handle:
        for rel, size, sha in rows:
            handle.write(json.dumps({"path": rel, "size": size, "sha256": sha}) + "\n")
    sign_file(path)


def read_class_partial(
    partials_dir: Path, class_dir_name: str
) -> list[tuple[str, int, str]]:
    """Load and re-verify one class's signed partial manifest rows."""
    path = _partial_path(partials_dir, class_dir_name)
    verify_signed_file(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        rows.append((record["path"], record["size"], record["sha256"]))
    return rows


def combine_partials(
    partials_dir: Path, class_names: list[str], manifest_path: Path
) -> list[tuple[str, int, str]]:
    """Concatenate every class's signed partial into one sorted canonical manifest."""
    rows: list[tuple[str, int, str]] = []
    for name in class_names:
        rows.extend(read_class_partial(partials_dir, name))
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Duplicate image paths across materialized class archives")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for rel, size, sha in rows:
            handle.write(json.dumps({"path": rel, "size": size, "sha256": sha}) + "\n")
    sign_file(manifest_path)
    return rows


def validate_manifest_cohort(
    rows: list[tuple[str, int, str]], class_names: list[str]
) -> None:
    """Require the exact locked TCGA-UT image count and class count."""
    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IMAGE_COUNT} materialized images, found {len(rows)}"
        )
    if len(set(class_names)) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} classes, found {len(set(class_names))}"
        )


def _class_images_present(scratch_root: Path, name: str, expected_count: int) -> bool:
    """Whether this node's scratch already holds every image a class's partial expects.

    A signed partial only proves the images were once correctly extracted and
    hashed; scratch is job-local ``/tmp``, wiped between job attempts (and not
    guaranteed to be the same node across a resubmit), so the bytes may be gone
    even when the partial is complete.
    """
    class_dir = scratch_root / "images" / name
    if not class_dir.is_dir():
        return False
    return sum(1 for path in class_dir.rglob("*") if path.is_file()) == expected_count


def _materialize_one_class(
    file: dict[str, Any],
    matched: dict[str, Any],
    shared_zip: Path,
    partials_dir: Path,
    scratch_root: Path,
) -> None:
    """Extract and hash one class archive's images, or skip it if already present."""
    name = source.class_name(file["key"])
    if class_partial_done(partials_dir, name):
        rows = read_class_partial(partials_dir, name)
        if _class_images_present(scratch_root, name, len(rows)):
            logger.info("TCGA-UT class %s already materialized; skipping", name)
            return
        logger.info(
            "TCGA-UT class %s verified but missing from this node's scratch;"
            " re-extracting",
            name,
        )
    else:
        logger.info("Materializing TCGA-UT class %s", name)
    archives_dir = scratch_root / "archives"
    try:
        nested_zip = source.stage_nested_archive(
            shared_zip, matched[file["key"]], file["md5"], archives_dir
        )
    except ValueError:
        logger.warning(
            "Shared copy of %s failed validation; re-downloading", file["key"]
        )
        nested_zip = source.redownload_member(file, archives_dir)
    rows = source.extract_class_images(nested_zip, name, scratch_root / "images")
    write_class_partial(partials_dir, name, rows)
    nested_zip.unlink()


def _materialize_classes(cfg: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    """Stream, verify, and extract every class archive, resuming completed ones.

    Streams and deletes one nested archive at a time so scratch only ever
    holds the extracted images plus the largest single class archive.
    """
    shared_zip = Path(cfg["shared_zip"])
    partials_dir = Path(cfg["manifest_partials_dir"])
    scratch_root = Path(cfg["scratch_root"])
    files = source.class_files(metadata["files"])
    matched = source.validate_shared_archive(shared_zip, metadata["files"])
    for file in files:
        _materialize_one_class(file, matched, shared_zip, partials_dir, scratch_root)
    return [source.class_name(f["key"]) for f in files]


def _pack_and_publish(
    cfg: dict[str, Any], class_names: list[str]
) -> tuple[Path, list[tuple[str, int, str]], str]:
    """Combine partials, pack the SqFS, verify it by re-extraction, and publish it."""
    partials_dir = Path(cfg["manifest_partials_dir"])
    output_sqfs = Path(cfg["output_sqfs"])
    scratch_root = Path(cfg["scratch_root"])
    manifest_path = partials_dir / "canonical_manifest.jsonl"
    rows = combine_partials(partials_dir, class_names, manifest_path)
    validate_manifest_cohort(rows, class_names)
    source_hash = tree_hash(rows)

    partial_sqfs = output_sqfs.with_suffix(output_sqfs.suffix + ".partial")
    squashfs.build_squashfs(scratch_root / "images", partial_sqfs)
    if squashfs.mount_and_hash(partial_sqfs, scratch_root / "verify") != source_hash:
        raise RuntimeError(
            "Mounted SquashFS tree hash differs from the source manifest; refusing to publish"
        )
    squashfs.publish_squashfs(partial_sqfs, output_sqfs)
    sign_file(output_sqfs)
    return manifest_path, rows, source_hash


def materialize(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full materialization pipeline, resuming any already-verified class."""
    cfg = config["materialize_tcga_ut"]
    metadata = source.load_or_fetch_metadata(
        Path(cfg["manifest_partials_dir"]) / "zenodo_metadata.json"
    )
    class_names = _materialize_classes(cfg, metadata)
    manifest_path, _rows, source_hash = _pack_and_publish(cfg, class_names)

    sidecar = squashfs.build_provenance_sidecar(
        metadata,
        manifest_path,
        source_hash,
        Path(cfg["output_sqfs"]),
        True,
        EXPECTED_IMAGE_COUNT,
        EXPECTED_CLASS_COUNT,
    )
    sidecar_path = Path(cfg["materialization_sidecar"])
    write_json(sidecar_path, sidecar)
    sign_file(sidecar_path)
    return sidecar
