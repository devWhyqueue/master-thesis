"""TCGA-UT SqFS packing: build, extract-verify, atomic publish, and provenance.

Isolated from :mod:`imbalance_benchmark.datasets.tcga_ut.pack` so tests can
stub out the ``mksquashfs``/``unsquashfs`` binaries (only present on a Hydra
compute node) without touching the manifest-packing logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import REPO_ROOT, compute_sha256

__all__ = [
    "tree_hash",
    "build_squashfs",
    "mount_and_hash",
    "publish_squashfs",
    "build_provenance_sidecar",
]


def tree_hash(rows: list[tuple[str, int, str]]) -> str:
    """Deterministic aggregate hash of a (path, size, sha256) manifest."""
    hasher = hashlib.sha256()
    for rel, size, sha in sorted(rows, key=lambda row: row[0]):
        hasher.update(f"{rel}\t{size}\t{sha}\n".encode("utf-8"))
    return hasher.hexdigest()


def build_squashfs(source_dir: Path, partial_path: Path) -> None:
    """Pack the extracted source tree into a SquashFS, initially under a ``.partial`` name."""
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.unlink(missing_ok=True)
    subprocess.run(
        ["mksquashfs", str(source_dir), str(partial_path), "-noappend"], check=True
    )


def mount_and_hash(sqfs_path: Path, extract_dir: Path) -> str:
    """Extract a SquashFS read-only and recompute its canonical tree hash.

    Uses ``unsquashfs`` rather than a live ``squashfuse`` mount: it needs no
    ``/dev/fuse`` access, which an unprivileged Apptainer container cannot
    reliably assume, and it validates the packed content just as strictly.
    """
    subprocess.run(
        ["unsquashfs", "-f", "-d", str(extract_dir), str(sqfs_path)], check=True
    )
    rows = [
        (
            str(path.relative_to(extract_dir)).replace(os.sep, "/"),
            path.stat().st_size,
            compute_sha256(path),
        )
        for path in sorted(extract_dir.rglob("*"))
        if path.is_file()
    ]
    return tree_hash(rows)


def publish_squashfs(partial_path: Path, final_path: Path) -> None:
    """Atomically publish a validated SquashFS under its final name."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_path, final_path)


def _tool_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_provenance_sidecar(
    metadata: dict[str, Any],
    manifest_path: Path,
    source_tree_hash: str,
    sqfs_path: Path,
    validated: bool,
    expected_image_count: int,
    expected_class_count: int,
) -> dict[str, Any]:
    """Assemble the signed provenance sidecar persisted next to the published SqFS."""
    return {
        "zenodo_record_id": metadata["record_id"],
        "zenodo_version": metadata["version"],
        "zenodo_files": metadata["files"],
        "aggregate_source_sha256": hashlib.sha256(
            json.dumps(
                sorted(metadata["files"], key=lambda f: f["key"]), sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "image_tree_manifest_path": str(manifest_path),
        "image_tree_manifest_sha256": compute_sha256(manifest_path),
        "image_tree_hash": source_tree_hash,
        "sqfs_path": str(sqfs_path),
        "sqfs_sha256": compute_sha256(sqfs_path),
        "tool_commit": _tool_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validated": validated,
        "expected_image_count": expected_image_count,
        "expected_class_count": expected_class_count,
    }
