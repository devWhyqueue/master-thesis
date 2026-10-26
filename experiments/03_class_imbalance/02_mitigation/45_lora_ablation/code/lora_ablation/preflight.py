"""Inventory and verify every reusable input before any head is fit.

Besides the adapted source records, the audit checks the prerequisites of the new fits: the
pinned Virchow2 checkpoint in the local cache (never downloaded here), the pretrained feature
cache, and the raw images CUDA reads. Run it where the image mounts exist.

Matching label arrays alone never establishes pairing: adapted run records carry no patient
IDs (``load_test_identity`` rebuilds them by position from the manifest). Identity is therefore
established through the signed exp-2 freeze, which must lock the exact manifest bytes, plus an
order-sensitive label check of every source record against that manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    compute_sha256,
    output_root,
    read_run_record,
)
from huggingface_hub import snapshot_download
from imbalance_benchmark.datasets.feature_provenance import (
    VIRCHOW2_REVISION,
    VIRCHOW2_WEIGHTS_SHA256,
    resolve_feature_provenance,
)

from breadth import exp2_split_paths

from centre import N_DRAWS, N_SPLITS

from decodability.evidence import load_freeze_meta

from mitigation.grid import stage1_jobs, stage2_jobs

from prevalence import patients_per_class
from prevalence.fit import _shard_context, class_counts

from lora_ablation import (
    config_digest,
    digest,
    source_root,
    source_run_dir,
    validate_protocol,
    workload_counts,
)

__all__ = [
    "audit_inputs",
    "write_inventory",
    "digest",
    "config_digest",
    "INVENTORY_NAME",
]

INVENTORY_NAME = "inventory.json"


def _strings(node: Any) -> Iterator[str]:
    """Every string value nested in a JSON-like structure."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _check_features(manifest: pd.DataFrame, split: int) -> str:
    """Every feature directory must carry the pinned Virchow2 provenance (fp16, dim 2560)."""
    pinned = resolve_feature_provenance({})
    for directory in sorted({str(Path(p).parent) for p in manifest["feature_path"]}):
        path = Path(directory) / "feature_provenance.json"
        if not path.is_file():
            raise ValueError(f"split {split}: feature provenance missing at {path}")
        if json.loads(path.read_text(encoding="utf-8")) != pinned:
            raise ValueError(
                f"split {split}: feature provenance differs from pinned Virchow2 at {path}"
            )
    return digest(pinned)


def cached_checkpoint() -> dict[str, str]:
    """Revision and weight hash of the pinned Virchow2 snapshot in the local cache; never downloads."""
    try:
        snapshot = Path(
            snapshot_download(
                "paige-ai/Virchow2",
                revision=VIRCHOW2_REVISION,
                allow_patterns=["config.json", "model.safetensors"],
                local_files_only=True,
            )
        )
    except (
        Exception
    ) as error:  # hub raises several types when the snapshot is not cached
        raise ValueError(
            f"pinned Virchow2 checkpoint is not cached: {error}"
        ) from error
    weights = snapshot / "model.safetensors"
    if not weights.is_file():
        raise ValueError(f"pinned Virchow2 weights missing at {weights}")
    return {"revision": VIRCHOW2_REVISION, "weights_sha256": compute_sha256(weights)}


def _check_checkpoint() -> dict[str, str]:
    found = cached_checkpoint()
    if found != {
        "revision": VIRCHOW2_REVISION,
        "weights_sha256": VIRCHOW2_WEIGHTS_SHA256,
    }:
        raise ValueError(f"checkpoint differs from the pinned Virchow2: {found}")
    return found


def _check_images(manifest: pd.DataFrame, split: int, samples: int = 64) -> int:
    """Raw images must be readable: an evenly spaced sample of manifest rows (always the first)."""
    rows = np.unique(np.linspace(0, len(manifest) - 1, num=samples).astype(int))
    for path in manifest["image_path"].iloc[rows]:
        if not Path(path).is_file():
            raise ValueError(f"split {split}: raw image missing prerequisite: {path}")
    return len(rows)


class _Split:
    """One split's verified identity evidence."""

    def __init__(self, config: dict[str, Any], split: int) -> None:
        paths = exp2_split_paths(config, split)
        manifest_path = paths["data"] / "manifest.csv"
        freeze = load_freeze_meta(paths)
        self.manifest_sha = compute_sha256(manifest_path)
        if self.manifest_sha not in set(_strings(freeze)):
            raise ValueError(
                f"split {split}: manifest identity is not locked by the freeze"
            )
        self.classes = list(freeze["class_names"])
        manifest = pd.read_csv(manifest_path)
        index = {name: i for i, name in enumerate(self.classes)}
        test = manifest.query("split == 'test'")
        self.test_y = np.array([index[c] for c in test["cancer_type"]], dtype=np.int64)
        self.train_df = manifest.query("split == 'train'").reset_index(drop=True)
        self.freeze_sha = digest(freeze)
        self.provenance_sha = _check_features(manifest, split)
        self.images_checked = _check_images(manifest, split)

    def counts(self, split: int, draw: int, arm: str, g: int) -> dict[str, int]:
        """Expected per-class training counts of one (draw, arm), from the shared allocator."""
        shard = _shard_context(self.train_df, self.classes, split, draw, g)
        counts = class_counts(
            arm, shard.perm, shard.available, shard.pool_counts, shard.g
        )
        return dict(zip(self.classes, (int(c) for c in counts)))


def _audit_record(
    config: dict[str, Any], evidence: _Split, split: int, job: tuple, draw: int
) -> dict[str, Any]:
    arm, method, param = job
    directory = source_run_dir(config, split, arm, method, param, draw)
    path = directory / RUN_RECORD_NAME
    if not path.is_file():
        raise ValueError(f"missing source record: {path}")
    record = read_run_record(directory, splits=("test",), array_fields=("labels",))
    if record is None:
        raise ValueError(f"missing source record: {path}")
    labels = np.asarray(record["splits"]["test"]["labels"])
    if not np.array_equal(labels, evidence.test_y):
        raise ValueError(
            f"source labels differ from manifest targets (order or content): {path}"
        )
    expected = evidence.counts(split, draw, arm, patients_per_class(config))
    if {k: int(v) for k, v in record["class_counts"].items()} != expected:
        raise ValueError(
            f"source class counts differ from the shared allocation: {path}"
        )
    return {
        "split": split,
        "draw": draw,
        "arm": arm,
        "method": method,
        "param": param,
        "path": path.relative_to(source_root(config)).as_posix(),
        "sha256": compute_sha256(path),
    }


def _inventory(
    config: dict[str, Any], evidence: dict[int, _Split], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Hashes of protocol, config, per-split evidence, and every adapted source record."""
    return {
        "dataset": config["dataset"]["name"],
        "protocol_sha256": digest([config["protocol"], config["source_grid"]]),
        "config_sha256": config_digest(config),
        "splits": {
            str(s): {
                "manifest_sha256": e.manifest_sha,
                "freeze_sha256": e.freeze_sha,
                "feature_provenance_sha256": e.provenance_sha,
            }
            for s, e in evidence.items()
        },
        "protocol_id": config["protocol"]["id"],
        "checkpoint": _check_checkpoint(),
        "images_checked": {str(s): e.images_checked for s, e in evidence.items()},
        "source_records": records,
        "workload": workload_counts(config),
    }


def audit_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Verify protocol, identity, provenance and every adapted candidate; return the inventory."""
    validate_protocol(config)
    jobs = [tuple(job) for job in (*stage1_jobs(config), *stage2_jobs(config))]
    evidence = {s: _Split(config, s) for s in range(N_SPLITS)}
    records = [
        _audit_record(config, evidence[s], s, job, d)
        for s in range(N_SPLITS)
        for d in range(N_DRAWS)
        for job in jobs
    ]
    inventory = _inventory(config, evidence, records)
    inventory["fingerprint"] = digest(inventory)
    return inventory


def write_inventory(config: dict[str, Any], inventory: dict[str, Any]) -> Path:
    """Persist the immutable inventory; an existing differing one is never overwritten."""
    path = output_root(config) / INVENTORY_NAME
    if path.exists():
        if (
            json.loads(path.read_text(encoding="utf-8"))["fingerprint"]
            != inventory["fingerprint"]
        ):
            raise ValueError(
                f"inventory at {path} differs from the current audit; inputs changed"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return path
