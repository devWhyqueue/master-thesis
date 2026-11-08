"""Fingerprints and record checks behind exp-48's ``provenance.json`` completion marker."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from imbalance_benchmark.common import read_run_record
from imbalance_benchmark.datasets.feature_provenance import (
    VIRCHOW2_MODEL,
    VIRCHOW2_REVISION,
    VIRCHOW2_WEIGHTS_SHA256,
)

import mitigation

__all__ = [
    "PROVENANCE_NAME",
    "check_record",
    "encoder_identity",
    "frame_digest",
    "labels_digest",
    "sequence_digest",
    "software_identity",
    "source_digest",
    "write_atomic_json",
]

PROVENANCE_NAME = "provenance.json"
_OWN_FILES = ("protocol.py", "provenance.py", "fit.py")
_METHODS_FILES = (
    "__init__.py",
    "data.py",
    "encoder.py",
    "images.py",
    "fit.py",
    "methods/sampling.py",
    "train/audit.py",
    "train/eval.py",
    "train/loop.py",
    "train/stage1.py",
    "train/stage1_step.py",
)


def source_digest() -> str:
    """SHA-256 over the training/evaluation source actually used, line endings normalized."""
    methods_dir = Path(mitigation.__file__).resolve().parent
    own_dir = Path(__file__).resolve().parent
    files = [own_dir / f for f in _OWN_FILES] + [
        methods_dir / f for f in _METHODS_FILES
    ]
    sha = hashlib.sha256()
    for path in files:
        sha.update(path.name.encode("utf-8"))
        sha.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return sha.hexdigest()


def sequence_digest(values: list[str]) -> str:
    """SHA-256 of an ordered string sequence."""
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def labels_digest(labels: np.ndarray) -> str:
    """SHA-256 of an ordered integer label vector."""
    return hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    """Order-sensitive SHA-256 of a frame's rows and column names (CSV rendering)."""
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def encoder_identity() -> dict[str, str]:
    """Pinned pretrained checkpoint the LoRA adapters start from."""
    return {
        "model": VIRCHOW2_MODEL,
        "revision": VIRCHOW2_REVISION,
        "weights_sha256": VIRCHOW2_WEIGHTS_SHA256,
    }


def software_identity() -> dict[str, str | None]:
    """Interpreter, library versions, and the device a fit ran on."""
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "peft": version("peft"),
        "timm": version("timm"),
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "cudnn": str(torch.backends.cudnn.version()) if cuda else None,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }


def check_record(out_dir: Path, provenance: dict[str, Any]) -> None:
    """Raise unless the stored test arrays match the provenance's length, classes, and order."""
    rec = read_run_record(
        out_dir, splits=("test",), array_fields=("labels", "preds", "logits")
    )
    test = (rec or {}).get("splits", {}).get("test", {})
    if not all(key in test for key in ("labels", "preds", "logits")):
        raise ValueError(f"run record without test arrays at {out_dir}")
    labels, preds = np.asarray(test["labels"]), np.asarray(test["preds"])
    logits = np.asarray(test["logits"])
    expected_shape = (int(provenance["n_test"]), len(provenance["class_names"]))
    problems = [
        (logits.shape != expected_shape, f"logit shape {logits.shape}"),
        (not np.all(np.isfinite(logits)), "nonfinite logits"),
        (labels_digest(labels) != provenance["test_labels_sha256"], "label order"),
        (not np.array_equal(preds, logits.argmax(axis=1)), "preds/logits order"),
    ]
    failed = [message for bad, message in problems if bad]
    if failed:
        raise ValueError(f"incompatible run record at {out_dir}: {', '.join(failed)}")


def write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to a sibling temp file, then rename it into place."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
