"""Preflight: dependencies, source run records of every reused arm, and eval features per split."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    ensure_dirs,
    output_root,
    split_paths,
    write_json,
)

from breadth.fit import init_shard

from sites import allocation_dir

from centre import N_DRAWS, N_SPLITS, PATIENT_COUNTS

from uncertainty import REUSED_ARM_FAMILIES, baseline_config

__all__ = ["run_preflight"]

logger = logging.getLogger(__name__)


def _missing_sources(config: dict[str, Any]) -> list[str]:
    """Reused arm record directories that lack a run record."""
    missing = []
    for key, families in REUSED_ARM_FAMILIES.items():
        source = baseline_config(config, key)
        for s in range(N_SPLITS):
            paths = split_paths(ensure_dirs(source), s)
            for fam in families:
                for g in PATIENT_COUNTS:
                    for d in range(N_DRAWS):
                        out = allocation_dir(paths, f"{fam}{g}", d)
                        if not (out / RUN_RECORD_NAME).exists():
                            missing.append(str(out))
    return missing


def run_preflight(config: dict[str, Any]) -> Path:
    """Verify scipy, all reused source records, and per-split feature loading; write ``preflight.json``."""
    scipy = importlib.import_module("scipy")
    missing = _missing_sources(config)
    if missing:
        raise RuntimeError(
            f"{len(missing)} source records missing, first: {missing[0]}"
        )
    for s in range(N_SPLITS):
        _, names, evals, _ = init_shard(config, s)
        logger.info(
            "split %d: %d classes, %d val rows, %d test rows",
            s,
            len(names),
            len(evals.val_y),
            len(evals.test_y),
        )
    path = output_root(config) / "preflight.json"
    write_json(path, {"scipy": scipy.__version__, "source_records": "complete"})
    return path
