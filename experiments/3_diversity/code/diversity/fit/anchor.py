"""Import exp-2's confirmed 'random'-anchor runs into exp-3's own results tree."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    read_run_record,
    split_paths,
)

from diversity.fit.context import load_selection
from diversity.fit.units import METHODS, N_SEEDS
from diversity.manifests import (
    ALLOCATIONS,
    ANCHOR_ASSIGNMENT,
    SOURCE_MANIFEST,
    exp2_split_paths,
)

__all__ = ["import_anchor"]

logger = logging.getLogger(__name__)


def _assert_anchor_matches(
    exp2_paths: dict[str, Path], exp3_paths: dict[str, Path], allocation: str
) -> None:
    """Refuse to import a stale anchor: manifest content and tuning selections must still match."""
    random_manifest = exp3_paths["data"] / f"manifest_{allocation}_random.csv"
    source_manifest = exp2_paths["data"] / SOURCE_MANIFEST[allocation]
    if compute_sha256(random_manifest) != compute_sha256(source_manifest):
        raise RuntimeError(
            f"{allocation}: manifest_{allocation}_random.csv no longer matches exp-2's "
            "source manifest sha256; refusing to import a stale anchor"
        )
    _assert_tuning_params_match(exp2_paths, allocation)


def _assert_tuning_params_match(exp2_paths: dict[str, Path], allocation: str) -> None:
    selections = load_selection(exp2_paths["data"], allocation)
    anchor = ANCHOR_ASSIGNMENT[allocation]
    for method in METHODS:
        selected = selections[anchor][allocation][method]
        for seed_index in range(N_SEEDS):
            record_dir = (
                exp2_paths["results"]
                / f"assignment={anchor}"
                / allocation
                / method
                / f"seed={seed_index}"
            )
            record = read_run_record(record_dir)
            if record is None:
                raise RuntimeError(
                    f"{allocation}/{method}/seed={seed_index}: exp-2 run record missing; "
                    "cannot import as the random anchor"
                )
            if record.get("tuning_params") != selected:
                raise RuntimeError(
                    f"{allocation}/{method}/seed={seed_index}: exp-2's recorded tuning_params "
                    "no longer match the current selection; refusing to import a stale anchor"
                )


def import_anchor(config: dict[str, Any]) -> None:
    """Copy exp-2's confirmed 'random'-anchor runs into exp-3's own results tree.

    Plan Stage 3: a directory copy of exp-2's ``assignment=unassigned/balanced``
    and ``assignment=native/severe`` trees into exp-3's ``assignment=random/...``,
    gated by :func:`_assert_anchor_matches` per allocation and split.
    """
    for split_index in range(3):
        exp2_paths = exp2_split_paths(config, split_index)
        exp3_paths = split_paths(ensure_dirs(config), split_index)
        if not (exp3_paths["data"] / "manifest_freeze.json").exists():
            raise RuntimeError(f"split {split_index}: run build before import-anchor")
        for allocation in ALLOCATIONS:
            _assert_anchor_matches(exp2_paths, exp3_paths, allocation)
            anchor = ANCHOR_ASSIGNMENT[allocation]
            source = exp2_paths["results"] / f"assignment={anchor}" / allocation
            if not source.exists():
                raise RuntimeError(
                    f"split {split_index}: exp-2 has no {source} to import"
                )
            destination = exp3_paths["results"] / "assignment=random" / allocation
            shutil.copytree(source, destination, dirs_exist_ok=True)
            logger.info(
                "import-anchor: split=%s %s -> %s", split_index, source, destination
            )
