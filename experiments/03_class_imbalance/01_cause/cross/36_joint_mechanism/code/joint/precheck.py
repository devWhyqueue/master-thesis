"""Precheck stage (0 new fits): verify this dataset's exp-34 separation dependency (frozen alpha,
signed native centres) and exp-25's TCGA-UT G=20 native controls are ready, then sign this
experiment's own precheck.json so main submission can verify the passing pilot's configuration and
artifact hashes (PLAN.md line 77) without re-deriving anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    output_root,
    read_run_record,
    sign_file,
    split_paths,
    write_json,
)

from breadth import N_SPLITS

from centre.pool import train_frame

from sites import allocation_dir

from spectrum import baseline_config

from separation.geometry import centres_path
from separation.precheck import load_alpha

__all__ = ["run_precheck"]

_NATIVE_PREVALENCE_ARMS = ("r1", "r100")


def _verify_separation(
    config: dict[str, Any], split_names: list[list[str]]
) -> dict[str, Any]:
    sep_config = baseline_config(config, "separation_outputs")
    precheck = load_alpha(sep_config)  # raises if exp-34's own alpha gate did not pass
    hashes = []
    for split_idx, names in enumerate(split_names):
        path = centres_path(sep_config, split_idx)
        if not path.exists():
            raise RuntimeError(f"Missing exp-34 native centres at {path}")
        hashes.append(compute_sha256(path))
    return {
        "alpha_expand": precheck.get("alpha_expand"),
        "alpha_contract": precheck.get("alpha_contract"),
        "centres_sha256": hashes,
    }


def _verify_native_prevalence(config: dict[str, Any]) -> bool:
    """TCGA-UT's own stored G=20 r1/r100 controls (gap-bridging, PLAN.md line 44) exist."""
    if config["dataset"]["name"] != "tcga_ut":
        return True
    prevalence_config = baseline_config(config, "native_prevalence_outputs")
    for split_idx in range(N_SPLITS):
        paths = split_paths(ensure_dirs(prevalence_config), split_idx)
        for arm in _NATIVE_PREVALENCE_ARMS:
            rec = read_run_record(
                allocation_dir(paths, arm, 0), splits=(), array_fields=()
            )
            if rec is None:
                return False
    return True


def run_precheck(config: dict[str, Any]) -> Path:
    """Verify dependencies are frozen and ready; sign this experiment's own precheck.json."""
    dataset = config["dataset"]["name"]
    split_names = [train_frame(config, s)[1] for s in range(N_SPLITS)]
    separation = _verify_separation(config, split_names)
    native_ready = _verify_native_prevalence(config)
    if not native_ready:
        raise RuntimeError("TCGA-UT native G=20 prevalence controls are not ready")
    payload = {
        "dataset": dataset,
        "class_names": split_names,
        "separation": separation,
        "native_prevalence_ready": native_ready,
    }
    out_p = output_root(config) / "data" / "precheck.json"
    write_json(out_p, payload)
    sign_file(out_p)
    return out_p
