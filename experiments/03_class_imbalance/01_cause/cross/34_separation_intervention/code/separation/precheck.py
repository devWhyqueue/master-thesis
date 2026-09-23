"""Precheck stage (0 new fits): each split's separation index J, and -- once the peer dataset's
own J is available -- the frozen alpha_expand/alpha_contract intervention strength and its gate.

J only depends on that dataset's own eligible training patients, so it is always written. Alpha
depends on BOTH datasets' J (BRACS expands toward TCGA-UT's separation; TCGA-UT's contraction is
the exact reciprocal), read through ``slurm.peer_outputs`` the way exp-33 reads its peer dataset's
own stored output. This command is therefore idempotent and must be run in order: TCGA-UT first
(writes its own J only), then BRACS (now reads TCGA-UT's J and computes alpha_expand), then
TCGA-UT again (now reads BRACS's J and computes its own reciprocal alpha_contract).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    output_root,
    sign_file,
    verify_signed_file,
    write_json,
)

from breadth import N_SPLITS

from centre.pool import train_frame

from spectrum import baseline_config

from separation import ALPHA_BOUNDS
from separation.geometry import run_centres

__all__ = ["run_precheck", "load_alpha"]

logger = logging.getLogger(__name__)


def _peer_j(config: dict[str, Any]) -> list[float] | None:
    """The peer dataset's own per-split J, if its precheck has already run."""
    if not config.get("slurm", {}).get("peer_outputs"):
        return None
    peer_config = baseline_config(config, "peer_outputs")
    path = output_root(peer_config) / "data" / "precheck.json"
    if not path.exists():
        return None
    verify_signed_file(path)
    return json.loads(path.read_text(encoding="utf-8"))["separation_index"]


def _own_separation_index(config: dict[str, Any]) -> list[float]:
    """Compute and store every split's centres; return this dataset's own per-split J."""
    j_values: list[float] = []
    for split_idx in range(N_SPLITS):
        train_df, names = train_frame(config, split_idx)
        out = run_centres(config, split_idx, train_df, names)
        meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        j_values.append(meta["separation_index"])
    return j_values


def _alpha_fields(
    dataset: str, j_values: list[float], peer_j: list[float]
) -> dict[str, Any]:
    """Expansion factor and this dataset's own alpha, once both datasets' J are known."""
    tcga_j = peer_j if dataset == "bracs" else j_values
    bracs_j = j_values if dataset == "bracs" else peer_j
    expansion = [t / b for t, b in zip(tcga_j, bracs_j)]
    lo, hi = ALPHA_BOUNDS
    fields: dict[str, Any] = {
        "expansion_factor": expansion,
        "alpha_gate_pass": all(lo <= e <= hi for e in expansion),
    }
    fields["alpha_expand" if dataset == "bracs" else "alpha_contract"] = (
        expansion if dataset == "bracs" else [1.0 / e for e in expansion]
    )
    return fields


def run_precheck(config: dict[str, Any]) -> Path:
    """Compute this dataset's per-split J; add alpha once the peer's J is available."""
    dataset = config["dataset"]["name"]
    j_values = _own_separation_index(config)
    payload: dict[str, Any] = {"dataset": dataset, "separation_index": j_values}
    peer_j = _peer_j(config)
    if peer_j is not None:
        payload |= _alpha_fields(dataset, j_values, peer_j)
    else:
        logger.info("Peer dataset's J not ready yet; alpha not computed this run.")

    out_p = output_root(config) / "data" / "precheck.json"
    write_json(out_p, payload)
    sign_file(out_p)
    if peer_j is not None and not payload["alpha_gate_pass"]:
        lo, hi = ALPHA_BOUNDS
        raise RuntimeError(
            f"Precheck alpha gate failed for {dataset}: expansion "
            f"{payload['expansion_factor']} outside [{lo}, {hi}] in some split; "
            "stopping before any fit submission."
        )
    return out_p


def load_alpha(config: dict[str, Any]) -> dict[str, Any]:
    """Load and verify this dataset's frozen precheck, including its alpha array."""
    path = output_root(config) / "data" / "precheck.json"
    verify_signed_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("alpha_gate_pass"):
        raise RuntimeError("Precheck alpha gate did not pass; refusing to fit.")
    return payload
