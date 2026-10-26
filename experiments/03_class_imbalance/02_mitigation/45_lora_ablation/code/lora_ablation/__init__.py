"""Protocol constants and config loading for the LoRA-omission follow-up (exp-45, v2).

Train-time logit adjustment (``la``), post-hoc logit adjustment (``posthoc_la``) and CUDA are
compared at ``r100`` with their LoRA-adapted counterparts of exp-41/43. The config merges the
source experiment's dataset and training settings with a restricted ``r100`` grid, so exp-40's
``stage1_jobs``/``stage2_jobs`` enumerate exactly the follow-up candidates (``ce`` is the
supporting control and the source head of post-hoc adjustment).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import REPO_ROOT, load_config, output_root

from centre import N_DRAWS, N_SPLITS

from mitigation import method_label
from mitigation.grid import stage1_jobs, stage2_jobs

__all__ = [
    "ARM",
    "CUDA_CONSTANTS",
    "CUDA_MICRO_BATCH",
    "INIT_SEED_BASE",
    "MARGIN_PP",
    "METHODS",
    "N_CONTRASTS",
    "PROTOCOL_ID",
    "SPEC_GRIDS",
    "STAGE1_RECIPES",
    "STAGE_OF",
    "config_digest",
    "digest",
    "source_root",
    "source_run_dir",
    "init_seed",
    "load_protocol_config",
    "normalized_grid",
    "protocol_grid",
    "validate_protocol",
    "workload_counts",
]

PROTOCOL_ID = "strong-mitigations-v2"
ARM = "r100"
METHODS = ("la", "posthoc_la", "cuda")
STAGE_OF = {"ce": "stage1", "la": "stage1", "cuda": "stage1", "posthoc_la": "stage2"}
N_CONTRASTS = 6
MARGIN_PP = 1.0
INIT_SEED_BASE = 10000
CUDA_MICRO_BATCH = 64
CUDA_CONSTANTS = {"gamma": 0.6, "check_size": 10, "max_strength": 10}
_CUDA: list[float | None] = [0.25, 0.5, 1.0]

# Design table: strengths per dataset; ``ce`` has none and feeds post-hoc adjustment.
SPEC_GRIDS: dict[str, dict[str, dict[str, list[float | None]]]] = {
    "bracs": {
        "stage1": {"ce": [None], "la": [0.5, 1.0], "cuda": _CUDA},
        "stage2": {"posthoc_la": [0.5, 1.0, 1.5, 2.0]},
    },
    "tcga_ut": {
        "stage1": {"ce": [None], "la": [0.25, 0.5, 1.0], "cuda": _CUDA},
        "stage2": {"posthoc_la": [0.25, 0.5, 1.0, 1.5, 2.0]},
    },
}
# The source first-stage recipe (exp-41 / exp-43 final configs).
STAGE1_RECIPES = {
    "bracs": {"batch_size": 256, "steps": 600, "lr": 0.0001, "weight_decay": 0.0001},
    "tcga_ut": {"batch_size": 256, "steps": 2400, "lr": 0.0001, "weight_decay": 0.0001},
}


def protocol_grid(dataset: str) -> dict[str, dict[str, list[float | None]]]:
    """The fixed design grid of one dataset (a copy, safe to mutate)."""
    if dataset not in SPEC_GRIDS:
        raise ValueError(f"no protocol grid for dataset {dataset!r}")
    return copy.deepcopy(SPEC_GRIDS[dataset])


def init_seed(split_idx: int, draw_idx: int, base: int = INIT_SEED_BASE) -> int:
    """New-head initialization seed of one (split, draw)."""
    return base + 100 * split_idx + draw_idx


def normalized_grid(grid: dict[str, Any]) -> dict[str, dict[str, list[float | None]]]:
    """``{arm: {stage: {method: values}}}`` as ``{stage: {method: [float|None]}}``; ``r100`` only."""
    if set(grid) != {ARM}:
        raise ValueError(f"grid arms must be exactly {[ARM]}, got {sorted(grid)}")
    return {
        stage: {
            method: [None if v is None else float(v) for v in values]
            for method, values in methods.items()
        }
        for stage, methods in grid[ARM].items()
    }


def _source_slice(source: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    """The source experiment's own ``r100`` grid entries for every protocol candidate."""
    try:
        source_arm = source["mitigation"]["grid"][ARM]
        return {
            ARM: {
                stage: {m: source_arm[stage][m] for m in methods}
                for stage, methods in grid.items()
            }
        }
    except KeyError as error:
        raise ValueError(f"source grid lacks protocol candidate {error}") from error


def load_protocol_config(path: str | Path) -> dict[str, Any]:
    """Build the follow-up config: source dataset/training settings + restricted ``r100`` grid."""
    own = load_config(path)
    source_path = Path(own["source_config"])
    source = load_config(
        source_path if source_path.is_absolute() else REPO_ROOT / source_path
    )
    protocol = own["protocol"]
    config = copy.deepcopy(source)
    config["mitigation"]["grid"] = {ARM: copy.deepcopy(protocol["grid"])}
    config["paths"] = {"outputs": own["paths"]["outputs"]}
    config["source_outputs"] = source["paths"]["outputs"]
    config["source_config"] = str(source_path)
    config["source_grid"] = _source_slice(source, protocol["grid"])
    config["protocol"] = protocol
    return config


def _check_recipe(config: dict[str, Any], dataset: str) -> None:
    mitigation = config["mitigation"]
    stage1 = mitigation["train"]["stage1"]
    expected = STAGE1_RECIPES[dataset]
    if {k: float(stage1.get(k, -1)) for k in expected} != {
        k: float(v) for k, v in expected.items()
    }:
        raise ValueError(f"stage-one recipe differs from the protocol: {stage1}")
    cuda = mitigation.get("cuda", {})
    if {k: float(cuda.get(k, -1)) for k in CUDA_CONSTANTS} != {
        k: float(v) for k, v in CUDA_CONSTANTS.items()
    }:
        raise ValueError(f"CUDA recipe differs from the protocol: {cuda}")
    protocol = config["protocol"]
    if (
        float(protocol["margin_pp"]) != MARGIN_PP
        or int(protocol["init_seed_base"]) != INIT_SEED_BASE
    ):
        raise ValueError(
            "recipe: margin or initialization seed differs from the protocol"
        )


def validate_protocol(config: dict[str, Any]) -> None:
    """Reject any deviation from the fixed design: legacy ids/outputs, grids, recipes, margin, seed."""
    protocol = config["protocol"]
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError(
            f"legacy protocol id {protocol.get('id')!r}; need {PROTOCOL_ID!r}"
        )
    if "strong_mitigations_v2" not in Path(config["paths"]["outputs"]).parts:
        raise ValueError(
            "legacy output root: v2 outputs live under strong_mitigations_v2"
        )
    dataset = config["dataset"]["name"]
    expected = protocol_grid(dataset)
    if normalized_grid(config["mitigation"]["grid"]) != expected:
        raise ValueError("config grid differs from the protocol grid")
    if normalized_grid(config["source_grid"]) != expected:
        raise ValueError(
            "source experiment grid differs from the protocol grid (stale grid)"
        )
    _check_recipe(config, dataset)


def workload_counts(config: dict[str, Any]) -> dict[str, int]:
    """New work and source records: CPU fits (ce, la), CUDA fits, score-only candidates, records."""
    shards = N_SPLITS * N_DRAWS
    stage1 = [job.method for job in stage1_jobs(config)]
    cuda = stage1.count("cuda") * shards
    cpu = (len(stage1) - stage1.count("cuda")) * shards
    score = len(stage2_jobs(config)) * shards
    return {
        "cpu_fits": cpu,
        "cuda_fits": cuda,
        "score_candidates": score,
        "records": cpu + cuda + score,
        "source_records": cpu + cuda + score,
    }


def digest(payload: object) -> str:
    """SHA-256 of a canonical JSON rendering."""
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_digest(config: dict[str, Any]) -> str:
    """Fingerprint of everything that defines the candidates and their training recipe."""
    return digest([config["mitigation"], config["prevalence"], config["protocol"]])


def source_root(config: dict[str, Any]) -> Path:
    """Output root of the adapted experiment (41/43); never created or written here."""
    return output_root({"paths": {"outputs": config["source_outputs"]}})


def source_run_dir(
    config: dict[str, Any],
    split: int,
    arm: str,
    method: str,
    param: float | None,
    draw: int,
) -> Path:
    """One adapted run's result directory, matching ``mitigation.data.run_dir`` without creating it."""
    label = f"{arm}_{method_label(method, param)}"
    return source_root(config) / f"split={split}" / "results" / label / f"draw_{draw}"
