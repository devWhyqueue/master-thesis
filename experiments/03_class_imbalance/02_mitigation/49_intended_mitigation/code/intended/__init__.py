"""Protocol constants, config loading and source paths for the intended-mitigation study (exp-49).

Frozen-feature methods (CE, LA, BS, DisAlign) are refit here with exp-25's logistic regression;
CE_SGD and CUDA r1 run through exp-45's frozen-encoder trainer; cRT, GCL and GCL2 (LoRA) and
CUDA r100 are read from exp-41/43/45 records. The merged config is the LoRA source config
(dataset, recipes, CUDA constants) with this study's grid, keyed ``arm -> stage -> method`` with
stages ``lr`` (logistic regression) and ``sgd`` (frozen-encoder SGD), so exp-40's
``analyze.select`` addresses own records directly.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import copy
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import REPO_ROOT, load_config, output_root

from centre import N_SPLITS

from sites import allocation_dir

from prevalence import patients_per_class

from lora_ablation import CUDA_CONSTANTS, STAGE1_RECIPES

__all__ = [
    "ARMS",
    "DATASETS",
    "LR_METHODS",
    "PROTOCOL_ID",
    "REPORT_DIR",
    "REUSED_LORA",
    "SGD_METHODS",
    "SOURCE_CONFIGS",
    "SPEC_GRIDS",
    "STAGE2_RECIPE",
    "cuda_r100_dirs",
    "damage_dir",
    "load_protocol_config",
    "lora_dirs",
    "lr_paths",
    "sgd_paths",
    "validate_protocol",
]

PROTOCOL_ID = "intended-mitigation-v1"
DATASETS = ("bracs", "tcga_ut")
ARMS = ("r1", "r100")
LR_METHODS = ("ce", "la", "bs", "disalign")
SGD_METHODS = ("ce", "cuda")
REUSED_LORA = ("crt", "gcl", "gcl2")
STAGE2_RECIPE = {"batch_size": 512, "steps": 200, "lr": 0.001, "weight_decay": 0.0}
REPORT_DIR = Path(__file__).resolve().parents[2] / "report"

_EXP = "experiments/03_class_imbalance"
SOURCE_CONFIGS: dict[str, dict[str, str]] = {
    "bracs": {
        "damage_config": f"{_EXP}/00_damage/26_damage_bracs/configs/bracs.yaml",
        "mitigation_config": f"{_EXP}/02_mitigation/41_mitigation_bracs/configs/bracs.yaml",
        "frozen_cuda_config": f"{_EXP}/02_mitigation/45_lora_ablation/configs/bracs.yaml",
    },
    "tcga_ut": {
        "damage_config": f"{_EXP}/00_damage/25_damage_tcga_ut/configs/tcga_ut.yaml",
        "mitigation_config": f"{_EXP}/02_mitigation/43_mitigation_tcga_ut/configs/tcga_ut.yaml",
        "frozen_cuda_config": f"{_EXP}/02_mitigation/45_lora_ablation/configs/tcga_ut.yaml",
    },
}
_SOURCE_KEYS = ("damage_config", "mitigation_config", "frozen_cuda_config")


def _grid(
    la: list[float | None],
) -> dict[str, dict[str, dict[str, list[float | None]]]]:
    """One dataset's design grid: r1 anchors (CE, DisAlign gamma 1, CUDA r1) and r100 mitigations."""
    return {
        "r1": {
            "lr": {"ce": [None], "disalign": [1.0]},
            "sgd": {"ce": [None], "cuda": [0.25, 0.5, 1.0]},
        },
        "r100": {
            "lr": {
                "ce": [None],
                "la": la,
                "bs": [0.5, 1.0],
                "disalign": [0.5, 1.0, 1.5, 2.0],
            },
            "sgd": {"ce": [None]},
        },
    }


SPEC_GRIDS: dict[str, dict[str, dict[str, dict[str, list[float | None]]]]] = {
    "bracs": _grid([0.5, 1.0, 1.5, 2.0]),
    "tcga_ut": _grid([0.25, 0.5, 1.0, 1.5, 2.0]),
}


def _normalized(tree: Any) -> Any:
    """Nested grid dicts with every leaf list coerced to ``float | None``."""
    if isinstance(tree, dict):
        return {key: _normalized(value) for key, value in tree.items()}
    return [None if value is None else float(value) for value in tree]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_protocol_config(path: str | Path) -> dict[str, Any]:
    """Own YAML merged onto the LoRA source config, with every source config resolved and recorded."""
    own = load_config(path)
    sources = {key: _resolve(own[key]) for key in _SOURCE_KEYS}
    damage, mitigation, frozen = (load_config(sources[key]) for key in _SOURCE_KEYS)
    config = copy.deepcopy(mitigation)
    config.update({key: str(value) for key, value in sources.items()})
    config["feature_extraction"] = copy.deepcopy(damage["feature_extraction"])
    config["paths"] = {"outputs": own["paths"]["outputs"]}
    config["slurm"] = {**mitigation["slurm"], **own.get("slurm", {})}
    config["mitigation"]["grid"] = _normalized(own["protocol"]["grid"])
    config["protocol"] = copy.deepcopy(own["protocol"])
    config["source_outputs"] = {
        "damage": damage["paths"]["outputs"],
        "mitigation": mitigation["paths"]["outputs"],
        "frozen_cuda": frozen["paths"]["outputs"],
    }
    config["sources"] = {
        "mitigation_grid": _normalized(mitigation["mitigation"]["grid"]),
        "frozen_cuda_grid": _normalized(frozen["protocol"]["grid"]),
        "damage_design": {
            "patients_per_class": patients_per_class(damage),
            "exp2_outputs": damage["slurm"]["exp2_outputs"],
        },
    }
    return config


def _check_sources(config: dict[str, Any], dataset: str) -> None:
    for key, relative in SOURCE_CONFIGS[dataset].items():
        if Path(config[key]) != REPO_ROOT / relative:
            raise ValueError(f"source config {key} is {config[key]}, need {relative}")
    design = config["sources"]["damage_design"]
    if design != {
        "patients_per_class": patients_per_class(config),
        "exp2_outputs": config["slurm"]["exp2_outputs"],
    }:
        raise ValueError(f"source designs differ: damage {design} vs mitigation config")
    source = config["sources"]["mitigation_grid"]
    found = {
        "la": source["r100"]["stage2"]["posthoc_la"],
        "bs": source["r100"]["stage1"]["bs"],
        "disalign_r100": source["r100"]["stage2"]["disalign"],
        "disalign_r1": source["r1"]["stage2"]["disalign"],
        "cuda": config["sources"]["frozen_cuda_grid"]["stage1"]["cuda"],
    }
    spec = SPEC_GRIDS[dataset]
    expected = {
        "la": spec["r100"]["lr"]["la"],
        "bs": spec["r100"]["lr"]["bs"],
        "disalign_r100": spec["r100"]["lr"]["disalign"],
        "disalign_r1": spec["r1"]["lr"]["disalign"],
        "cuda": spec["r1"]["sgd"]["cuda"],
    }
    if found != expected:
        raise ValueError(
            f"source experiment grid differs from the protocol grid: {found}"
        )


def _same(found: dict[str, Any], expected: dict[str, Any]) -> bool:
    return {k: float(found.get(k, -1)) for k in expected} == {
        k: float(v) for k, v in expected.items()
    }


def _check_recipe(config: dict[str, Any], dataset: str) -> None:
    mitigation = config["mitigation"]
    train = mitigation["train"]
    if not _same(train["stage1"], STAGE1_RECIPES[dataset]):
        raise ValueError(f"SGD recipe differs from the protocol: {train['stage1']}")
    if not _same(train["stage2"], STAGE2_RECIPE):
        raise ValueError(
            f"DisAlign recipe differs from the protocol: {train['stage2']}"
        )
    if not _same(mitigation.get("cuda", {}), CUDA_CONSTANTS):
        raise ValueError(
            f"CUDA recipe differs from the protocol: {mitigation.get('cuda')}"
        )


def validate_protocol(config: dict[str, Any]) -> None:
    """Reject any deviation from the fixed design: id, source configs, grids, recipes."""
    if config["protocol"].get("id") != PROTOCOL_ID:
        raise ValueError(
            f"protocol id {config['protocol'].get('id')!r}; need {PROTOCOL_ID!r}"
        )
    dataset = config["dataset"]["name"]
    if config["mitigation"]["grid"] != SPEC_GRIDS[dataset]:
        raise ValueError("config grid differs from the protocol grid")
    _check_sources(config, dataset)
    _check_recipe(config, dataset)


def _split_dirs(root: Path) -> dict[int, dict[str, Path]]:
    """Per-split ``root``/``results`` directories, shaped like exp-40 ``result_paths``; never created."""
    return {
        s: {"root": root / f"split={s}", "results": root / f"split={s}" / "results"}
        for s in range(N_SPLITS)
    }


def _source_root(config: dict[str, Any], key: str) -> Path:
    return output_root({"paths": {"outputs": config["source_outputs"][key]}})


def lr_paths(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """Own logistic-regression records (CE refit, LA, BS, DisAlign) per split."""
    return _split_dirs(output_root(config) / "lr")


def sgd_paths(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """Own frozen-encoder SGD records (CE_SGD, CUDA r1) per split."""
    return _split_dirs(output_root(config) / "sgd")


def damage_dir(config: dict[str, Any], split: int, arm: str, draw: int) -> Path:
    """The stored exp-25/26 CE record of one (split, arm, draw)."""
    return allocation_dir(_split_dirs(_source_root(config, "damage"))[split], arm, draw)


def lora_dirs(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """exp-41/43 result roots per split (cRT, GCL, GCL2, LoRA CE)."""
    return _split_dirs(_source_root(config, "mitigation"))


def cuda_r100_dirs(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """exp-45 result roots per split (frozen CUDA r100, CPU CE_SGD r100)."""
    return _split_dirs(_source_root(config, "frozen_cuda"))
