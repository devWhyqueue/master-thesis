"""Pilot run-record readers shared by every gate: raw balanced accuracy, B/P/S/R damage, and the
BRACS joint-rescue contrast (damage fall, R-arm rise, B-arm drop) for one (split, draw).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import ensure_dirs, read_run_record, split_paths

from sites import allocation_dir

from joint import ARMS

__all__ = [
    "NATIVE_REF_SETTING",
    "NATIVE_REF_DRAW",
    "allocation_out_dir",
    "record",
    "val_ba",
    "test_ba",
    "val_damage",
    "rescue_contrast",
]

NATIVE_REF_SETTING = {"bracs": "native_bracs_replay", "tcga_ut": "tcga_native_10"}
NATIVE_REF_DRAW = (
    0  # exp-34's own pilot draw; its main phase never ran (gate failed upstream)
)


def allocation_out_dir(
    config: dict[str, Any], allocation: str, split_idx: int, draw_idx: int
) -> Path:
    """This (allocation, split, draw)'s result directory."""
    paths = split_paths(ensure_dirs(config), split_idx)
    return allocation_dir(paths, allocation, draw_idx)


def record(
    config: dict[str, Any], allocation: str, split_idx: int, draw_idx: int
) -> dict[str, Any]:
    """This (allocation, split, draw)'s stored run record."""
    rec = read_run_record(
        allocation_out_dir(config, allocation, split_idx, draw_idx),
        splits=("validation", "test"),
        array_fields=(),
    )
    if rec is None:
        raise RuntimeError(
            f"Missing pilot run record: {allocation} split={split_idx} draw={draw_idx}"
        )
    return rec


def val_ba(
    config: dict[str, Any], allocation: str, split_idx: int, draw_idx: int
) -> float:
    """This (allocation, split, draw)'s validation patient-macro balanced accuracy, in pp."""
    rec = record(config, allocation, split_idx, draw_idx)
    return (
        float(
            rec["splits"]["validation"]["endpoints"]["patient_macro_balanced_accuracy"]
        )
        * 100.0
    )


def test_ba(
    config: dict[str, Any], allocation: str, split_idx: int, draw_idx: int
) -> float:
    """This (allocation, split, draw)'s test patient-macro balanced accuracy, in pp."""
    rec = record(config, allocation, split_idx, draw_idx)
    return (
        float(rec["splits"]["test"]["endpoints"]["patient_macro_balanced_accuracy"])
        * 100.0
    )


def val_damage(
    config: dict[str, Any], setting: str, split_idx: int, draw_idx: int
) -> dict[str, float]:
    """This setting's B/D_P/D_S/D_R, from validation balanced accuracy across its four arms."""
    b, p, s, r = (
        val_ba(config, f"{setting}_{arm}", split_idx, draw_idx) for arm in ARMS
    )
    return {"B": b, "D_P": b - p, "D_S": b - s, "D_R": b - r}


def rescue_contrast(
    config: dict[str, Any], split_idx: int, draw_idx: int
) -> dict[str, float]:
    """Native-minus-joint damage fall, R-arm rise, and B-arm drop for one (split, draw)."""
    native = val_damage(config, "native", split_idx, draw_idx)
    joint = val_damage(config, "joint", split_idx, draw_idx)
    joint_r = val_ba(config, "joint_R", split_idx, draw_idx)
    native_r = val_ba(config, "native_R", split_idx, draw_idx)
    return {
        "damage_fall_pp": native["D_R"] - joint["D_R"],
        "r_rise_pp": joint_r - native_r,
        "b_drop_pp": native["B"] - joint["B"],
    }
