"""Tail-class join, tiering, and table-emit checks for the constructed report."""

import json
from pathlib import Path

import pandas as pd

from analysis.plotting.support.tail_class import (
    _native_tiers,
    _regime_param,
    tail_class_frame,
    write_tail_class_tables,
)


def test_native_tiers_top_and_bottom_eight() -> None:
    # native is sorted descending (most frequent first), so head=first 8, tail=last 8.
    native = pd.Series({f"c{i:02d}": 100 - i for i in range(32)})
    tiers = _native_tiers(native)
    assert tiers["c00"] == "head" and tiers["c07"] == "head"
    assert tiers["c08"] == "body" and tiers["c23"] == "body"
    assert tiers["c24"] == "tail" and tiers["c31"] == "tail"


def test_regime_param_handles_g_and_plain_formats() -> None:
    assert _regime_param("order=native_prevalence/param=1") == 1.0
    assert _regime_param("order=native_prevalence/param=0.5") == 0.5
    assert _regime_param("order=native_prevalence/param=1.5") == 1.5


def _write_run(root: Path, regime: str, method: str, seed: int, payload: dict) -> None:
    run = root / "tuning" / "wsi" / regime / method / "v0" / f"seed={seed}"
    run.mkdir(parents=True)
    (run / "test_results.json").write_text(json.dumps(payload), encoding="utf-8")


def test_tail_class_frame_joins_support_and_tier(tmp_path: Path) -> None:
    names = [f"c{i:02d}" for i in range(32)]
    native = pd.Series({name: 100 - i for i, name in enumerate(names)})
    constructed = tmp_path / "constructed"
    split = constructed / "constructed_order=native_prevalence_parameter=0.5_seed=0"
    split.mkdir(parents=True)
    (split / "target_counts.json").write_text(
        json.dumps({name: i + 1 for i, name in enumerate(names)}), encoding="utf-8"
    )
    results = tmp_path / "results"
    payload = {
        "class_names": names,
        "recall_per_class": [0.9] * 32,
        "f1_per_class": [0.8] * 32,
    }
    _write_run(results, "order=native_prevalence/param=0.5", "rankmix_mil", 0, payload)
    output = tmp_path / "output"
    output.mkdir()
    (output / "tuning_selection.json").write_text(
        json.dumps(
            [
                {
                    "benchmark": "wsi",
                    "regime": "order=native_prevalence/param=0.5",
                    "method": "rankmix_mil",
                    "variant": "v0",
                }
            ]
        ),
        encoding="utf-8",
    )

    frame = tail_class_frame(results, output, constructed, native)
    assert len(frame) == 32
    head = frame[frame["class_name"] == "c00"].iloc[0]
    assert head["tier"] == "head" and head["train_support"] == 1
    assert head["f1"] == 0.8 and head["recall"] == 0.9
    assert frame[frame["class_name"] == "c31"].iloc[0]["tier"] == "tail"

    write_tail_class_tables(frame, tmp_path)
    table = (tmp_path / "result_tail_class_wsi_bag.tex").read_text(encoding="utf-8")
    assert "RankMix" in table and "Head & Body & Tail" in table
