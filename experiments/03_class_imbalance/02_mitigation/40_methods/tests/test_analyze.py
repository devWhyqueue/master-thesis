"""Analyze-stage unit tests on synthetic run records (no BootstrapContext/dataset needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from imbalance_benchmark.common import RUN_RECORD_NAME, ensure_dirs, split_paths

from analyze.paths import fixed_dirs, run_dir
from analyze.report import Distributions, family_share, method_summary, pack_methods
from analyze.select import select, selected_frequency
from mitigation import method_label


def _config(tmp_path: Path) -> dict:
    return {
        "paths": {"outputs": str(tmp_path)},
        "mitigation": {"grid": {"r100": {"stage1": {"bs": [0.5, 1.0]}}}},
    }


def _write_validation_ba(paths: dict, arm: str, method: str, param: float | None, d: int, ba: float) -> None:
    out = run_dir(paths, arm, method, param, d)
    out.mkdir(parents=True, exist_ok=True)
    record = {"splits": {"validation": {"endpoints": {"patient_macro_balanced_accuracy": ba}}}}
    (out / RUN_RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")


def test_run_dir_label_matches_mitigation_data_convention(tmp_path: Path) -> None:
    """``analyze.paths.run_dir`` must point at exactly the directory ``mitigation.data.run_dir`` wrote."""
    paths = split_paths(ensure_dirs(_config(tmp_path)), 0)
    assert run_dir(paths, "r100", "bs", 0.5, 3).name == "draw_3"
    assert run_dir(paths, "r100", "bs", 0.5, 3).parent.name == f"r100_{method_label('bs', 0.5)}"
    assert run_dir(paths, "r100", "ce", None, 0).parent.name == "r100_ce"


def test_fixed_dirs_spans_every_shard(tmp_path: Path) -> None:
    """A fixed (arm, method, param) resolves to one directory per (split, draw) shard."""
    from centre import N_DRAWS, N_SPLITS

    paths = {s: split_paths(ensure_dirs(_config(tmp_path)), s) for s in range(N_SPLITS)}
    dirs = fixed_dirs(paths, "r100", "ce", None)
    assert len(dirs) == N_SPLITS * N_DRAWS
    assert dirs[(0, 0)] != dirs[(1, 0)]


def test_select_picks_best_validation_ba_independently_per_shard(tmp_path: Path) -> None:
    """Selection reads only the validation endpoint, and each shard chooses independently."""
    from analyze.paths import shard_keys
    from centre import N_SPLITS

    config = _config(tmp_path)
    paths = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    # Every shard favors bs=0.5, except (0, 1), which favors bs=1.0.
    for s, d in shard_keys():
        flipped = (s, d) == (0, 1)
        _write_validation_ba(paths[s], "r100", "bs", 0.5, d, ba=70.0 if flipped else 80.0)
        _write_validation_ba(paths[s], "r100", "bs", 1.0, d, ba=90.0 if flipped else 60.0)

    selection = select(config, paths, "r100", "stage1", "bs")

    assert selection[(0, 0)] == 0.5
    assert selection[(0, 1)] == 1.0
    assert selection[(1, 0)] == 0.5


def test_selected_frequency_counts_winning_params() -> None:
    """Frequency tally keys paramless/None selections as ``"null"`` and params with ``:g`` formatting."""
    freq = selected_frequency({(0, 0): 0.5, (0, 1): 0.5, (1, 0): None})
    assert freq == {"0.5": 2, "null": 1}


def test_method_summary_recovers_and_full_recovery_thresholds() -> None:
    """``recovers`` needs >=1pp with a CI excluding zero; ``full_recovery`` needs the r1 point inside the CI."""
    recovers = method_summary(
        "stage1",
        selected_est={"point": 75.0, "ci_2_5": 73.0, "ci_97_5": 77.0},
        recovery_est={"point": 5.0, "ci_2_5": 2.0, "ci_97_5": 8.0},
        frequency={"0.5": 30},
        r1_point=80.0,
    )
    assert recovers["recovers"] is True
    assert recovers["full_recovery"] is False  # 80.0 not within [73.0, 77.0]

    unresolved = method_summary(
        "stage2",
        selected_est={"point": 71.0, "ci_2_5": 69.0, "ci_97_5": 73.0},
        recovery_est={"point": 0.5, "ci_2_5": -1.0, "ci_97_5": 2.0},
        frequency={"null": 30},
        r1_point=80.0,
    )
    assert unresolved["recovers"] is False


def test_family_share_averages_point_estimates_by_family() -> None:
    """Family share is the mean of each family's method-level share *point* estimates."""
    methods = ("bs", "posthoc_la")
    family_of = {"bs": "stage1", "posthoc_la": "stage2"}
    share_dist = {"bs": np.array([0.4, 0.3, 0.5]), "posthoc_la": np.array([0.6, 0.5, 0.7])}

    shares = family_share(methods, family_of, share_dist)

    assert shares == {"stage1": 0.4, "stage2": 0.6}


def test_pack_methods_writes_one_estimate_triple_per_method() -> None:
    """Every method contributes selected/recovery/share estimates plus one verdict row."""
    dists = Distributions(
        ba={"r1_ce": np.array([80.0, 81.0, 79.0]), "r100_ce": np.array([70.0, 69.0, 71.0])},
        damage=np.array([10.0, 12.0, 8.0]),
        ba_selected={"bs": np.array([75.0, 76.0, 74.0])},
        recovery={"bs": np.array([5.0, 7.0, 3.0])},
        share={"bs": np.array([0.5, 7 / 12, 3 / 8])},
        frequency={"bs": {"0.5": 30}},
    )

    estimates, methods_out = pack_methods(("bs",), {"bs": "stage1"}, dists, "r1_ce", "r100_ce")

    assert estimates["damage"]["point"] == 10.0
    assert {"selected_bs", "recovery_bs", "share_bs"} <= estimates.keys()
    assert methods_out["bs"]["family"] == "stage1"


def test_specific_recovery_subtracts_balanced_gain() -> None:
    """Imbalance-specific recovery is ``R - G``, and its share divides by the CE damage."""
    from analyze.balanced import specific_estimates

    gains = {"cuda": {"gain": np.array([1.0, 0.5, 1.5])}}
    recovery = {"cuda": np.array([3.0, 2.5, 3.5]), "la": np.zeros(3)}
    out = specific_estimates(gains, recovery, damage=np.array([4.0, 4.0, 4.0]))
    assert set(out) == {"specific_recovery_cuda", "specific_share_cuda"}
    assert out["specific_recovery_cuda"]["point"] == 2.0
    assert out["specific_share_cuda"]["point"] == 0.5
