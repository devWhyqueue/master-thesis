from __future__ import annotations

from pathlib import Path

import pytest
import torch

from imbalance_benchmark.analysis.query import load_seed_predictions
from imbalance_benchmark.commands import confirm
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    combined_scopes,
)
from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
    merge_round_state,
)
from imbalance_benchmark.commands.confirm import shard as confirm_shard
from imbalance_benchmark.common import write_run_record
from imbalance_benchmark.datasets.bracs import LABELS as BRACS_LABELS
from imbalance_benchmark.modeling.context import (
    GRIDS,
    LEARNING_RATE_GRID,
    get_grid_configs,
    roster_for_regime,
)
from imbalance_benchmark.modeling.workflows.tuning.search_windows import (
    LR_ENVELOPE,
    initial_window,
    shift_window,
    winner_is_interior,
)
from imbalance_benchmark.modeling.context import Regime
from imbalance_benchmark.modeling.workflows.confirmation import (
    RunContext,
    confirm_crt_seed,
    confirm_method,
)


def _write_seed_record(
    method_dir: Path,
    seed_idx: int,
    tuning_params: dict[str, object] | None = None,
) -> None:
    write_run_record(
        method_dir / f"seed={seed_idx}",
        {
            "benchmark": "patch",
            "class_names": ["A", "B"],
            "tuning_params": tuning_params or {},
            "splits": {
                "test": {
                    "labels": [0, 1],
                    "preds": [0, 1],
                    "probabilities": [[0.9, 0.1], [0.2, 0.8]],
                    "logits": [[2.0, 0.0], [0.0, 2.0]],
                }
            },
        },
    )


def test_train_time_logit_adjustment_confirmation_preserves_selected_tau(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Confirmation must evaluate train-time logit adjustment with its tuned tau."""
    observed: list[float] = []
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation.build_training_ctx",
        lambda *_: {"seed": 7},
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._timed_fit",
        lambda *_: ({}, 0.0),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._run_and_record",
        lambda *args: observed.append(float(args[7])),
    )
    run = RunContext(
        device=torch.device("cpu"),
        config={},
        n_classes=2,
        is_mil=False,
        class_names=["A", "B"],
        val_loader=object(),
        test_loader=object(),
        paths={"data": tmp_path, "results": tmp_path},
        seeds=[7],
        assignment="native",
    )

    confirm_method("severe", "logit_adjustment", {"parameter": 0.25}, object(), run)

    assert observed == [0.25]


def test_crt_records_its_stage_one_ce_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._training_context",
        lambda *args: {"param_config": args[-1]},
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._timed_fit",
        lambda *_: ({}, 0.0),
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.confirmation._run_and_record",
        lambda *args: observed.append(args[3]["param_config"]),
    )
    run = RunContext(
        device=torch.device("cpu"),
        config={},
        n_classes=2,
        is_mil=False,
        class_names=["A", "B"],
        val_loader=object(),
        test_loader=object(),
        paths={"data": tmp_path, "results": tmp_path},
        seeds=[7],
        assignment="native",
    )

    confirm_crt_seed("balanced", {"lr": 1e-3}, {"lr": 3e-5}, object(), run, seed_idx=0)

    assert observed == [{"lr": 1e-3, "stage_one": {"lr": 3e-5}}]


def test_confirmation_condition_uses_the_frozen_class_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    locked = list(BRACS_LABELS)
    observed: list[list[str] | None] = []

    def load_dataset(*_: object, **kwargs: object) -> object:
        observed.append(kwargs.get("class_names"))
        return object()

    monkeypatch.setattr(confirm, "load_training_dataset", load_dataset)
    monkeypatch.setattr(confirm, "confirm_ce", lambda *args: [])
    split_data = tmp_path / "split=0" / "data"
    merge_round_state(
        tmp_path / "data",
        "moderate",
        {"ce": {"resolved": True, "tuning_limited": False}},
    )
    run = RunContext(
        device=torch.device("cpu"),
        config={},
        n_classes=len(locked),
        is_mil=False,
        class_names=locked,
        val_loader=object(),
        test_loader=object(),
        paths={"data": split_data},
        seeds=[],
        assignment="native",
    )

    confirm._confirm_condition("moderate", ("ce",), {"ce": {}}, run)

    assert observed == [locked]


def test_unfitted_seed_is_not_done(tmp_path: Path) -> None:
    paths = {"results": tmp_path}

    assert not confirm_shard._seed_already_done(
        paths, "native", "severe", "weighted_ce", 0, {"weighted_ce": {}}, False
    )


def test_ordinary_method_is_done_once_its_test_record_exists(tmp_path: Path) -> None:
    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    config = {"lr": 3e-5, "parameter": 1.0}
    _write_seed_record(method_dir, 0, config)

    assert confirm_shard._seed_already_done(
        paths, "native", "severe", "weighted_ce", 0, {"weighted_ce": config}, False
    )


def test_changed_tuning_config_invalidates_an_existing_seed(tmp_path: Path) -> None:
    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    _write_seed_record(method_dir, 0, {"lr": 1e-4, "parameter": 1.0})

    assert not confirm_shard._seed_already_done(
        paths,
        "native",
        "severe",
        "weighted_ce",
        0,
        {"weighted_ce": {"lr": 3e-5, "parameter": 1.0}},
        False,
    )


def test_ce_unit_is_not_done_until_its_folded_post_hoc_record_also_exists(
    tmp_path: Path,
) -> None:
    """post-hoc rides with its seed's ce fit, so resuming a ce unit must not skip
    it on ce's record alone -- the folded post-hoc pass would be silently lost."""
    paths = {"results": tmp_path}
    configs = {
        "ce": {"lr": 3e-5},
        "post_hoc_logit_adjustment": {"parameter": 0.5, "taus": {}},
    }
    _write_seed_record(
        tmp_path / "assignment=native" / "severe" / "ce", 0, configs["ce"]
    )

    assert not confirm_shard._seed_already_done(
        paths, "native", "severe", "ce", 0, configs, False
    )

    _write_seed_record(
        tmp_path / "assignment=native" / "severe" / "post_hoc_logit_adjustment",
        0,
        {"parameter": 0.5},
    )

    assert confirm_shard._seed_already_done(
        paths, "native", "severe", "ce", 0, configs, False
    )


def test_crt_requires_its_selected_stage_one_ce_config(tmp_path: Path) -> None:
    paths = {"results": tmp_path}
    configs = {"crt": {"lr": 1e-3}, "ce": {"lr": 3e-5}}
    method_dir = tmp_path / "assignment=native" / "balanced" / "crt"
    _write_seed_record(method_dir, 0, configs["crt"])

    assert not confirm_shard._seed_already_done(
        paths, "native", "balanced", "crt", 0, configs, False
    )

    _write_seed_record(method_dir, 0, {**configs["crt"], "stage_one": configs["ce"]})

    assert confirm_shard._seed_already_done(
        paths, "native", "balanced", "crt", 0, configs, False
    )


def test_a_truncated_run_record_is_treated_as_not_done(tmp_path: Path) -> None:
    """A crash mid-write can leave a corrupt run.json; resuming must refit rather
    than trust it, or the seed would be silently dropped from confirmation."""
    paths = {"results": tmp_path}
    result_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce" / "seed=0"
    result_dir.mkdir(parents=True)
    (result_dir / "run.json").write_text("{not valid json")

    assert not confirm_shard._seed_already_done(
        paths, "native", "severe", "weighted_ce", 0, {"weighted_ce": {}}, False
    )


def test_combined_tuning_scope_passes_the_frozen_class_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    locked = list(BRACS_LABELS)
    regime = Regime(
        torch.device("cpu"), {}, len(locked), False, locked_class_names=locked
    )
    observed: list[list[str] | None] = []

    def load_dataset(*_: object, **kwargs: object) -> object:
        observed.append(kwargs.get("class_names"))
        return object()

    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.tuning.tuning_schedule.load_training_dataset",
        load_dataset,
    )
    combined_scopes([({"data": tmp_path}, regime, object())], "moderate", ("native",))

    assert observed == [locked]


def test_combined_scopes_assigns_distinct_scope_index_per_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each scope must get a unique scope_index; a stale default of 0 for every
    scope makes _validate_observations see duplicate (scope_index, seed_index)
    pairs and reject an otherwise-complete round-shard payload."""
    locked = list(BRACS_LABELS)
    regime = Regime(
        torch.device("cpu"), {}, len(locked), False, locked_class_names=locked
    )
    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.tuning.tuning_schedule.load_training_dataset",
        lambda *_, **__: object(),
    )
    raw_scopes = [({"data": tmp_path}, regime, object()) for _ in range(3)]

    scopes = combined_scopes(raw_scopes, "moderate", ("native",))

    assert [scope.scope_index for scope in scopes] == [0, 1, 2]
    assert [scope.split_index for scope in scopes] == [0, 1, 2]


def test_roster_for_regime_matches_report_table():
    patch = roster_for_regime(False)
    wsi = roster_for_regime(True)
    shared = {
        "ce",
        "balanced_sampling",
        "weighted_ce",
        "focal",
        "logit_adjustment",
        "post_hoc_logit_adjustment",
        "crt",
        "class_balanced_ce",
        "pilot_difficulty_ce",
    }
    assert shared <= set(patch) and shared <= set(wsi)
    assert set(patch) - shared == {
        "ce_soft_f1",
        "ce_soft_mcc",
        "cfal",
        "oko",
        "independent_support_ce",
        "semantic_scale_ce",
    }
    assert set(wsi) - shared == {"rankmix", "sc_mil", "mde"}


def test_ce_and_crt_grids_are_lr_only():
    assert get_grid_configs("ce") == [{"lr": lr} for lr in LEARNING_RATE_GRID]
    assert get_grid_configs("crt") == [{"lr": lr} for lr in LEARNING_RATE_GRID]


def test_post_hoc_grid_has_no_learning_rate():
    configs = get_grid_configs("post_hoc_logit_adjustment")
    assert configs == [{"parameter": p} for p in GRIDS["post_hoc_logit_adjustment"]]
    assert all("lr" not in c for c in configs)


def test_weighted_ce_grid_crosses_16_configurations():
    configs = get_grid_configs("weighted_ce")
    assert len(configs) == 16
    assert {c["parameter"] for c in configs} == set(GRIDS["weighted_ce"])
    assert {c["lr"] for c in configs} == set(LEARNING_RATE_GRID)


def test_oko_grid_capped_by_k_minus_1():
    configs = get_grid_configs("oko", n_classes=3)
    assert max(c["parameter"] for c in configs) <= 2
    configs_binary = get_grid_configs("oko", n_classes=2)
    assert {c["parameter"] for c in configs_binary} == {1}


def test_get_grid_configs_honors_an_explicit_lr_window():
    shifted = LR_ENVELOPE[3:7]
    configs = get_grid_configs("ce", lr_window=shifted)
    assert configs == [{"lr": lr} for lr in shifted]


def test_initial_window_positions_by_regime():
    assert initial_window("low", LR_ENVELOPE) == LR_ENVELOPE[0:4]
    assert initial_window("current", LR_ENVELOPE) == LEARNING_RATE_GRID
    assert initial_window("high", LR_ENVELOPE) == LR_ENVELOPE[3:7]


def test_initial_window_rejects_unknown_regime():
    with pytest.raises(ValueError):
        initial_window("mid", LR_ENVELOPE)


def test_winner_is_interior_true_only_off_the_edges():
    window = LEARNING_RATE_GRID
    assert winner_is_interior(window, window[1])
    assert winner_is_interior(window, window[2])
    assert not winner_is_interior(window, window[0])
    assert not winner_is_interior(window, window[-1])


def test_shift_window_reuses_three_overlapping_values_outward():
    window = LEARNING_RATE_GRID  # LR_ENVELOPE[2:6]
    shifted = shift_window(window, window[-1], LR_ENVELOPE)
    assert shifted == LR_ENVELOPE[3:7]
    assert set(window) & set(shifted) == set(window[1:])


def test_shift_window_reuses_three_overlapping_values_inward_direction():
    window = LEARNING_RATE_GRID
    shifted = shift_window(window, window[0], LR_ENVELOPE)
    assert shifted == LR_ENVELOPE[1:5]
    assert set(window) & set(shifted) == set(window[:-1])


def test_shift_window_exhausts_at_the_envelope_boundary():
    top_window = LR_ENVELOPE[-4:]
    assert shift_window(top_window, top_window[-1], LR_ENVELOPE) is None
    bottom_window = LR_ENVELOPE[:4]
    assert shift_window(bottom_window, bottom_window[0], LR_ENVELOPE) is None


def test_shift_window_rejects_an_interior_winner():
    window = LEARNING_RATE_GRID
    with pytest.raises(ValueError):
        shift_window(window, window[1], LR_ENVELOPE)


def test_repeated_shifts_never_duplicate_a_previously_evaluated_column():
    window = initial_window("low", LR_ENVELOPE)
    seen = set(window)
    for _ in range(len(LR_ENVELOPE) - 4):
        shifted = shift_window(window, window[-1], LR_ENVELOPE)
        assert shifted is not None
        new_columns = set(shifted) - seen
        assert new_columns == {shifted[-1]}
        seen |= new_columns
        window = shifted
    assert shift_window(window, window[-1], LR_ENVELOPE) is None


def test_confirmation_provenance_payload_carries_appendix_a_fields() -> None:
    """The run record's provenance carries the Appendix A fields flagged as missing.

    Finding: "Required run provenance is incomplete" — model/optimizer config,
    candidate grid, freeze hash, dataset version, achieved T/rho, and pilot quota.
    """
    from imbalance_benchmark.modeling.workflows.confirmation_helpers import (
        RunContext,
        _provenance_payload,
    )

    run = RunContext(
        device=torch.device("cpu"),
        config={},
        n_classes=2,
        is_mil=False,
        val_loader=None,  # type: ignore[arg-type]
        test_loader=None,  # type: ignore[arg-type]
        paths={},
        seeds=[0],
        class_names=["A", "B"],
        assignment="native",
    )
    freeze = {
        "content_sha256": "freezehash",
        "dataset_provenance": {"version": "v1"},
        "shared_T": 100,
        "min_support": 20,
        "method_grids": {"weighted_ce": [{"lr": 1e-4, "parameter": 1.0}]},
        "assignment_conditions": {"native": {"severe": {"achieved_rho": 99.0}}},
    }

    out = _provenance_payload(run, "severe", "weighted_ce", freeze)
    assert out["model_optimizer_config"]["optimizer"] == "AdamW"
    assert out["model_optimizer_config"]["weight_decay"] == 1e-4
    assert out["candidate_grid"] == [{"lr": 1e-4, "parameter": 1.0}]
    assert out["freeze_content_sha256"] == "freezehash"
    assert out["dataset_version"] == "v1"
    assert out["achieved_T"] == 100
    assert out["achieved_rho"] == 99.0
    assert out["pilot_min_support"] == 20


def test_partial_confirmation_block_is_rejected(tmp_path: Path) -> None:
    """Fewer than five valid confirmation seeds must stop inference, not be averaged."""
    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    for seed_idx in range(3):  # only three of the required five present
        _write_seed_record(method_dir, seed_idx)

    with pytest.raises(RuntimeError, match="incomplete"):
        load_seed_predictions(paths, "severe", "weighted_ce", "native")


def test_complete_confirmation_block_stacks_all_five_seeds(tmp_path: Path) -> None:
    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    for seed_idx in range(5):
        _write_seed_record(method_dir, seed_idx)

    stacked = load_seed_predictions(paths, "severe", "weighted_ce", "native")

    assert stacked is not None
    assert stacked["preds"].shape[0] == 5


def test_confirmation_block_reads_each_valid_seed_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from imbalance_benchmark.analysis import query

    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    for seed_idx in range(5):
        _write_seed_record(method_dir, seed_idx)
    calls: list[Path] = []
    original_read = query.read_run_record

    def tracking_read(*args: object, **kwargs: object) -> dict[str, object] | None:
        calls.append(args[0])  # type: ignore[arg-type]
        return original_read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(query, "read_run_record", tracking_read)

    stacked = load_seed_predictions(paths, "severe", "weighted_ce", "native")

    assert stacked is not None
    assert calls == [method_dir / f"seed={seed_idx}" for seed_idx in range(5)]


def test_confirmation_block_rejects_unreadable_seed(tmp_path: Path) -> None:
    paths = {"results": tmp_path}
    method_dir = tmp_path / "assignment=native" / "severe" / "weighted_ce"
    for seed_idx in range(5):
        _write_seed_record(method_dir, seed_idx)
    (method_dir / "seed=3" / "run.json").unlink()

    with pytest.raises(RuntimeError, match=r"missing/failed \[3\]"):
        load_seed_predictions(paths, "severe", "weighted_ce", "native")


def test_missing_confirmation_method_is_not_silently_skipped(tmp_path: Path) -> None:
    """A roster method with no directory is a failed confirmation block."""
    with pytest.raises(RuntimeError, match="missing"):
        load_seed_predictions({"results": tmp_path}, "severe", "weighted_ce", "native")
