from __future__ import annotations

from pathlib import Path

import pytest
import torch

from imbalance_benchmark.commands import tuning
from imbalance_benchmark.commands.tuning import shard as tuning_shard
from imbalance_benchmark.commands.confirm import require_tuning_configs
from imbalance_benchmark.modeling.context import Regime
from imbalance_benchmark.modeling.workflows import tuning_aggregate
from imbalance_benchmark.modeling.workflows.tuning_aggregate import (
    TuningScope,
    _select_post_hoc,
    _select_trainable,
    summarize_tuning_cost,
)
from typing import Any
import argparse

def test_tuning_uses_the_frozen_initialization_seeds() -> None:
    """Post-freeze CLI seeds must not alter the two locked tuning repetitions."""
    freeze = {
        "seed_roles": {
            "tuning_initialization_0": 101,
            "tuning_initialization_1": 202,
        }
    }

    assert tuning._tuning_seeds(freeze) == [101, 202]

def test_tuning_uses_the_frozen_candidate_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-code grid change after freeze cannot affect configuration selection."""
    observed: list[dict[str, Any]] = []
    regime = Regime(
        torch.device("cpu"),
        {},
        2,
        False,
        method_grids={"ce": [{"lr": 0.123}]},
    )
    scope = TuningScope(regime, object(), object())

    def evaluate(
        _: str, cfg: dict[str, Any], *__: object
    ) -> tuple[dict[str, Any], dict[str, float]]:
        observed.append(cfg)
        return {}, {"balanced_accuracy": 0.5, "macro_f1": 0.5, "nll": 0.5}

    monkeypatch.setattr(
        "imbalance_benchmark.modeling.workflows.tuning_aggregate._evaluate", evaluate
    )

    _select_trainable("ce", [scope], [7])

    assert observed == [{"lr": 0.123}]

def test_select_post_hoc_persists_every_taus_averaged_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The signed selection must reproduce every tau's metrics, not only the winner."""
    regime = Regime(
        torch.device("cpu"),
        {},
        2,
        False,
        method_grids={
            "post_hoc_logit_adjustment": [{"parameter": p} for p in (0.5, 1.0, 2.0)]
        },
    )
    train_ds = type("DS", (), {"get_int_targets": lambda self: [0, 1, 0, 1]})()
    scope = TuningScope(regime, object(), train_ds)
    canned = {0.5: (0.6, 0.5, 0.9), 1.0: (0.8, 0.7, 0.5), 2.0: (0.4, 0.3, 1.2)}
    fake_model = type("Model", (), {"load_state_dict": lambda self, state: None})()

    def fake_run_evaluation(
        model: Any, val_loader: Any, device: Any, is_mil: Any, n_classes: Any,
        tau: float, priors: Any,
    ) -> dict[str, float]:
        ba, f1, nll = canned[tau]
        return {"balanced_accuracy": ba, "macro_f1": f1, "nll": nll}

    monkeypatch.setattr(tuning_aggregate, "_evaluate", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(
        tuning_aggregate, "build_training_ctx", lambda *a, **k: {"model": fake_model}
    )
    monkeypatch.setattr(tuning_aggregate, "run_evaluation", fake_run_evaluation)

    result = _select_post_hoc({}, [scope], [7])

    assert result["parameter"] == 1.0
    assert result["taus"] == {
        "0.5": {"balanced_accuracy": 0.6, "macro_f1": 0.5, "nll": 0.9},
        "1.0": {"balanced_accuracy": 0.8, "macro_f1": 0.7, "nll": 0.5},
        "2.0": {"balanced_accuracy": 0.4, "macro_f1": 0.3, "nll": 1.2},
    }

def test_tuning_cost_summarizes_parameter_counts_and_effective_passes() -> None:
    """Validation-search cost must expose its model size and realized data passes."""
    cost = summarize_tuning_cost(
        [
            {
                "processed_examples": 12,
                "unique_training_examples": 6,
                "total_parameters": 10,
                "trainable_parameters": 8,
                "training_footprint_parameters": 20,
            }
        ]
    )

    assert cost["effective_passes_through_unique_examples"] == 2.0
    assert cost["maximum_total_parameters"] == 10
    assert cost["maximum_trainable_parameters"] == 8
    assert cost["maximum_training_footprint_parameters"] == 20

def test_tuning_rejects_a_single_split_as_a_definitive_selection() -> None:
    args = argparse.Namespace(split_index=0, config=None, seed=0)

    with pytest.raises(ValueError, match="all three"):
        tuning.cmd_tune(args)

def test_tuning_selection_signed_lock_detects_tampering(tmp_path: Path) -> None:
    from imbalance_benchmark.common import sign_file, verify_signed_file, write_json

    selection = tmp_path / "tuning_selections.json"
    write_json(selection, {"native": {"severe": {"weighted_ce": {"lr": 1e-3}}}})
    sign_file(selection)

    verify_signed_file(selection)  # unaltered: passes

    write_json(selection, {"native": {"severe": {"weighted_ce": {"lr": 3e-3}}}})
    with pytest.raises(RuntimeError, match="no longer matches"):
        verify_signed_file(selection)

    unsigned = tmp_path / "tuning_selections_severe.json"
    write_json(unsigned, {})
    with pytest.raises(RuntimeError, match="no signed lock"):
        verify_signed_file(unsigned)

def test_missing_tuning_selection_stops_confirmation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing tuning selection"):
        require_tuning_configs(
            tmp_path, "severe", {"ce": {"lr": 1e-3}}, ("ce", "weighted_ce")
        )

def test_unresolved_tuning_lock_stops_confirmation(tmp_path: Path) -> None:
    from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
        merge_round_state,
    )

    merge_round_state(
        tmp_path, "severe", {"ce": {"resolved": False, "tuning_limited": False}}
    )
    with pytest.raises(RuntimeError, match="tuning lock unresolved"):
        require_tuning_configs(tmp_path, "severe", {"ce": {"lr": 1e-3}}, ("ce",))

def test_resolved_tuning_lock_admits_confirmation(tmp_path: Path) -> None:
    from imbalance_benchmark.modeling.workflows.tuning.candidate_registry import (
        merge_round_state,
    )

    merge_round_state(
        tmp_path, "severe", {"ce": {"resolved": True, "tuning_limited": False}}
    )
    configs = {"ce": {"lr": 1e-3}}
    assert require_tuning_configs(tmp_path, "severe", configs, ("ce",)) == configs

def test_run_shard_shares_one_cost_records_list_across_a_bundles_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bundle's cached scopes must share one cost_records list per shard.

    Regression: giving each cached scope its own fresh ``[]`` left
    ``scopes[0].cost_records`` empty whenever ``_fit_payload`` evaluated an
    observation on a different scope index, failing
    ``validate_shard_payload`` in production after a full natural-condition
    fit had already run.
    """
    regime = Regime(torch.device("cpu"), {}, 2, False)
    base_scopes = [
        TuningScope(regime, object(), object(), split_index=i) for i in range(3)
    ]
    monkeypatch.setattr(tuning_shard, "condition_is_reusable", lambda *_: False)
    monkeypatch.setattr(tuning_shard, "combined_scopes", lambda *_a, **_k: base_scopes)
    captured: list[list[TuningScope]] = []

    def fake_run_candidate_shard(spec, scopes, seeds, fingerprint, output_root, stage):
        del spec, seeds, fingerprint, output_root, stage
        captured.append(scopes)

    monkeypatch.setattr(tuning_shard, "run_candidate_shard", fake_run_candidate_shard)
    freeze = {
        "seed_roles": {"tuning_initialization_0": 1, "tuning_initialization_1": 2},
        "tail_assignments": {"native": []},
    }
    spec = tuning_shard.ShardSpec("natural", "ce", 0, "base", observation_index=0)
    built: dict = {}

    base = {"data": None}
    tuning_shard._run_shard(base, [(None, regime, None)], freeze, ["fp"], built, spec)
    tuning_shard._run_shard(base, [(None, regime, None)], freeze, ["fp"], built, spec)

    first, second = captured
    assert len({id(s.cost_records) for s in first}) == 1
    assert len({id(s.cost_records) for s in second}) == 1
    assert first[0].cost_records is not second[0].cost_records
    assert first[0].cost_records == [] and second[0].cost_records == []
