"""Four-arm allocation/weight identities, block seeds, and protocol config validation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from imbalance_benchmark.common import REPO_ROOT, load_config

from lora_decomposition.protocol import (
    arm_allocations,
    block_seed,
    load_protocol_config,
    loss_weights,
    validate_protocol,
    workload_counts,
)

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_four_arm_weight_identities():
    balanced = np.array([2, 2])
    skewed = np.array([3, 1])
    assert arm_allocations("B") == ("r1", "r1")
    assert arm_allocations("F") == ("r1", "r100")
    assert arm_allocations("C") == ("r100", "r1")
    assert arm_allocations("R") == ("r100", "r100")
    np.testing.assert_allclose(loss_weights(balanced, balanced), [1, 1])
    np.testing.assert_allclose(loss_weights(skewed, skewed), [1, 1])
    np.testing.assert_allclose(loss_weights(balanced, skewed), [1.5, 0.5])
    np.testing.assert_allclose(loss_weights(skewed, balanced), [2 / 3, 2])


def test_block_seeds_are_distinct():
    assert len({block_seed(s, d) for s in range(3) for d in range(10)}) == 30
    assert block_seed(0, 0) == 48000 and block_seed(2, 9) == 48209
    for bad in ((3, 0), (0, 10), (-1, 0)):
        with pytest.raises(ValueError):
            block_seed(*bad)


def test_invalid_arms_and_counts_are_rejected():
    with pytest.raises(ValueError):
        arm_allocations("X")
    for data, target in (([2, 0], [1, 1]), ([2, -1], [1, 1]), ([2, 2], [1, 1, 1])):
        with pytest.raises(ValueError):
            loss_weights(np.array(data), np.array(target))


@pytest.mark.parametrize("name,steps,patients", [("bracs", 600, 10), ("tcga_ut", 2400, 20)])
def test_protocol_config_keeps_source_and_recipe(name: str, steps: int, patients: int):
    own = load_config(_CONFIGS / f"{name}.yaml")
    source_path = REPO_ROOT / own["source_config"]
    source_before = load_config(source_path)
    config = load_protocol_config(_CONFIGS / f"{name}.yaml")
    assert load_config(source_path) == source_before
    assert config["mitigation"]["train"]["stage1"]["steps"] == steps
    assert config["prevalence"]["patients_per_class"] == patients
    assert "grid" not in config["mitigation"] and "inline_stage2" not in config["mitigation"]
    assert config["slurm"]["code_dir"].endswith("48_lora_decomposition/code")
    assert "48_lora_decomposition" in config["slurm"]["output_dir"]
    assert workload_counts(config) == {"blocks": 30, "fits": 120}


def test_protocol_rejects_altered_recipe_encoder_rho_and_output():
    config = load_protocol_config(_CONFIGS / "bracs.yaml")
    validate_protocol(config)
    edits = [
        lambda c: c["mitigation"]["train"]["stage1"].update(steps=300),
        lambda c: c["mitigation"]["lora"].update(r=4),
        lambda c: c["mitigation"].update(encoder="uni2h"),
        lambda c: c["protocol"].update(imbalanced_allocation="r50"),
        lambda c: c["paths"].update(outputs="experiments/elsewhere/outputs"),
        lambda c: c["prevalence"].update(patients_per_class=20),
    ]
    for edit in edits:
        changed = copy.deepcopy(config)
        edit(changed)
        with pytest.raises(ValueError):
            validate_protocol(changed)
