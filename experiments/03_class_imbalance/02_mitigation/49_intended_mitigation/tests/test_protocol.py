"""Protocol constants, configs and source paths of exp-49."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from intended import (
    DATASETS,
    SPEC_GRIDS,
    cuda_r100_dirs,
    damage_dir,
    load_protocol_config,
    lora_dirs,
    lr_paths,
    sgd_paths,
    validate_protocol,
)

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _config(dataset: str) -> dict:
    return load_protocol_config(_CONFIGS / f"{dataset}.yaml")


@pytest.mark.parametrize("dataset", DATASETS)
def test_spec_grids_match_sources(dataset: str) -> None:
    config = _config(dataset)
    validate_protocol(config)
    source = config["sources"]["mitigation_grid"]
    spec = SPEC_GRIDS[dataset]
    assert spec["r100"]["lr"]["la"] == source["r100"]["stage2"]["posthoc_la"]
    assert spec["r100"]["lr"]["disalign"] == source["r100"]["stage2"]["disalign"]
    assert spec["r1"]["lr"]["disalign"] == source["r1"]["stage2"]["disalign"]
    assert spec["r100"]["lr"]["bs"] == [0.5, 1.0]
    assert spec["r1"]["sgd"]["cuda"] == config["sources"]["frozen_cuda_grid"]["stage1"]["cuda"]
    assert config["mitigation"]["grid"] == spec


@pytest.mark.parametrize(
    ("arm", "stage", "method"),
    [("r100", "lr", "la"), ("r100", "lr", "bs"), ("r1", "lr", "disalign"), ("r1", "sgd", "cuda")],
)
def test_validate_rejects_changed_grid(arm: str, stage: str, method: str) -> None:
    config = copy.deepcopy(_config("bracs"))
    config["mitigation"]["grid"][arm][stage][method].append(9.0)
    with pytest.raises(ValueError, match="grid"):
        validate_protocol(config)


def test_validate_rejects_stale_source_grid() -> None:
    config = copy.deepcopy(_config("tcga_ut"))
    config["sources"]["mitigation_grid"]["r100"]["stage2"]["posthoc_la"] = [0.5, 1.0]
    with pytest.raises(ValueError, match="source"):
        validate_protocol(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("stage1", "steps", 2400),
        ("stage1", "lr", 1e-3),
        ("stage1", "weight_decay", 0.0),
        ("stage1", "batch_size", 128),
        ("stage2", "steps", 100),
        ("cuda", "gamma", 0.5),
        ("cuda", "check_size", 5),
        ("cuda", "max_strength", 20),
    ],
)
def test_validate_rejects_changed_recipe(section: str, key: str, value: float) -> None:
    config = copy.deepcopy(_config("bracs"))
    mitigation = config["mitigation"]
    target = mitigation["cuda"] if section == "cuda" else mitigation["train"][section]
    target[key] = value
    with pytest.raises(ValueError, match="recipe"):
        validate_protocol(config)


def test_validate_rejects_changed_source_path() -> None:
    config = copy.deepcopy(_config("bracs"))
    config["damage_config"] = config["damage_config"].replace("26_damage_bracs", "25_damage_tcga_ut")
    with pytest.raises(ValueError, match="source"):
        validate_protocol(config)


@pytest.mark.parametrize("dataset", DATASETS)
def test_paths_resolve_to_existing_configs(dataset: str) -> None:
    config = _config(dataset)
    for key in ("damage_config", "mitigation_config", "frozen_cuda_config"):
        path = Path(config[key])
        assert path.is_absolute() and path.is_file()
    damage = "26_damage_bracs" if dataset == "bracs" else "25_damage_tcga_ut"
    lora = "41_mitigation_bracs" if dataset == "bracs" else "43_mitigation_tcga_ut"
    assert damage_dir(config, 2, "r100", 7).parts[-6:] == (dataset, "patch", "split=2", "results", "r100", "draw_7")
    assert damage in damage_dir(config, 0, "r1", 0).parts
    own = {"lr": lr_paths(config), "sgd": sgd_paths(config)}
    for kind, paths in own.items():
        assert sorted(paths) == [0, 1, 2]
        assert paths[1]["results"].parts[-4:] == (dataset, kind, "split=1", "results")
        assert "49_intended_mitigation" in paths[1]["results"].parts
    assert lora in lora_dirs(config)[0]["results"].parts
    assert "strong_mitigations_v2" in cuda_r100_dirs(config)[0]["results"].parts
