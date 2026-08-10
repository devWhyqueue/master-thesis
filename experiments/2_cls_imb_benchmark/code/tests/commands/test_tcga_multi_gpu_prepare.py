from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest
import yaml

from imbalance_benchmark.commands import prepare
from imbalance_benchmark.datasets.panda_materialize import select_physical_shard
from imbalance_benchmark.common import load_config
from imbalance_benchmark.hydra.workflow import build_workflow, render_sbatch


def test_tcga_patch_prepare_is_one_staged_finalize_job() -> None:
    """The heavy extraction runs as a separate `prepare-extract-shard` array
    (see test_prepare_extract_shard.py); this `prepare` stage just finalizes
    splits/provenance against the already-populated feature cache."""
    config = load_config(
        Path(__file__).resolve().parents[3] / "configs" / "tcga_ut_patch.yaml"
    )
    prepare = next(job for job in build_workflow(config) if job.name == "prepare")
    script = render_sbatch(prepare, config)

    assert (prepare.partition, prepare.gpus, prepare.cpus, prepare.memory) == (
        "gpu-2h",
        1,
        8,
        "16G",
    )
    assert "#SBATCH --array=" not in script
    assert script.count("cp /home/space/datasets-sqfs/tcga-ut/patch.sqfs") == 1


def test_slide_shard_splits_round_robin_by_slide() -> None:
    frame = pd.DataFrame(
        {
            "slide_id": ["s3", "s3", "s1", "s2", "s2"],
            "patch_id": ["a", "b", "c", "d", "e"],
        }
    )

    # sorted slide ids: s1, s2, s3 -> shard 0 takes s1, s3; shard 1 takes s2
    shard0 = select_physical_shard(frame, {"name": "tcga_ut"}, 0, 2)
    shard1 = select_physical_shard(frame, {"name": "tcga_ut"}, 1, 2)

    assert sorted(shard0["slide_id"].unique()) == ["s1", "s3"]
    assert sorted(shard1["slide_id"].unique()) == ["s2"]
    assert len(shard0) + len(shard1) == len(frame)


def _shard_config_path(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"outputs": str(tmp_path / "outputs")},
                "dataset": {"name": "tcga_ut", "regime": "patch"},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_cmd_prepare_extract_shard_extracts_only_its_slides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _shard_config_path(tmp_path)
    frame = pd.DataFrame(
        {
            "slide_id": ["s1", "s2"],
            "case_id": ["c1", "c2"],
            "cancer_type": ["LUAD", "LUSC"],
            "patch_id": ["s1/0", "s2/0"],
            "image_path": ["s1/0.jpg", "s2/0.jpg"],
        }
    )
    monkeypatch.setattr(prepare, "build_manifest", lambda _config: frame)
    calls: list[pd.DataFrame] = []
    monkeypatch.setattr(
        prepare,
        "attach_extracted_features",
        lambda shard_df, *_args, **_kwargs: calls.append(shard_df) or shard_df,
    )

    prepare.cmd_prepare_extract_shard(
        argparse.Namespace(config=str(config_path), shard_index=0, shard_count=2)
    )

    assert len(calls) == 1
    assert list(calls[0]["slide_id"]) == ["s1"]


def test_cmd_prepare_extract_shard_skips_feature_chunk_manifests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _shard_config_path(tmp_path)
    frame = pd.DataFrame({"slide_id": ["s1"], "feature_path": ["s1_0.pt"]})
    monkeypatch.setattr(prepare, "build_manifest", lambda _config: frame)
    monkeypatch.setattr(
        prepare,
        "attach_extracted_features",
        lambda *_a, **_k: pytest.fail("should not extract a feature-chunk manifest"),
    )

    prepare.cmd_prepare_extract_shard(
        argparse.Namespace(config=str(config_path), shard_index=0, shard_count=1)
    )


def test_cmd_prepare_extract_shard_noops_on_empty_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _shard_config_path(tmp_path)
    frame = pd.DataFrame(
        {
            "slide_id": ["s1"],
            "case_id": ["c1"],
            "cancer_type": ["LUAD"],
            "patch_id": ["s1/0"],
            "image_path": ["s1/0.jpg"],
        }
    )
    monkeypatch.setattr(prepare, "build_manifest", lambda _config: frame)
    monkeypatch.setattr(
        prepare,
        "attach_extracted_features",
        lambda *_a, **_k: pytest.fail("should not extract an empty shard"),
    )

    prepare.cmd_prepare_extract_shard(
        argparse.Namespace(config=str(config_path), shard_index=3, shard_count=4)
    )
