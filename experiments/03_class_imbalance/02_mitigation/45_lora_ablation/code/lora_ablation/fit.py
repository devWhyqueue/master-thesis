"""Fit the cached-feature candidates (CE, train-time LA, post-hoc LA) without any encoder work.

One CE head per draw is stored and scored for every post-hoc strength; every output is bound to
the audited inventory through its sidecar, and a post-hoc record is reusable only while its CE
head weights are unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from breadth.sampling import load_features_for_df

from centre import N_DRAWS

from mitigation.data import ImageShard, load_shard, run_dir
from mitigation.grid import stage1_jobs, stage2_jobs
from mitigation.train import (
    EvalSplit,
    RunMeta,
    Stage1Artifacts,
    evaluate_and_record,
    run_posthoc_la,
)

from prevalence import patients_per_class

from lora_ablation import ARM
from lora_ablation.heads import fit_candidate, frozen_artifacts, load_head
from lora_ablation.resume import (
    AUDIT_SIDECAR,
    expected_sidecar,
    is_complete,
    measure,
    require_inventory,
    write_sidecar,
)

__all__ = [
    "AUDIT_SIDECAR",
    "HEAD_NAME",
    "fit_candidate",
    "fit_cpu_split",
    "frozen_artifacts",
]

HEAD_NAME = "head.pt"
_CPU = torch.device("cpu")
_CPU_SETTINGS = {"device": "cpu", "precision": "fp16 cached features, fp32 head"}


def _record(
    shard: ImageShard,
    out_dir: Path,
    key: tuple[str, float | None],
    run: tuple[Stage1Artifacts, tuple[np.ndarray, np.ndarray], dict[str, Any]],
) -> None:
    method, param = key
    artifacts, logits, config = run
    counts = dict(zip(shard.classes, (int(c) for c in artifacts.counts)))
    evaluate_and_record(
        out_dir,
        RunMeta(config, method, param, ARM, counts),
        EvalSplit(logits[0], artifacts.val_y, shard.val_identity),
        EvalSplit(logits[1], artifacts.test_y, shard.test_identity),
    )


class _Draw:
    """One (split, draw)'s fitting context: shard, features, fingerprints, and lazily built artifacts."""

    def __init__(
        self,
        config: dict[str, Any],
        shard: ImageShard,
        where: tuple[int, int],
        eval_features: tuple[torch.Tensor, torch.Tensor],
        fingerprint: str,
    ) -> None:
        self.config, self.shard, self.where = config, shard, where
        self.eval_features = eval_features
        self.expected = expected_sidecar(config, fingerprint, *where)
        self._artifacts: Stage1Artifacts | None = None

    def out_dir(self, method: str, param: float | None) -> Path:
        """Result directory of one candidate of this draw."""
        return run_dir(self.shard, ARM, method, param, self.where[1])

    def complete(
        self, method: str, param: float | None, extra: dict | None = None
    ) -> bool:
        """Whether the candidate's record and sidecar already match the audit (and ``extra`` fields)."""
        expected = {**self.expected, **(extra or {})}
        return is_complete(self.out_dir(method, param), expected)

    @property
    def artifacts(self) -> Stage1Artifacts:
        """Training features load once per draw, and only when something is pending."""
        if self._artifacts is None:
            self._artifacts = frozen_artifacts(
                self.shard,
                *self.where,
                patients_per_class(self.config),
                self.eval_features,
            )
        return self._artifacts

    def fit(self, method: str, param: float | None) -> Path:
        """Fit and record one CE/LA candidate; the sidecar goes last."""
        out_dir = self.out_dir(method, param)
        with measure() as resources:
            head, val, test = fit_candidate(
                self.config, self.artifacts, method, param, *self.where
            )
            run = (self.artifacts, (val, test), self.config)
            _record(self.shard, out_dir, (method, param), run)
            if method == "ce":
                torch.save(head.state_dict(), out_dir / HEAD_NAME)
        extra = {**_CPU_SETTINGS, "stage": "stage1", "resources": resources}
        write_sidecar(out_dir, self.expected, extra)
        return out_dir


def _ce_head(draw: _Draw) -> tuple[dict[str, torch.Tensor], str]:
    """The draw's CE head and its file hash, refitting when the record or weights are unusable."""
    out_dir = draw.out_dir("ce", None)
    dim, k = draw.eval_features[0].shape[1], len(draw.shard.classes)
    state = load_head(out_dir / HEAD_NAME, dim, k)
    if state is None or not draw.complete("ce", None):
        out_dir = draw.fit("ce", None)
        state = load_head(out_dir / HEAD_NAME, dim, k)
    if state is None:
        raise ValueError(f"CE head missing after fit: {out_dir}")
    return state, hashlib.sha256((out_dir / HEAD_NAME).read_bytes()).hexdigest()


def _score_posthoc(draw: _Draw, state: dict[str, torch.Tensor], head_sha: str) -> None:
    """Score every post-hoc strength from the one CE head; no extra CE fit per strength."""
    extra = {"ce_head_sha256": head_sha}
    for job in stage2_jobs(draw.config):
        if draw.complete("posthoc_la", job.param, extra):
            continue
        out_dir = draw.out_dir("posthoc_la", job.param)
        artifacts = dataclasses.replace(draw.artifacts, head_state=state)
        tau = float(job.param or 0.0)
        with measure() as resources:
            logits = run_posthoc_la(artifacts, tau, len(artifacts.counts), _CPU)
            run = (artifacts, logits, draw.config)
            _record(draw.shard, out_dir, ("posthoc_la", job.param), run)
        settings = {**_CPU_SETTINGS, "stage": "stage2", "resources": resources}
        write_sidecar(out_dir, {**draw.expected, **extra}, settings)


def _fit_cpu_draw(draw: _Draw) -> None:
    state, head_sha = _ce_head(draw)
    for job in stage1_jobs(draw.config):
        if job.method == "la" and not draw.complete("la", job.param):
            draw.fit("la", job.param)
    _score_posthoc(draw, state, head_sha)


def fit_cpu_split(
    config: dict[str, Any], split_idx: int, draw_idx: int | None = None
) -> None:
    """Fit all missing CE/LA candidates and post-hoc scores of one split (ten draws, or one pilot draw)."""
    fingerprint = require_inventory(config)
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())
    torch.set_num_threads(int(cpus))
    shard = load_shard(config, split_idx)
    val_x, test_x = (
        torch.from_numpy(load_features_for_df(df)).to(torch.float16)
        for df in (shard.val_df, shard.test_df)
    )
    for draw in range(N_DRAWS) if draw_idx is None else [draw_idx]:
        context = (config, shard, (split_idx, draw), (val_x, test_x), fingerprint)
        _fit_cpu_draw(_Draw(*context))
