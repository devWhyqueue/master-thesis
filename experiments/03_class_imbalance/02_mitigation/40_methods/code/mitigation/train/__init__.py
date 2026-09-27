"""Stage-one LoRA/head training and stage-two cached-embedding methods."""

from __future__ import annotations

from mitigation.train.artifacts import (
    STAGE1_ARTIFACT_NAME,
    Stage1Artifacts,
    load_stage1,
    save_stage1,
)
from mitigation.train.eval import (
    EvalSplit,
    RunMeta,
    evaluate_and_record,
    forward_embeddings,
    head_logits,
    stream_embeddings,
)
from mitigation.train.stage1 import ArmBatch, Stage1Output, run_stage1
from mitigation.train.stage2 import run_crt, run_disalign, run_gcl2, run_posthoc_la

__all__ = [
    "STAGE1_ARTIFACT_NAME",
    "Stage1Artifacts",
    "load_stage1",
    "save_stage1",
    "EvalSplit",
    "RunMeta",
    "evaluate_and_record",
    "forward_embeddings",
    "head_logits",
    "stream_embeddings",
    "ArmBatch",
    "Stage1Output",
    "run_stage1",
    "run_crt",
    "run_disalign",
    "run_gcl2",
    "run_posthoc_la",
]
