"""Constants for the mitigation-methods experiment (exp-40).

Six mitigations plus the CE reference, defined in ``report/40_methods.tex``, all
trained on a Virchow2 encoder adapted only through LoRA (attention ``qkv``/``proj``
projections, every block) with a linear or cosine head. ``ce``, ``bs``, ``mixup``,
``la``, and ``gcl`` train stage one (encoder LoRA + head); ``posthoc_la``, ``crt``,
``disalign``, and ``gcl2`` are stage-two, cached-embedding methods that reuse a
stage-one run's exported features instead of a fresh forward pass.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from imbalance_benchmark.datasets.feature_provenance import FEATURE_DIM

__all__ = [
    "FEATURE_DIM",
    "STAGE1_METHODS",
    "STAGE2_METHODS",
    "STAGE2_SOURCE",
    "PARAMLESS_METHODS",
    "LORA_R",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_TARGET_REGEX",
    "lora_constants",
    "gcl_constants",
    "method_label",
    "TrainHParams",
    "train_hparams",
]

STAGE1_METHODS: tuple[str, ...] = ("ce", "bs", "mixup", "la", "gcl")
STAGE2_METHODS: tuple[str, ...] = ("posthoc_la", "crt", "disalign", "gcl2")
# Which stage-one run's exported embeddings a stage-two method is fit from.
STAGE2_SOURCE: dict[str, str] = {
    "posthoc_la": "ce",
    "crt": "ce",
    "disalign": "ce",
    "gcl2": "gcl",
}
# Methods whose strength has no free control parameter (Section "Methods").
PARAMLESS_METHODS: frozenset[str] = frozenset({"ce", "crt", "gcl2"})

# LoRA defaults (rank, scaling alpha, dropout); attention qkv/proj across every
# block, never patch_embed.proj (also named "proj", excluded by the regex's
# ``blocks.<i>.attn.`` prefix).
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_REGEX = r".*blocks\.\d+\.attn\.(qkv|proj)$"


def lora_constants(config: dict[str, Any]) -> tuple[int, int, float, str]:
    """Return this config's LoRA rank, alpha, dropout, and target-module regex."""
    lora = config.get("mitigation", {}).get("lora", {})
    return (
        int(lora.get("r", LORA_R)),
        int(lora.get("alpha", LORA_ALPHA)),
        float(lora.get("dropout", LORA_DROPOUT)),
        str(lora.get("target_regex", LORA_TARGET_REGEX)),
    )


def gcl_constants(config: dict[str, Any]) -> tuple[float, float, float]:
    """Return GCL's cosine scale ``s`` and effective-number range ``(a, b)``.

    Fixed before training per the report's GCL section; unlike ``sigma`` (the
    tuned control), these have no code default and must be set explicitly.
    """
    gcl = config.get("mitigation", {}).get("gcl")
    if not gcl:
        raise ValueError(
            "config must set mitigation.gcl: {s, a, b} before GCL training"
        )
    return float(gcl["s"]), float(gcl["a"]), float(gcl["b"])


def method_label(method: str, param: float | None) -> str:
    """Arm-directory label for one (method, param) pair, e.g. ``bs0.5``, ``ce``."""
    if method in PARAMLESS_METHODS or param is None:
        return method
    return f"{method}{param:g}"


class TrainHParams(NamedTuple):
    """Optimizer/schedule hyperparameters for one stage's fixed-step training loop."""

    batch_size: int
    steps: int
    lr: float
    weight_decay: float


def train_hparams(config: dict[str, Any], stage: str) -> TrainHParams:
    """Resolve one stage's training hyperparameters, config override else defaults."""
    defaults = {"batch_size": 256, "steps": 2000, "lr": 1e-4, "weight_decay": 1e-4}
    stage_config = config.get("mitigation", {}).get("train", {}).get(stage, {})
    merged = {**defaults, **stage_config}
    return TrainHParams(
        int(merged["batch_size"]),
        int(merged["steps"]),
        float(merged["lr"]),
        float(merged["weight_decay"]),
    )
