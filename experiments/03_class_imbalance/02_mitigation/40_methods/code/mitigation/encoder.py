"""LoRA-adapted Virchow2 encoder and its CLS + mean-patch pooling.

Loads the same pinned Virchow2 snapshot as the frozen-feature benchmark
(``imbalance_benchmark.datasets.features.load_feature_model``), then wraps it
in a trainable ``peft`` LoRA adapter on every block's attention projections.
Pooling mirrors ``imbalance_benchmark.datasets.features._virchow2_pool`` but
keeps the result on-device, since that helper's ``.cpu()`` is unusable inside
a training graph.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from timm.data.config import resolve_data_config
from imbalance_benchmark.datasets.features import load_feature_model
from imbalance_benchmark.datasets.feature_provenance import (
    VIRCHOW2_MODEL,
    VIRCHOW2_REVISION,
    VIRCHOW2_WEIGHTS_SHA256,
)
from transfer.uni import load_uni_model

from mitigation import lora_constants

__all__ = ["load_lora_encoder", "pool_tokens"]


def pool_tokens(output: torch.Tensor) -> torch.Tensor:
    """Concatenate the CLS token and mean patch token into a 2560-d feature, on-device.

    A 2-D output is already pooled (UNI2-h's token pooling) and passes through.
    """
    if output.ndim == 2:
        return output
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


def _enable_training_perf(
    peft_model: PeftModel, device: torch.device, perf: dict[str, Any]
) -> None:
    """Gradient checkpointing, optional per-block ``torch.compile``, and TF32 matmul.

    ``mitigation.perf``: ``grad_checkpointing`` (default on; measured necessary on an
    11.6GB card at batch 32) can go off when micro-batching bounds activation memory;
    ``compile`` (default off) compiles each transformer block (needs a C compiler).
    """
    peft_model.set_grad_checkpointing(bool(perf.get("grad_checkpointing", True)))
    if perf.get("compile", False):
        for block in cast(Any, peft_model.base_model).model.blocks:
            block.compile()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True


def _load_base(
    uni2h: bool, device: torch.device, virchow2: tuple[str, str, str]
) -> Any:
    """The frozen-benchmark encoder: pinned UNI2-h, else the pinned Virchow2 snapshot."""
    if uni2h:
        return load_uni_model(device)[0]
    return load_feature_model(virchow2[0], device, virchow2[1], virchow2[2])[0]


def load_lora_encoder(
    config: dict[str, Any],
    device: torch.device,
    model_name: str = VIRCHOW2_MODEL,
    revision: str = VIRCHOW2_REVISION,
    weights_sha256: str = VIRCHOW2_WEIGHTS_SHA256,
) -> tuple[PeftModel, dict[str, Any], int]:
    """Load Virchow2 (or UNI2-h, ``mitigation.encoder: uni2h``), freeze it, and wrap its attention projections with LoRA.

    Returns the trainable ``PeftModel``, its resolved ``timm`` data config
    (input size, mean, std) for building the eval-matching decode transform,
    and the pooled feature dimension (``pool_tokens``: CLS + mean patch, twice
    the encoder's embedding width) the head must be sized to.
    """
    mitigation = config.get("mitigation", {})
    uni2h = mitigation.get("encoder", "virchow2") == "uni2h"
    base_model = _load_base(uni2h, device, (model_name, revision, weights_sha256))
    data_config: dict[str, Any] = resolve_data_config(
        base_model.pretrained_cfg, model=base_model
    )
    pooled_dim = cast(int, base_model.embed_dim) * (1 if uni2h else 2)
    base_model.train()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    r, alpha, dropout, target_regex = lora_constants(config)
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_regex
    )
    peft_model = get_peft_model(base_model, lora_config)
    _enable_training_perf(peft_model, device, mitigation.get("perf", {}))
    return peft_model, data_config, pooled_dim
