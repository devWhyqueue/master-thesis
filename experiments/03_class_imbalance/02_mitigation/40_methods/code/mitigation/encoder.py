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

from mitigation import lora_constants

__all__ = ["load_lora_encoder", "pool_tokens"]


def pool_tokens(output: torch.Tensor) -> torch.Tensor:
    """Concatenate the CLS token and mean patch token into a 2560-d feature, on-device."""
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


def _enable_training_perf(peft_model: PeftModel, device: torch.device) -> None:
    """Gradient checkpointing (measured necessary: 11.6GB card, batch 32, OOM without
    it) and TF32 matmul; both are training-only, harmless under ``inference_mode``.
    """
    peft_model.set_grad_checkpointing(True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True


def load_lora_encoder(
    config: dict[str, Any],
    device: torch.device,
    model_name: str = VIRCHOW2_MODEL,
    revision: str = VIRCHOW2_REVISION,
    weights_sha256: str = VIRCHOW2_WEIGHTS_SHA256,
) -> tuple[PeftModel, dict[str, Any], int]:
    """Load Virchow2, freeze it, and wrap its attention projections with LoRA.

    Returns the trainable ``PeftModel``, its resolved ``timm`` data config
    (input size, mean, std) for building the eval-matching decode transform,
    and the pooled feature dimension (``pool_tokens``: CLS + mean patch, twice
    the encoder's embedding width) the head must be sized to.
    """
    base_model, _ = load_feature_model(model_name, device, revision, weights_sha256)
    data_config: dict[str, Any] = resolve_data_config(
        base_model.pretrained_cfg, model=base_model
    )
    pooled_dim = 2 * cast(int, base_model.embed_dim)
    base_model.train()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    r, alpha, dropout, target_regex = lora_constants(config)
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_regex
    )
    peft_model = get_peft_model(base_model, lora_config)
    _enable_training_perf(peft_model, device)
    return peft_model, data_config, pooled_dim
