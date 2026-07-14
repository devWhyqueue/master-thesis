from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, cast
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data

from imbalance_benchmark.analysis.calibration import (
    apply_target_prior_correction,
    balanced_decision_logits,
    estimate_prior,
    softmax,
)
from imbalance_benchmark.analysis.metrics import assign_tiers, classification_payload
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    clustered_endpoints,
)
from imbalance_benchmark.common import write_run_record
from imbalance_benchmark.modeling.context import Regime, param_counts, updates_for
from imbalance_benchmark.modeling.training import (
    class_priors,
    resolve_batch_size,
    run_evaluation,
    update_budget,
)
from imbalance_benchmark.manifest.seeds import derive_seed


@dataclass
class RunContext(Regime):
    """Shared per-condition confirmation inputs: locked val/test loaders, paths, and seeds."""

    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    paths: dict[str, Path]
    seeds: list[int]
    class_names: list[str]
    assignment: str


def _cost_payload(
    method: str,
    budget: int,
    batch_size: int,
    elapsed: float,
    model: torch.nn.Module,
    n_unique_examples: int,
    parameter: int | float | None,
    n_unique_exposed: int,
) -> dict[str, Any]:
    """Assemble one confirmation run's update/example/timing/parameter cost record."""
    updates = updates_for(method, budget)
    mult = 2 if method == "mde" else int(parameter or 0) + 2 if method == "oko" else 1
    epu = batch_size * mult
    processed = updates * epu
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return {
        "updates": updates,
        "processed_examples": processed,
        "wall_clock_seconds": elapsed,
        "accelerator_hours": elapsed / 3600 if torch.cuda.is_available() else 0.0,
        "peak_accelerator_memory_bytes": peak,
        "examples_per_update": epu,
        "unique_training_examples": n_unique_examples,
        "unique_examples_exposed": n_unique_exposed,
        "effective_passes_through_unique_examples": processed
        / max(n_unique_examples, 1),
        **param_counts(model),
    }


def _checkpoint_hash(state: dict[str, Any]) -> str:
    """SHA-256 of the selected checkpoint's parameter tensors, for provenance."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(state[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _test_prediction_hash(splits: dict[str, Any]) -> str:
    """SHA-256 of the locked test predictions/probabilities."""
    test = splits.get("test", {})
    h = hashlib.sha256()
    for f in ("labels", "preds", "probabilities"):
        h.update(
            np.ascontiguousarray(
                np.asarray(test.get(f, []), dtype=np.float64)
            ).tobytes()
        )
    return h.hexdigest()


def _environment_payload() -> dict[str, Any]:
    """Capture the executable environment required to reproduce a confirmation run."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _condition_tiers(run: RunContext, cond: str) -> dict[str, str] | None:
    """Head/body/tail tiers for one condition from the frozen allocation and assignment."""
    fp = run.paths["data"] / "manifest_freeze.json"
    if not fp.exists():
        return None
    fr = json.loads(fp.read_text())
    alloc = fr.get("conditions", {}).get(cond, {}).get("allocated_counts") or fr.get(
        "assignment_conditions", {}
    ).get(run.assignment, {}).get(cond, {}).get("allocated_counts", {})
    if not alloc or not run.class_names:
        return None
    return assign_tiers(
        run.class_names,
        alloc,
        fr.get("tail_assignments", {}).get(run.assignment, run.class_names),
    )


def _split_payload(
    res: dict[str, Any],
    class_names: list[str],
    method: str,
    tau: float,
    train_priors: torch.Tensor,
    target_priors: np.ndarray,
    identity: Any,
    bootstrap_seed: int,
    tiers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble one evaluated split's real classwise metrics and prediction arrays."""
    raw_l, priors = res["logits"], train_priors.detach().cpu().numpy()
    dec_l = balanced_decision_logits(raw_l, method, tau, priors)
    tar_l = apply_target_prior_correction(raw_l, method, tau, priors, target_priors)
    preds = softmax(dec_l).argmax(axis=1)
    y, p_list, probs = res["targets"].tolist(), preds.tolist(), softmax(tar_l).tolist()
    payload = classification_payload(
        y,
        p_list,
        probs,
        class_names,
        tiers=tiers,
        ordinal=bool(class_names)
        and all(name.startswith("ISUP") for name in class_names),
    )
    payload.update(
        labels=y,
        preds=p_list,
        probabilities=probs,
        logits=tar_l.tolist(),
        raw_logits=raw_l.tolist(),
        raw_probabilities=res["probs"].tolist(),
        balanced_decision_logits=dec_l.tolist(),
        target_prior_logits=tar_l.tolist(),
        target_prior_probabilities=probs,
        clustered_endpoints=clustered_endpoints(
            res["targets"], preds, probs, identity, bootstrap_seed
        ),
    )
    return payload


def _run_and_record(
    cond: str,
    method: str,
    seed_idx: int,
    ctx: dict[str, Any],
    state: dict[str, Any],
    run: RunContext,
    elapsed: float,
    logit_adj_tau: float = 1.0,
    class_priors_tensor: torch.Tensor | None = None,
) -> None:
    """Evaluate a locked confirmation checkpoint on validation and test, and write its run record."""
    model = ctx["model"]
    model.load_state_dict({k: v.to(run.device) for k, v in state.items()})
    if class_priors_tensor is None:
        class_priors_tensor = class_priors(
            ctx["train_labels"], run.n_classes, run.device
        )
    raw_results = {
        n: run_evaluation(model, l, run.device, run.is_mil, run.n_classes)
        for n, l in (("validation", run.val_loader), ("test", run.test_loader))
    }
    target_priors = estimate_prior(raw_results["validation"]["targets"], run.n_classes)
    tiers = _condition_tiers(run, cond)
    splits = {}
    for name, result in raw_results.items():
        loader = run.val_loader if name == "validation" else run.test_loader
        splits[name] = _split_payload(
            result,
            run.class_names,
            method,
            logit_adj_tau,
            class_priors_tensor,
            target_priors,
            cast(Any, loader.dataset).df,
            derive_seed(ctx["seed"], "resampling"),
            tiers,
        )
    b_size = resolve_batch_size(run.config, run.is_mil)
    budget = update_budget(len(ctx["train_dataset"]), b_size)
    write_run_record(
        run.paths["results"]
        / f"assignment={run.assignment}"
        / cond
        / method
        / f"seed={seed_idx}",
        {
            "benchmark": "wsi" if run.is_mil else "patch",
            "condition": cond,
            "assignment": run.assignment,
            "method": method,
            "seed": ctx["seed"],
            "class_names": run.class_names,
            "tuning_params": ctx["param_config"],
            "selected_checkpoint_sha256": _checkpoint_hash(state),
            "test_prediction_sha256": _test_prediction_hash(splits),
            "train_priors": class_priors_tensor.detach().cpu().tolist(),
            "target_priors": target_priors.tolist(),
            "cost": _cost_payload(
                method,
                budget,
                b_size,
                elapsed,
                model,
                len(ctx["train_dataset"]),
                ctx["param_config"].get("parameter"),
                len(ctx.get("exposed_indices", set())),
            ),
            "method_diagnostics": ctx.get("method_diagnostics", {}),
            "environment": _environment_payload(),
            "splits": splits,
        },
    )
