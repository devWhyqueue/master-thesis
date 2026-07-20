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
    temperature_scaled_payload,
)
from imbalance_benchmark.analysis.metrics import (
    assign_tiers,
    classification_payload,
)
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    clustered_endpoints,
)
from imbalance_benchmark.common import REPO_ROOT, compute_sha256, write_run_record
from imbalance_benchmark.modeling.context import (
    Regime,
    cost_payload,
    resolve_update_budget,
)
from imbalance_benchmark.modeling.training import (
    class_priors,
    resolve_batch_size,
    run_evaluation,
)


@dataclass
class RunContext(Regime):
    """Shared per-condition confirmation inputs: locked val/test loaders, paths, and seeds."""

    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    paths: dict[str, Path]
    seeds: list[int]
    class_names: list[str]
    assignment: str


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
    lock_path = REPO_ROOT / "uv.lock"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dependency_lock": {
            "path": "uv.lock",
            "sha256": compute_sha256(lock_path) if lock_path.is_file() else None,
        },
    }


def _condition_tiers(run: RunContext, cond: str) -> dict[str, str] | None:
    """Head/body/tail tiers for one condition from the frozen allocation and assignment."""
    if cond == "balanced":
        return None
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
        clustered_endpoints=clustered_endpoints(res["targets"], preds, probs, identity),
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
        name: run_evaluation(model, loader, run.device, run.is_mil, run.n_classes)
        for name, loader in (("validation", run.val_loader), ("test", run.test_loader))
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
            tiers,
        )
    _attach_temperature_scaled_test_outputs(splits)
    b_size = resolve_batch_size(run.config, run.is_mil)
    budget = resolve_update_budget(ctx, b_size)
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
            "selected_checkpoint_step": ctx.get("selected_checkpoint_step", 0),
            "test_prediction_sha256": _test_prediction_hash(splits),
            "train_priors": class_priors_tensor.detach().cpu().tolist(),
            "target_priors": target_priors.tolist(),
            "cost": cost_payload(
                method,
                budget,
                elapsed,
                model,
                len(ctx["train_dataset"]),
                len(ctx.get("exposed_indices", set())),
                int(ctx.get("processed_examples", 0)),
                ctx.get("training_footprint_parameters"),
                ctx.get("peak_memory_bytes"),
                processed_instances=(
                    int(ctx.get("processed_instances", 0)) if run.is_mil else None
                ),
            ),
            "method_diagnostics": ctx.get("method_diagnostics", {}),
            "environment": _environment_payload(),
            "splits": splits,
        },
    )


def _attach_temperature_scaled_test_outputs(splits: dict[str, dict[str, Any]]) -> None:
    """Persist post-selection temperature outputs used by calibration reporting."""
    validation, test = splits["validation"], splits["test"]
    test.update(
        temperature_scaled_payload(
            np.asarray(validation["target_prior_logits"]),
            np.asarray(validation["labels"]),
            np.asarray(test["target_prior_logits"]),
            np.asarray(test["labels"]),
        )
    )
