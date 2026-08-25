from __future__ import annotations

import platform
import sys
from typing import Any, cast
import hashlib

import numpy as np
import torch

from imbalance_benchmark.analysis.calibration import (
    apply_target_prior_correction,
    balanced_decision_logits,
    estimate_prior,
    softmax,
    temperature_scaled_payload,
)
from imbalance_benchmark.analysis.metrics import classification_payload
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    clustered_endpoints,
)
from imbalance_benchmark.common import REPO_ROOT, compute_sha256, write_run_record
from imbalance_benchmark.modeling.context import resolve_update_budget
from imbalance_benchmark.modeling.training import (
    class_priors,
    resolve_batch_size,
    run_evaluation,
)
from imbalance_benchmark.modeling.workflows.confirmation_provenance import (
    _condition_tiers,
    _load_record_freeze,
    _provenance_payload,
)
from imbalance_benchmark.modeling.workflows.run_context import (
    RunContext,
    RunExposure,
    cost_payload,
)


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


def _split_payload(
    res: dict[str, Any],
    class_names: list[str],
    method: str,
    tau: float,
    train_priors: torch.Tensor,
    target_priors: np.ndarray,
    identity: Any,
    tiers: dict[str, str] | None = None,
    is_mil: bool = False,
) -> dict[str, Any]:
    """Assemble one evaluated split's real classwise metrics and prediction arrays."""
    raw_l, priors = res["logits"], train_priors.detach().cpu().numpy()
    dec_l = balanced_decision_logits(raw_l, method, tau, priors)
    tar_l = apply_target_prior_correction(raw_l, method, tau, priors, target_priors)
    preds = softmax(dec_l).argmax(axis=1)
    probs_arr = softmax(tar_l)
    y, p_list, probs = res["targets"].tolist(), preds.tolist(), probs_arr.tolist()
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
        raw_probabilities=res["probs"].tolist(),
        clustered_endpoints=clustered_endpoints(
            res["targets"], preds, probs_arr, identity, is_mil=is_mil
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
        name: run_evaluation(model, loader, run.device, run.is_mil, run.n_classes)
        for name, loader in (("validation", run.val_loader), ("test", run.test_loader))
    }
    target_priors = estimate_prior(raw_results["validation"]["targets"], run.n_classes)
    freeze = _load_record_freeze(run)
    tiers = _condition_tiers(run, cond, freeze)
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
            run.is_mil,
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
            "feature_provenance": run.feature_provenance,
            "train_priors": class_priors_tensor.detach().cpu().tolist(),
            "target_priors": target_priors.tolist(),
            "cost": cost_payload(
                method,
                budget,
                elapsed,
                model,
                RunExposure(
                    len(ctx["train_dataset"]),
                    len(ctx.get("exposed_indices", set())),
                    int(ctx.get("processed_examples", 0)),
                    ctx.get("training_footprint_parameters"),
                    ctx.get("peak_memory_bytes"),
                    int(ctx.get("processed_instances", 0)) if run.is_mil else None,
                ),
            ),
            "method_diagnostics": ctx.get("method_diagnostics", {}),
            "environment": _environment_payload(),
            "provenance": _provenance_payload(run, cond, method, freeze),
            "splits": splits,
        },
    )


def _attach_temperature_scaled_test_outputs(splits: dict[str, dict[str, Any]]) -> None:
    """Persist post-selection temperature outputs used by calibration reporting."""
    validation, test = splits["validation"], splits["test"]
    test.update(
        temperature_scaled_payload(
            np.asarray(validation["logits"]),
            np.asarray(validation["labels"]),
            np.asarray(test["logits"]),
            np.asarray(test["labels"]),
        )
    )
