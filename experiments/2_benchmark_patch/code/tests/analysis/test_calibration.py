from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imbalance_benchmark.analysis.calibration import (
    apply_temperature,
    apply_target_prior_correction,
    balanced_decision_logits,
    fit_temperature,
    temperature_scaled_probabilities,
)
from imbalance_benchmark.analysis.calibration import (
    seed_averaged_reliability_curve,
    temperature_scaled_payload,
)
from imbalance_benchmark.analysis.inference.gates import (
    calibration_gate,
)
from imbalance_benchmark.analysis.metrics import (
    negative_log_likelihood,
)
from imbalance_benchmark.analysis.reporting.secondary_intervals.calibration_intervals import (
    _distribution_summary,
)
from imbalance_benchmark.analysis.reporting.ingestion import (
    _run_calibration,
    calibration_summary,
)
from imbalance_benchmark.common import read_run_record, write_run_record

def test_temperature_scaling_lowers_synthetic_overconfidence_nll():
    rng = np.random.default_rng(0)
    n, n_classes = 200, 3
    labels = rng.integers(0, n_classes, size=n)
    logits = np.zeros((n, n_classes))
    logits[np.arange(n), labels] = 8.0
    logits += rng.normal(scale=0.5, size=(n, n_classes))
    flip = rng.random(n) < 0.15
    logits[flip] = logits[flip][:, ::-1]
    fit = fit_temperature(logits, labels)
    from imbalance_benchmark.analysis.calibration import apply_temperature

    raw_nll = negative_log_likelihood(labels, apply_temperature(logits, 1.0))
    calibrated_nll = negative_log_likelihood(
        labels, apply_temperature(logits, fit.temperature)
    )
    assert fit.temperature > 1.0
    assert calibrated_nll < raw_nll

def test_target_prior_correction_identity_for_unscoped_methods():
    logits = np.array([[1.0, 2.0, 3.0]])
    pi_train = np.array([0.5, 0.3, 0.2])
    pi_target = np.array([0.2, 0.3, 0.5])
    out = apply_target_prior_correction(logits, "weighted_ce", 1.0, pi_train, pi_target)
    assert np.array_equal(out, logits)

def test_target_prior_correction_posthoc_formula():
    logits = np.array([[1.0, 2.0, 3.0]])
    pi_train = np.array([0.5, 0.3, 0.2])
    pi_target = np.array([0.2, 0.3, 0.5])
    out = apply_target_prior_correction(
        logits, "post_hoc_logit_adjustment", 0.5, pi_train, pi_target
    )
    expected = logits - 0.5 * np.log(pi_train) + np.log(pi_target)
    assert np.allclose(out, expected)
    balanced = balanced_decision_logits(
        logits, "post_hoc_logit_adjustment", 0.5, pi_train
    )
    assert np.allclose(balanced, logits - 0.5 * np.log(pi_train))


def test_target_prior_correction_posthoc_varies_with_tau():
    """Regression: the calibration branch used to discard tau, aliasing CE exactly."""
    logits = np.array([[1.0, 2.0, 3.0]])
    pi_train = np.array([0.5, 0.3, 0.2])
    pi_target = np.array([0.2, 0.3, 0.5])
    outputs = [
        apply_target_prior_correction(
            logits, "post_hoc_logit_adjustment", tau, pi_train, pi_target
        )
        for tau in (0.25, 0.5, 2.0)
    ]
    assert not np.allclose(outputs[0], outputs[1])
    assert not np.allclose(outputs[1], outputs[2])
    assert not np.allclose(outputs[0], outputs[2])
    # tau = 1.0 is CE's own probabilities; post-hoc LA must differ from it
    # whenever it selects a different tau, or the two methods are byte-identical.
    ce_at_one = apply_target_prior_correction(
        logits, "ce", 1.0, pi_train, pi_target
    )
    assert not np.allclose(outputs[1], ce_at_one)

def test_calibration_gate_thresholds():
    assert calibration_gate(0.19, (0.02, 0.30)) is True
    assert calibration_gate(0.17, (0.01, 0.28)) is False

def test_calibration_summary_separates_observed_estimate_from_bootstrap() -> None:
    assert _distribution_summary([0.9, 0.1, 0.2], "ECE") == {
        "ECE": 0.9,
        "ECE 95% CI": "[0.103, 0.198]",
    }

def test_temperature_scaled_ece_is_computed_without_a_per_run_bootstrap() -> None:
    payload = temperature_scaled_payload(
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        np.array([0, 1]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0, 1]),
    )

    assert "temperature_scaled_ece_ci" not in payload


def test_temperature_payload_omits_reconstructible_sample_arrays() -> None:
    payload = temperature_scaled_payload(
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        np.array([0, 1]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([0, 1]),
    )

    assert {"temperature_scaled_logits", "temperature_scaled_probabilities"}.isdisjoint(
        payload
    )


def test_temperature_scaled_probabilities_reconstruct_and_preserve_legacy_order() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    temperature = 1.7
    scaled = apply_temperature(logits, temperature)

    assert np.allclose(
        temperature_scaled_probabilities(
            {"temperature": temperature, "logits": logits.tolist()}
        ),
        scaled,
    )
    assert np.array_equal(
        temperature_scaled_probabilities(
            {
                "temperature": temperature,
                "logits": logits.tolist(),
                "temperature_scaled_probabilities": [[0.4, 0.6], [0.7, 0.3]],
            }
        ),
        [[0.4, 0.6], [0.7, 0.3]],
    )
    assert np.array_equal(
        temperature_scaled_probabilities({"probabilities": [[0.8, 0.2]]}),
        [[0.8, 0.2]],
    )

def test_reliability_bins_are_averaged_over_seeds_not_probabilities() -> None:
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.6, 0.4]],
        ]
    )

    _, confidence, accuracy = seed_averaged_reliability_curve(
        probabilities, np.array([0, 1])
    )

    assert confidence.tolist() == pytest.approx([0.6, 0.9])
    assert accuracy.tolist() == pytest.approx([0.5, 1.0])

def test_calibration_summary_reports_all_claimed_metrics() -> None:
    record = {
        "splits": {
            "validation": {"labels": [0, 1], "logits": [[4.0, 0.0], [0.0, 4.0]]},
            "test": {
                "labels": [0, 1],
                "logits": [[2.0, 0.0], [0.0, 2.0]],
                "probabilities": [[0.88, 0.12], [0.12, 0.88]],
            },
        }
    }

    summary = _run_calibration(record)

    assert summary is not None
    assert {"temperature_scaled_logits", "temperature_scaled_probabilities"} & set(
        summary
    ) == set()
    assert {
        "temperature_scaled_test_nll",
        "temperature_scaled_test_brier",
        "temperature_scaled_test_ece",
    } <= set(summary)
    assert "temperature_scaled_reliability" in summary


def test_selective_calibration_matches_full_record_output(tmp_path: Path) -> None:
    result_dir = tmp_path / "results" / "balanced" / "ce" / "seed=0"
    record = {
        "benchmark": "patch",
        "condition": "balanced",
        "method": "ce",
        "splits": {
            "validation": {
                "labels": [0, 1],
                "logits": [[4.0, 0.0], [0.0, 4.0]],
                "probabilities": [[0.98, 0.02], [0.02, 0.98]],
                "raw_probabilities": [[0.95, 0.05], [0.05, 0.95]],
            },
            "test": {
                "labels": [0, 1],
                "logits": [[2.0, 0.0], [0.0, 2.0]],
                "probabilities": [[0.88, 0.12], [0.12, 0.88]],
                "raw_probabilities": [[0.8, 0.2], [0.2, 0.8]],
            },
        },
    }
    write_run_record(result_dir, record)
    full_record = read_run_record(result_dir)
    assert full_record is not None
    expected = _run_calibration(full_record)

    summary = calibration_summary({"results": tmp_path / "results"})

    assert summary == {"patch:native:balanced:ce:seed=0": expected}
