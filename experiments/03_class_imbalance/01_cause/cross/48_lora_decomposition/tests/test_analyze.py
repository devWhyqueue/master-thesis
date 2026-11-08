"""Contrast algebra, honest share intervals, uncertainty diagnostics, and completeness checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imbalance_benchmark.analysis.inference.context import BootstrapContext

from lora_decomposition import analyze
from lora_decomposition.analyze import contrasts, share_summary, summarize, summarize_dataset

_FIT_SPLIT = np.repeat(np.arange(3), 10)


def _per_fit(rng: np.random.Generator, offsets: dict[str, float], n_rep: int = 200) -> dict[str, np.ndarray]:
    base = 70 + rng.normal(0, 1, size=(30, n_rep + 1))
    return {arm: base - offset + rng.normal(0, 0.1, size=base.shape) for arm, offset in offsets.items()}


def _draw_weights(rng: np.random.Generator, n_rep: int = 200) -> np.ndarray:
    w = np.ones((30, n_rep + 1))
    for s in range(3):
        w[_FIT_SPLIT == s, 1:] = rng.multinomial(10, np.full(10, 0.1), size=n_rep).T
    return w


def test_contrast_algebra_and_untruncated_share() -> None:
    ba = {k: np.array([v]) for k, v in {"B": 70.0, "F": 68.0, "C": 69.0, "R": 66.0}.items()}
    out = contrasts(ba)
    expected = {"D": 4, "D_F": 2, "D_C": 1, "I": 1, "G": 3, "h": 0.75}
    for key, value in expected.items():
        np.testing.assert_allclose(out[key], [value])
    np.testing.assert_allclose(out["G"], out["D_F"] + out["I"])
    above_one = contrasts({k: np.array([v]) for k, v in {"B": 70.0, "F": 66.0, "C": 71.0, "R": 66.0}.items()})
    np.testing.assert_allclose(above_one["h"], [1.25])


def test_summarize_keeps_observed_out_of_quantiles() -> None:
    result = summarize(np.array([999.0, 1, 2, 3, 4]))
    assert result["estimate"] == 999.0
    assert result["lower"] == pytest.approx(np.quantile([1, 2, 3, 4], 0.025))
    assert result["upper"] == pytest.approx(np.quantile([1, 2, 3, 4], 0.975))


def test_share_status_for_unstable_denominators() -> None:
    stable = share_summary(np.array([4.0, 3.0, 5.0, 4.5]), np.array([3.0, 2.0, 4.0, 3.0]))
    assert stable["usable_for_hypothesis"] is True and stable["interval"]["estimate"] == 0.75

    crossing_d = np.array([2.0, -1.0, 3.0, 0.5])
    crossing = share_summary(crossing_d, np.array([1.0, 1.0, 1.0, 1.0]))
    assert crossing["usable_for_hypothesis"] is False
    assert crossing["nonpositive_denominators"] == 1 and crossing["zero_denominators"] == 0
    ratios = np.array([1.0, 1.0, 1.0, 1.0]) / crossing_d
    assert crossing["interval"]["lower"] == pytest.approx(np.quantile(ratios[1:], 0.025))

    small = share_summary(np.array([0.5, 0.4, 0.6]), np.array([0.2, 0.2, 0.2]))
    assert small["usable_for_hypothesis"] is False and "1 pp" in small["reason"]

    at_zero = share_summary(np.array([2.0, 0.0, 3.0]), np.array([1.0, 1.0, 1.0]))
    assert at_zero["interval"] is None and at_zero["zero_denominators"] == 1
    json.dumps(at_zero, allow_nan=False)


def test_identical_arms_give_zero_damage_and_serializable_status() -> None:
    rng = np.random.default_rng(0)
    per_fit = _per_fit(rng, {"B": 0.0, "F": 1.0, "C": 0.5})
    per_fit["R"] = per_fit["B"].copy()
    out = summarize_dataset(per_fit, _draw_weights(rng), _FIT_SPLIT)
    assert out["contrasts"]["D"] == {"estimate": 0.0, "lower": 0.0, "upper": 0.0}
    assert out["share"]["usable_for_hypothesis"] is False and out["share"]["interval"] is None
    json.dumps(out, allow_nan=False)


def test_diagnostics_use_equal_draws_or_observed_patients() -> None:
    rng = np.random.default_rng(1)
    per_fit = _per_fit(rng, {"B": 0.0, "F": 2.0, "C": 1.0, "R": 4.0}, n_rep=50)
    w = _draw_weights(rng, n_rep=50)
    arms = analyze.arm_distributions(per_fit, w)
    np.testing.assert_allclose(arms["patient_only"]["B"], per_fit["B"].mean(axis=0))
    observed = per_fit["B"][:, 0]
    np.testing.assert_allclose(arms["draw_only"]["B"], (w * observed[:, None]).sum(0) / w.sum(0))
    np.testing.assert_allclose(arms["full"]["B"], (w * per_fit["B"]).sum(0) / w.sum(0))
    out = summarize_dataset(per_fit, w, _FIT_SPLIT)
    assert set(out["per_split"]) == {"0", "1", "2"}
    assert out["contrasts"]["D"]["estimate"] == pytest.approx(arms["full"]["B"][0] - arms["full"]["R"][0])


def test_missing_arm_or_incomplete_block_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(2)
    per_fit = _per_fit(rng, {"B": 0.0, "F": 1.0, "C": 0.5})
    with pytest.raises(ValueError, match="arms"):
        summarize_dataset(per_fit, _draw_weights(rng), _FIT_SPLIT)
    monkeypatch.setattr(analyze, "validate_run", lambda c, s, d, arm: not (s == 1 and d == 3 and arm == "C"))
    with pytest.raises(ValueError, match=r"\(1, 3, 'C'\)"):
        analyze.require_complete({"protocol": {}})


def _manifest(root: Path, split: int, cases: list[str]) -> None:
    data = root / f"split={split}" / "data"
    data.mkdir(parents=True)
    pd.DataFrame(
        {"case_id": cases, "slide_id": [f"{c}_s" for c in cases], "cancer_type": ["a", "b"] * (len(cases) // 2), "split": "test"}
    ).to_csv(data / "manifest.csv", index=False)


def test_patient_repeated_across_splits_shares_its_bootstrap_weight(tmp_path: Path) -> None:
    """Audit of the existing crossed bootstrap: one weight per patient across overlapping splits."""
    _manifest(tmp_path, 0, ["P1", "P2", "P3", "P4"])
    _manifest(tmp_path, 1, ["P3", "P5", "P6", "P7"])
    ctxs = [
        BootstrapContext({"root": tmp_path / f"split={s}", "data": tmp_path / f"split={s}" / "data"}, False, 50, 9)
        for s in (0, 1)
    ]
    row0 = int(np.flatnonzero(ctxs[0].case_ids == "P3")[0])
    row1 = int(np.flatnonzero(ctxs[1].case_ids == "P3")[0])
    weight0 = ctxs[0].weights.patient[ctxs[0].weights.row_patient[row0]]
    weight1 = ctxs[1].weights.patient[ctxs[1].weights.row_patient[row1]]
    np.testing.assert_array_equal(weight0, weight1)


def _synthetic(values: dict[str, float]) -> dict:
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.2, size=(30, 101))
    noise[:, 0] = 0.0
    per_fit = {arm: value + noise for arm, value in values.items()}
    return summarize_dataset(per_fit, _draw_weights(rng, n_rep=100), _FIT_SPLIT)


def test_report_renders_table_and_hides_unusable_benchmark(tmp_path: Path) -> None:
    from lora_decomposition import report

    stable = _synthetic({"B": 70.0, "F": 68.0, "C": 69.0, "R": 66.0})
    unstable = _synthetic({"B": 70.0, "F": 69.8, "C": 69.9, "R": 69.5})
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"datasets": {"bracs": stable, "tcga_ut": unstable}}), encoding="utf-8")
    report.render_report(analysis, tmp_path)

    table = (tmp_path / "arm_table.tex").read_text(encoding="utf-8")
    d_row = next(line for line in table.splitlines() if "$D$" in line)
    h_row = next(line for line in table.splitlines() if "$h" in line)
    assert d_row.split("&")[1].strip().startswith("4.00 [")
    assert h_row.split("&")[1].strip().startswith("0.75 [")
    assert "unavailable" in h_row.split("&")[2]
    assert report.benchmark_band(stable) is not None and report.benchmark_band(unstable) is None
    assert (tmp_path / "component_figure.pdf").is_file() and (tmp_path / "recovery_figure.pdf").is_file()


def test_analysis_refuses_blocks_that_break_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    from lora_decomposition import check

    monkeypatch.setattr(analyze, "validate_run", lambda *a: True)
    monkeypatch.setattr(analyze, "load_block", lambda config, s, d: {"block": (s, d)})
    monkeypatch.setattr(analyze, "block_problems", lambda prov: ["seed differs"] if prov["block"] == (2, 7) else [])
    with pytest.raises(ValueError, match=r"\(2, 7\).*seed differs"):
        analyze.require_complete({"protocol": {}})
    assert len(check.phase_blocks("main")) == 30 and len(check.phase_blocks("pilot")) == 3
