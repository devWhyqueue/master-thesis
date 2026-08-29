# 02 — Analysis-layer corrections

**Depends on:** 01. **Gates:** 07. **Cluster cost:** none (analysis code only).

## Context

Six choices in the inference and signal-profile layers bias the results in identifiable directions.
None require retraining. Landing them before the re-run means `analyze` produces correct output the
first time rather than needing a second pass.

## 1. Shortage scores are z-scored across datasets, not within

[matching.py:136](../experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/predictors/signals/matching.py#L136)
pools all 18 units from every dataset root before `_standardize`. The "dominant shortage" is
therefore a rank within the benchmark population, not a property of the unit — TCGA-UT's nominal
shortage reads `-1.00` because it is smaller than CAMELYON16's, not because TCGA-UT has none.

Standardize within dataset instead.

## 2. `S_nom` is constant across tail assignments

In `signal_profiles.tex`, BRACS moderate reads `-0.86` for native, aligned and reversed alike; every
dataset behaves the same way. The allocated-count multiset is identical across assignments and only
permuted, so the nominal-shortage score cannot see the assignment at all. Neither can `rho`.

This matters because the assignment drives the largest damage differences in the whole benchmark —
on BRACS the reversed assignment damages roughly three times as much as the aligned one — while two
of the four axes are blind to it. Compute the shortage over deprived classes weighted by difficulty
so the label responds to the manipulation.

## 3. Support–difficulty alignment is degenerate on binary targets

`A` is a correlation over two points, hence exactly `+/-1` on CAMELYON16 and PANDA (confirmed in
`signal_profiles.tex`). Half the datasets contribute a saturated difficulty score and a
non-identifiable RQ3 predictor. The report notices the symptom (`beta_A = -0.0008`) but attributes it
to coarse encoding rather than to non-identifiability.

Either exclude binary datasets from the alignment axis, or replace the correlation with a statistic
that is not saturated at K = 2. State which in the protocol.

## 4. Holm family size scales with how many gates a dataset opens

`confirmatory_family` ([holm.py:51-67](../experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/inference/confirmatory/holm.py#L51-L67))
pools both gates, both severities and all assignments into one per-dataset family. Realized sizes:
BRACS 68 tests, CAMELYON16 12, PANDA 11, TCGA-UT 12. Two BRACS prevalence-weighted tests with
recovery above 0.92 fail adjustment purely because BRACS also opened the calibration gate.

Opening a second gate should not cost a dataset its confirmatory power. Make the family per
(dataset, gate). The two endpoints are co-primary and answer different questions, which is the
justification the protocol amendment needs.

## 5. One global gate threshold for four datasets

`_report_thresholds` (`experiments/2_benchmark_patch/code/derive_deficit_thresholds.py:163-186`)
takes `2 * max` seed-SD **across all datasets**, so `DISCRIMINATION_THRESHOLD = 0.01443` is
calibrated to the noisiest dataset (BRACS) and applied to the quietest (TCGA-UT, whose largest
deficit is 0.016). Stable datasets are systematically under-gated.

The script already collects per-dataset `DispersionRow`s — group instead of taking the global max.

**Flag the consequence:** this changes gate outcomes. TCGA-UT difficulty-aligned severe has deficit
0.0110 and is currently gated out; against a per-dataset threshold falling back to the
`STABILITY_FLOOR = 0.01` prespecified minimum-material effect, it opens.

## 6. Permutation p-values and bootstrap CIs disagree in the same row

PANDA severe, independent support: `R = -0.018`, `CI [-0.050, 0.005]`, `Holm p = 0.0001`. The CI
covers zero while the test is highly significant, because the two come from different resampling
schemes (partitioning-unit-block permutation vs. paired patient bootstrap) and are printed side by
side without reconciliation. A reader will notice.

Either report a permutation-based interval alongside the permutation p, or label the CI explicitly as
a dispersion summary that is not the test.

## Verification

- Add a test asserting standardization is computed within a dataset root, not across the pooled set.
- Add a test asserting two units differing only in tail assignment receive different nominal-shortage
  scores.
- Extend `tests/analysis/test_inference.py` with a Holm-family test asserting a dataset that opens
  two gates does not receive a larger family than the same dataset opening one.
- Re-derive thresholds from the existing artifacts and record the before/after gate routing, so the
  changed TCGA-UT outcome is a documented consequence rather than a surprise in the next report.

Run `clean-code` and the affected tests under `tests/analysis/`.

## Done when

All six corrections land with tests, and a before/after gate-routing table exists showing exactly
which units change status under the new thresholds and family construction.
