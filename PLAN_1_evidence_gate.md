# Plan 1 — Open BRACS for inference: case-macro estimator + pooled evidence cell

**Scope:** re-analysis only. No retraining, no new splits, no GPU.
**Depends on:** nothing. **Blocks:** the blind re-run shared with Plan 3.

## Context

The patch benchmark yields exactly one inference-grade dataset, PANDA, which is binary. At K=2
the five-signal confirmatory contrast of `sec:five-family` is unidentifiable: prevalence,
nominal support and independent support all collapse to a single head/tail ratio, and difficulty
and diversity are two-point orderings. Extra severity on PANDA cannot recover the signal axis
RQ2 exists to test — a multiclass dataset has to clear inference.

BRACS, CAMELYON16 and TCGA-UT are blocked by `is_descriptive_only`, computed at freeze from the
**test manifest alone** ([preflight.py:56](experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/inference/preflight.py#L56);
identity built at [freeze_execution.py:229-235](experiments/2_benchmark_patch/code/imbalance_benchmark/commands/freeze_execution.py#L229-L235)).
It never sees training data, so no severity change can move it. Two mismatches make it fire:

1. **Patch-micro weighting.** `_class_preflight` weights a case by its patch count
   ([preflight.py:35](experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/inference/preflight.py#L35)),
   so one large ROI or one macro-metastasis slide carries >50% of a class's weight.
   CAMELYON16 tumor: `frac_dominant` 0.66–0.69 on 14–20 slides.
2. **Per-split cell.** `is_descriptive_only` takes `any()` over split×class cells
   ([preflight.py:166](experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/inference/preflight.py#L166)),
   but `BootstrapContext` draws one shared weight per case over the **union** of the three test
   frames ([context.py:72-80](experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/inference/context.py#L72-L80))
   and the confirmatory effect is the split-averaged one. The per-split cell certifies an
   estimand nobody reports. The pooled `by_class` diagnostic is already computed at
   [preflight.py:189](experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/inference/preflight.py#L189)
   and never read.

Both are corrections of a mismatch between a diagnostic and the estimator it certifies,
justified independently of any outcome.

## Verified outcome

The preflight statistics are invariant to the Dirichlet normaliser: with `w_i = n·g_i / Σ_j g_j`
(`g ~ Exp(1)`, as in `resample_patient_weights`) and per-case multiplicity `m_i`,

```
kish = (Σ m_i w_i)² / Σ (m_i w_i)² = (Σ m_i g_i)² / Σ (m_i g_i)²
```

so `n` cancels and each class reduces to its own `m` Exp(1) draws. Simulating that exactly
(10,000 replicates) reproduces `_class_preflight`. Classes failing the evidence rule:

| scenario | BRACS | CAM16 | TCGA-UT |
|---|---|---|---|
| today (patch-micro, per-split cell) | 7/7 | 1/2 | 11/31 |
| case-macro only | 7/7 | 1/2 | 10/31 |
| pooled cell only | 4/7 | 1/2 | 0/31 |
| **both (this plan)** | **0/7** | **0/2** | **1/31** |

Neither change alone rescues BRACS. TCGA-UT's residual failure is Cholangiocarcinoma
(m=10 pooled, p2.5 kish 3.36) — handled in Plan 3.

## Changes

### 1. Case-macro estimators — `analysis/inference/bootstrap.py`, `context.py`

The case-macro endpoints in `analysis/reporting/clustered_endpoints.py` are unweighted point
estimates in the reporting layer only; the gate path uses patch-micro
`weighted_balanced_accuracy` / `weighted_macro_nll`.

No new machinery. `PatientWeights.sums(values, mask)` already takes a per-row value vector.
With `v(row) = 1 / rows_of_that_case_within_this_class`:

```
denominator = sums(v, mask)           = Σ_c w_c
numerator   = sums(v · correct, mask) = Σ_c w_c · r_c     (r_c = case c's correct fraction)
recall_k    = numerator / denominator                      # case-macro
```

Same substitution for macro NLL with `v(row) = nll(row) / rows_of_case_in_class`.

- `bootstrap.py` — one helper computing the divisor from `(labels, case_ids)`;
  `weighted_balanced_accuracy` and `weighted_macro_nll` apply it inside their existing
  per-class loops.
- `context.py` — compute the divisor once in `BootstrapContext.__init__` (it already holds
  `self.case_ids`) and thread it through `ba_distribution` and `tail_nll_distribution`.

Leave `weighted_ece` and the secondary-interval endpoints alone: ECE is secondary and never
enters a gate. Mark the divisor helper with a `ponytail:` comment naming its ceiling — it
assumes rows of one case within one class are exchangeable, which is what case-macro means.

### 2. Matching permutation statistic

The p-value must test the statistic the effect reports, so the divisor
`1 / (n_classes · n_cases_in_class · rows_of_case_in_class)` replaces the patch-count divisor at:

- [permutation.py:125](experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/inference/permutation.py#L125)
  and [:167](experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/inference/permutation.py#L167)
  (`n_classes * mask.sum()`, `len(present_tails) * mask.sum()`). `crossed_permutation.py`
  delegates to `_ba_patient_contributions` / `_tail_nll_patient_contributions`, so one fix
  covers both.
- `analysis/inference/confirmatory/arms.py` — `_member_ba_observed` and
  `_member_tail_nll_observed` compute observed values patch-micro and need the same treatment.

The additive patient-contribution structure is unchanged, so `_contribution_p_value` and
`_crossed_contributions` need no edit.

### 3. Preflight matched to the estimator — `analysis/inference/preflight.py`

- Line 35: drop `multiplicity[:, None] *`. A case's cell weight is its Dirichlet weight;
  multiplicity becomes reporting-only.
- Line 166: designate from `by_class`, not `by_split_class`. Keep `by_split_class` in the written
  report, and keep it feeding `all_split_level_metrics_computable` — a separate validity check
  consumed by `require_valid_preflight` that must not be weakened.
- Docstring: state that the cell now matches the shared case-level draw of
  `resample_patient_weights`.

**Thresholds unchanged**: `p2.5 kish ≥ 5`, `frac_dominant ≤ 0.05`, `uniq ≥ 5`, and
`DISCRIMINATION_THRESHOLD = 0.02` / `CALIBRATION_THRESHOLD = 0.05` in `gates.py`.

### 4. Freeze regeneration

`is_descriptive_only` is stored in the signed freeze and read by `gates_and_recovery`, so the
freeze must be regenerated. Re-run the existing `freeze` command rather than adding a surgical
preflight-only path: zero new code, and the drift check is needed either way.

**Verify** every field other than `bootstrap_preflight` is byte-identical to the stored freeze
(per-condition manifest hashes, `shared_T`). If construction is not bit-reproducible, stop — a
drifted manifest invalidates the tuning selections and the no-retraining premise. Fallback is a
`refreeze-preflight` command recomputing only the preflight against the existing signed freeze,
recording the prior hash in the `supersedes` / `superseded_freeze_file_hashes` fields the schema
already carries.

### 5. Protocol — `report/2_benchmark_protocol_patch.tex`

Needs explicit go-ahead: `code/CLAUDE.md` names the protocol the authority and forbids editing it
unasked.

- `tab:locked`: gate aggregation unit (case-macro) and evidence-cell definition (pooled across
  split appearances).
- Discrimination-endpoints section: promote case-macro from a second summarisation to the gate
  estimator; patch-micro becomes secondary.
- Uncertainty section: the evidence diagnostic is evaluated on the same pooled case set the
  shared weight draw uses.
- Limitations: replace the BRACS descriptive-exclusion paragraph; add CAMELYON16's saturation
  (deficit well under 0.02 at ρ=100 — an effect-size limit, not an evidence limit).

No disclosure paragraph; the blind re-run is the control on the revision.

## Blind re-run

Do not read recovery output before the recomputed gates.

1. **Quarantine, do not delete.** Move to a timestamped `superseded/` directory, for all four
   datasets (the estimator changed for PANDA too): `split=*/data/gates_and_recovery.json`,
   `data/cross_split_gates_and_recovery.json`, `tables/`, `figures/`, `outputs/rq3_combined/`.
   `results.sqlite` and every manifest stay untouched.
2. Run `freeze` (BRACS / CAMELYON16 / PANDA, 3 splits each) and the drift check.
3. Confirm the regenerated `bootstrap_preflight.json` before running `analyze`.
4. `analyze` → `analyze-combine`. `match` and `combine-rq3` only once Plan 3's TCGA-UT is done.
5. Read `method == "ce"` gate rows first (`gate`, `gate_passed`, `effect`, `ci`). Only then recovery.

Compute: `freeze` on `cpu-2d` (BRACS / CAMELYON16) and `cpu-5h` (PANDA); `analyze` on `cpu-5h`,
PANDA on `cpu-2d` per the note in `configs/panda_patch.yaml`. No GPU.
`analyze` has no checkpointing — on TIMEOUT jump straight to the longest partition.
Cross-check `squeue` before any resubmit; the BRACS chain 4715349–51 must finish or be cancelled
first, since its outputs are among those quarantined.

## Verification

**Tests** — these encode the semantics being changed and must be updated, not deleted:
`test_preflight_is_descriptive_when_any_split_class_fails_kish_threshold` (pooled-cell semantics),
`test_descriptive_only_cell_never_opens_a_gate_or_permutes` (same behaviour, new designation
source), `test_kish_preflight_*` (case-macro weights).

**One new runnable check** — the smallest thing that fails if the estimator is wrong: at
replicate 0 every case weight is 1, so case-macro `weighted_balanced_accuracy` must equal
`_macro_classification(...)[0]` from `clustered_endpoints.py` exactly. Assert on a fixture with
deliberately unequal rows-per-case, and assert the permutation `_ba_observed` matches it.

```powershell
uv run pytest experiments/2_benchmark_patch/code/tests/analysis
uv run python "$env:USERPROFILE\.codex\skills\clean-code\run.py" --scope experiments/2_benchmark_patch/code --vulture-scope experiments/2_benchmark_patch/code
```

**Freeze determinism:** all fields except `bootstrap_preflight` byte-identical; `shared_T`
unchanged (BRACS 16796, CAMELYON16 36127).

**Preflight against the simulation** — `is_descriptive_only: false` for BRACS and CAMELYON16,
tightest cells near:

| dataset | tightest class | m | expected p2.5 kish |
|---|---|---|---|
| BRACS | DCIS | 16 | ≈ 5.5 |
| BRACS | FEA | 17 | ≈ 5.9 |
| CAMELYON16 | tumor | 46 | ≈ 17.5 |

A materially lower value means the pooled cell is not being formed over the union of split
appearances.

**Gate outcome.** BRACS should open 6/6 discrimination cells. Under the pre-sync pipeline its
case-macro deficits were 0.027–0.134 against a 0.02 threshold, so the margin is wide — but treat
those as indicative, not as targets to reproduce (see caveat).

**Success condition:** BRACS carries a confirmatory Holm family with all five signal-contrast
methods at K=7. That, not the dataset count, is what makes RQ2 answerable.

## Caveat on prior numbers

`cross_split_gates_and_recovery.json` on the cluster dates 2026-08-02 to 08-16, but the analysis
code changed 08-17 (`signals`, `match`, `confirmatory/`, `crossed_permutation.py`). Every deficit
and recovery figure quoted from those files predates the current pipeline. The evidence-gate
simulation is unaffected — it reads manifests, and `preflight.py` is unchanged in every line it
depends on.

## Risks

- **Do not adjust `DISCRIMINATION_THRESHOLD` to rescue a cell.** The threshold is protocol.
- **CAMELYON16 gains an open evidence gate and still shows nothing.** Report it as a saturation
  result, not a null mitigation result.
- **PANDA's published numbers change** because the estimator changed for every dataset. Budget
  for re-reporting PANDA in the results text.
- **Freeze drift** would invalidate the no-retraining premise — check before `analyze`, not after.
