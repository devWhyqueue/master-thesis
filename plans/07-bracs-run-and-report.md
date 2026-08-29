# 07 — Wave 1: BRACS and TCGA-UT, report, and go/no-go

**Depends on:** 01–06 all landed. **Cluster cost:** the two multiclass datasets, together ~9 % of
PANDA's per-run budget. **Produces:** a decision on whether the two binary datasets are worth running.

## Why these two, and why together

Plan 03's replacement measurement reorders the priority, and plan 02's identifiability guard settles
it. Three facts combine:

| Dataset | Classes | Independent dose (spread arm) | Nominal sensitivity (old run, $D_{\mathrm{BA}}$) | Calibration gate | Updates |
| --- | ---: | --- | --- | --- | ---: |
| **BRACS** | 7 | **0.41–0.67** (weak) | **0.026–0.134** (strongest) | **6 of 6 units** | 3,960 → 6,600 |
| **TCGA-UT** | 30 | **1.46–1.79** (strong) | 0.0007–0.016 (weakest) | none | 10,500 → 17,500 |
| CAMELYON16 | 2 | 1.66–2.53 | 0.002–0.018 | none | 8,490 |
| PANDA | 2 | 0.85–0.89 | 0.026–0.146 | none | 257,790 |

1. **RQ3 can only see these two.** `_alignment_identifiable`
   ([rq3_wiring.py:61-70](../experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/predictors/rq3_wiring.py#L61-L70))
   excludes binary-target cells from both fits, because a Pearson correlation over two points is
   always ±1 and would confound the pooled alignment coefficient. CAMELYON16 and PANDA therefore
   contribute **nothing** to RQ3 no matter how strong their independent dose is. BRACS and TCGA-UT
   are not merely the cheapest pair — they are the only pair that gives RQ3 two groups.
2. **Their weaknesses are complementary, which is what makes the pair informative.** BRACS is the
   sensitive dataset at a weak dose; TCGA-UT is the insensitive dataset at a near-nominal dose
   (1.46–1.79 against `moderate`'s nominal `log(10) = 2.30`). Neither alone can separate *"the
   independent axis does not damage"* from *"this dataset does not respond"*. Together they can:
   a null on both, at opposite ends of both scales, is a real negative result.
3. **They are the two multiclass datasets**, so they are the only ones that exercise the tail
   assignment contrast and the head/body/tail incidence finding.

## The risk this wave exists to retire

The superseded narrowed design would have dosed the independent axis at roughly a quarter of the
weakest nominal dose, and the honest reading was that a quarter-dose may damage nothing. The spread
design changes that picture asymmetrically:

- On **TCGA-UT** the dose is 63–78 % of `moderate`'s nominal dose, across 29 of 30 classes. If the
  independent axis is going to damage anything anywhere, it should show here.
- On **BRACS** the dose is 18–29 % of it, across 6 of 7 classes — still weak, but on the dataset
  whose nominal deficits are an order of magnitude larger and whose probability-quality axis is the
  only one that exists.

Check the construction gate on BRACS first — cheapest, and the weakest measured dose, so it is the
arm most likely to trip a degeneracy check.

## Order

1. **Land 01–06 and verify locally.** `clean-code`, affected tests, `smoke`, and
   `submit --dry-run` to see the real array sizes with 5 controlled conditions.
2. **`freeze` BRACS** (3-split array). Highest-risk non-checkpointable stage — the spread pools add a
   second `designate_shared_patch_pools` probe per candidate total in `cap_feasible_shared_total`,
   which writes no intermediate state. Use `cpu-2d` or `cpu-7d`.
3. **`signals` → Gate A on BRACS** (below). Do not freeze TCGA-UT until BRACS passes it; the two
   freezes are independent, but a Gate A failure is a design failure and would apply to both.
4. **`freeze` → `signals` TCGA-UT**, then Gate A again.
5. **`match`** over both roots. Pooled standardization spans both datasets, so it must run after both
   have `signals`.
6. **`tune`** each dataset, all five conditions. `tune-wave` and `tune-decide` are host-only — they
   shell out to `sbatch` and must not run inside Apptainer. Gate on every condition's `tuning_locked`.
7. **`confirm` → `analyze`** per dataset, then `analyze-combine`, then `combine-rq3`.
8. **Gate B — the dose question** (below).
9. **Report.**

## Gate A — construction, before a single GPU hour

Per dataset:

- `reject_constant_signal_axes`, `reject_degenerate_spreading` and `reject_non_nested_pools` all pass;
- `signal_profile.json` shows a **non-zero, varying** `independent_shortage`;
- at least one unit's dominant shortage is independent support on a **positive raw score**;
- the spread pool is a strict patient superset of the concentrated pool with unchanged nominal counts;
- `pool_fraction_retained` is computed against the spread frame, not the concentrated one.

Any failure stops the plan. Nothing downstream is recoverable if the construction did not create the
shortage — that is the entire lesson of the original defect and of the superseded narrowed design.

## Gate B: three outcomes, three different next plans

Read the `balanced` deficit **against the `balanced_spread` reference**, per dataset, against that
dataset's own discrimination threshold (per-dataset since `eb5de78`).

**(a) The independent contrast opens a gate on either dataset, and independent-support weighting
behaves differently there than in the concentrated arm.** The taxonomy question is live. Proceed to
plan 08 for RQ1/RQ2 replication on the binary datasets.

**(b) It opens a gate, but independent-support weighting still recovers nothing.** The original
conclusion survives contact with a real independent shortage, which is a *stronger* result than the
current report can claim — it was previously untestable. Scale plan 08 down to CAMELYON16 only, or
close it and report the two multiclass datasets.

**(c) It opens no gate on either dataset.** A dose of 1.46–1.79 on 29 of 30 TCGA-UT classes, and
0.41–0.67 on the most nominal-sensitive dataset in the benchmark, damages nothing measurable.
**Stop.** Neither binary dataset can change that: CAMELYON16 has a comparable dose but a third of
BRACS's nominal sensitivity, and PANDA has a weaker dose than TCGA-UT at 39× the cost. Report the
bound itself — *the design can create an independent-support shortage approaching the nominal dose
and it does not damage the endpoints* — which is a real scope result and is now a much stronger
statement than the superseded plan could have made, because the dose is no longer a quarter of the
nominal one.

Outcome (c) is not the failure case. It is the case this wave exists to reach cheaply.

## What a two-dataset report can and cannot say

**Can:** RQ1 on both axes, including the calibration axis that exists only on BRACS; the full tail
assignment contrast on both datasets; RQ2's five-family separation and — for the first time — a
matched-versus-unmatched contrast with a real independent arm; calibration recovery; the exploratory
roster under exposure-matched budgets; **RQ3 with two dataset–target groups**, which is the minimum
at which `sigma_u` is estimable and leave-one-group-out has something to hold out.

**Cannot:**

- **The few-classes finding.** *"Damage is largest where the fixed budget splits between few
  classes"* rested on PANDA (0.146) against TCGA-UT (0.016) at the same nominal rho. Both binary
  datasets are out of wave 1, so this is unavailable.
- **Any four-group RQ3 variance decomposition.** Two groups estimate `sigma_u` but weakly; the
  leave-one-group-out check reduces to two single-group holdouts. Report it as such rather than
  reusing the previous report's framing.
- **Validation of the per-dataset gate thresholds** beyond these two datasets' own.

Amend `2_benchmark_patch_protocol.tex` with a scope note stating which datasets ran and why, rather
than leaving a reader to infer it from a missing section.

## Report

Rewrite `3_benchmark_patch_results.tex` in place, scoped to the two datasets. Maintaining a second
divergent results document is worse than rewriting once now and again if plan 08 lands.

Sections whose conclusions change:

- **Realized signal profiles** — the identically-zero independent column is replaced by the crossed
  design; the ambiguous-designation count moves; the argmax sign guard changes labels. The report's
  current §2.1 paragraph explaining that the axis was never varied must be replaced by the achieved
  doses, not merely softened.
- **Matched against unmatched signals** — previously *"the contrast measures whether the unit happened
  to land in the one corner served by a working method"*. With a real independent arm this becomes an
  actual test of the matching hypothesis, and the independent-support member can be genuinely matched
  for the first time.
- **Focal loss** — appears throughout the old report as recovering nothing. After plan 01 that is a
  real measurement for the first time.
- **The wider method roster** — under exposure matching, cRT and OKO lose budget, so the "no unmet
  need" conclusion is expected to move. Four of the exploratory recoverers must be disclosed as one
  mechanism (plan 05, item 3).
- **Discussion** — requirement 3 (per-condition estimation) rested on RQ3's leave-one-group-out,
  which now exists but with two groups; restate it at the strength two groups support.

Per CLAUDE.md: use the `/scientific-writing` skill, compile from the report directory into a
temporary build directory, copy out only the PDF, leave no auxiliary files in the worktree.

**Housekeeping:** `2_benchmark_protocol_patch.*` auxiliary files are still sitting in
`experiments/2_benchmark_patch/report/`, against the CLAUDE.md rule. Clean them.

## Done when

BRACS and TCGA-UT complete freeze through `analyze`, `match` and `combine-rq3` run over both roots,
Gate A and Gate B both have recorded answers, the results report is rewritten to two-dataset scope
with RQ3 fitted at two groups, and plan 08 is either scoped to an outcome or closed.
