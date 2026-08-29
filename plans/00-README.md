# Patch-benchmark remediation plans

`experiments/2_benchmark_patch` ran to completion and was written up in
[3_benchmark_patch_results.tex](../experiments/2_benchmark_patch/report/3_benchmark_patch_results.tex).
Reading the results back against the protocol and the code surfaced four verified bugs and a set of
protocol choices that bias the conclusions. The benchmark's two headline claims — *only prevalence
and nominal support carry recovery*, and *no gate-passing condition resists repair, so there is no
unmet need* — both rest on machinery that is broken or confounded.

Eight plans, ordered so nothing expensive runs before the thing that decides what it should measure —
and so the programme can be **stopped after one wave** if the answer is already in.

## Status

| # | Plan | Cluster cost | State |
| --- | --- | --- | --- |
| 01 | [Code bug fixes](01-code-bug-fixes.md) | none | **done** (`a2ea309`) |
| 02 | [Analysis-layer corrections](02-analysis-corrections.md) | none | **done** (`2f21e5c`, `eb5de78`) |
| 03 | [Independent-support feasibility](03-independent-support-feasibility.md) | read-only jobs | **re-decided** — narrowing refuted, spreading measured |
| 04 | [Crossed condition family](04-crossed-condition-family.md) | none yet | **needs rework** — `15b0f35` built the superseded narrowed arm |
| 05 | [Budget and tuning protocol](05-budget-and-tuning-protocol.md) | none | pending — must land before 07 |
| 06 | [Method implementation fixes](06-method-implementation-fixes.md) | none | pending — must land before 07 |
| 07 | [Wave 1: BRACS + TCGA-UT](07-bracs-run-and-report.md) | the two multiclass datasets | pending |
| 08 | [Wave 2: the binary datasets](08-remaining-datasets.md) | the expensive part | **conditional on 07** |

```
01 ─┬─ 05 ─┐
    ├─ 06 ─┤
02 ─────────┤
            ├─ 07 BRACS+TCGA-UT ──[Gate B]── 08 CAMELYON16 (+PANDA) / scaled down / closed
03 ─ 04 ────┘
```

## What changed, and why the ordering moved

Plan 03 originally proposed creating the independent-support shortage by **narrowing** the evidence
pool, and its per-assignment proxy ranked BRACS as the only full-factorial carrier. Plan 04 was
implemented against that decision (`15b0f35`) and the real crossed construction refuted it: the
globally feasible narrowing ratio on BRACS is **1.0000**, because `_expand_pool` already builds the
minimal patient set that supplies the cross-assignment maximum count. The pool cannot be narrowed
because it is *already* the narrowest the contribution caps permit — on all four datasets, not just
BRACS.

Plan 03 now creates the axis by **spreading** instead: the concentrated pool stays as the deprived
arm, and a new spread arm distributes the same nominal patch counts over the full eligible patient
inventory. Spreading pushes against inventory rather than against the caps, and it *lowers* per-unit
contribution, so it moves away from the constraint that blocked narrowing. Measured achieved doses,
against `moderate`'s nominal `log(10) = 2.30`:

| Dataset | Classes | Independent dose | Nominal sensitivity (old run) | Calibration gate | Updates |
| --- | ---: | --- | --- | --- | ---: |
| **BRACS** | 7 | 0.41–0.67 | **0.026–0.134** (strongest) | **6 of 6 units** | 3,960 |
| **TCGA-UT** | 30 | **1.46–1.79** | 0.0007–0.016 (weakest) | none | 10,500 |
| CAMELYON16 | 2 | 1.66–2.53 | 0.002–0.018 | none | 8,490 |
| PANDA | 2 | 0.85–0.89 | 0.026–0.146 | none | 257,790 |

Every dataset now carries the axis in every assignment, so the per-dataset scoping map becomes a cost
knob rather than a feasibility gate.

## Why BRACS and TCGA-UT together

Three facts, and the first is decisive:

1. **RQ3 can only see these two.** `_alignment_identifiable` excludes binary-target cells from both
   fits, because a Pearson correlation over two points is always ±1. CAMELYON16 and PANDA contribute
   nothing to RQ3 regardless of their independent dose. BRACS and TCGA-UT are not merely the cheapest
   pair — they are the only pair that gives RQ3 two groups.
2. **Their weaknesses are complementary.** BRACS is the sensitive dataset at a weak dose; TCGA-UT is
   the insensitive dataset at a near-nominal dose. Neither alone separates *"the independent axis does
   not damage"* from *"this dataset does not respond"*. Together they can.
3. **They are the two multiclass datasets**, so they are the only ones that exercise the tail
   assignment contrast and the head/body/tail incidence finding.

Together they cost about 9 % of PANDA's per-run budget.

## The question wave 1 exists to answer cheaply

The superseded narrowed design would have dosed the independent axis at roughly a quarter of the
weakest nominal dose, and a quarter-dose may damage nothing. The spread design removes that
objection on TCGA-UT, where the dose reaches 63–78 % of `moderate`'s nominal dose across 29 of 30
classes. If the independent axis damages anything anywhere, it should show there; and if it does not,
the negative result is now strong rather than merely under-powered.

## Verified bugs, for reference

| | Defect | Site |
| --- | --- | --- |
| B | A locked strength of `0.0` trained at `1.0` (`float(param or 1.0)`; `0.0` is falsy). 13 selection units, 10 of them focal loss, including all six TCGA-UT conditions. | `modeling/training/__init__.py:101,104,107` |
| B2 | The soft hybrids were CE-anchored but train under a forced balanced sampler, so their strength-0 point is balanced-sampling CE and the alias substituted the wrong metrics. | `modeling/workflows/tuning/search_windows.py:35-37` |
| C | Post-hoc logit adjustment discarded its tuned `tau` on the calibration axis, making its probabilities identical to CE's and its tail-NLL recovery exactly `0.000 [0.000, 0.000]`. | `analysis/calibration.py:195` |
| A | Independent-support shortage was structurally zero in all 18 units, and the unguarded argmax let that constant-zero column *win* the dominant-shortage label. | `analysis/predictors/rq3_features.py:54-72`, `analysis/predictors/signals/matching.py:109-126` |

## The trap, repeated because it would silently waste a run

Every shortage in `rq3_features.py` averages over `_deprived_classes`, defined by **nominal** count
below balanced count. Between `balanced` and `balanced_spread` no class is nominally deprived, so the
set is empty and `_independent_shortage` returns `0.0` — defect A reproduced inside the cell built to
fix it. Each shortage needs its own deprivation set. Plan 04 covers it; plan 07's Gate A verifies it
before any GPU time is spent.

## The methodological lesson from the superseded decision

A feasibility claim about pool designation must come from `designate_patch_pool` itself, run against
`required_counts` unioned over every condition **and every assignment**. The superseded decision used
a per-class, per-assignment proxy and a bound (`MIN_POOL_PATIENTS / P_wide`) that assumed the pool
had slack it never had. Both errors are cheap to avoid and expensive to discover downstream — the
narrowed arm reached a full implementation before the real construction refuted it.

## What a wave-1 report cannot say

- **The few-classes finding.** *"Damage is largest where the fixed budget splits between few
  classes"* rests on PANDA against TCGA-UT at equal nominal rho. Both binary datasets are in wave 2.
- **A four-group RQ3 variance decomposition.** Two groups estimate `sigma_u` weakly and reduce
  leave-one-group-out to two single-group holdouts. Wave 2 does not improve this: the binary datasets
  are excluded from RQ3 by construction, so **two groups is the ceiling for this benchmark** unless a
  third multiclass dataset is added.
- **Validation of the per-dataset gate thresholds** beyond the two datasets' own.
