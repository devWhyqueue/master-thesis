# Controlled Power-Law Imbalance Experiment (follow-up)

This file preserves the design of the **controlled class-imbalance severity sweep**
that was originally drafted into the design-dataset report. It has been split out
so the main report can focus on *extending the class-imbalance mitigation benchmark
to additional native datasets* (BRACS first). The controlled construction below is a
separate follow-up experiment and is not part of the current report.

Status: **deferred**. The TCGA-UT power-law results and the BRACS power-law results
already exist on the cluster and in `report/outputs/` (TCGA-UT) and
`report/outputs/bracs_power_law/` (BRACS). Only the framing was removed; the numbers
can be reused when this experiment is written up on its own.

## Motivation

A native class distribution is realistic but confounds several factors at once: how
often each class appears, tumor morphology, tissue diversity, slide acquisition, and
the representation geometry of the frozen encoder. A controlled construction isolates
one factor — training-support frequency — by imposing known power-law distributions
while holding validation and test data fixed. The question it answers cleanly: **do
mitigation methods keep their relative advantage as the imbalance between frequent
and rare classes intensifies along a fixed class ordering?**

## TCGA-UT controlled severity benchmark

- Shares the TCGA-UT case-disjoint split protocol with the companion mitigation study
  (`queisler2026imbalanceTCGA`). All 32 cancer types are retained.
- For each of three seeds, validation and test splits are held fixed; artificial
  imbalance is applied only to the training split. Assignment is patient-disjoint (all
  slides/feature rows for one TCGA participant stay in one partition).
- After slide selection, input evidence is capped at 30 deterministic patches (patch
  regime) or 30 frozen feature instances (WSI-bag regime) per slide. Virchow2 features
  stay frozen, represented as CLS ⊕ mean(patch tokens) = 2560-dim vectors.

### Power-law training support

For an ordered class list `(c_1, …, c_C)`, the intended share for rank `i` under
severity `λ` is

```
p_i(λ) = i^(-λ) / Σ_j j^(-λ)
```

- `λ = 0` → uniform over ranks; larger `λ` → more support to early-ranked (frequent)
  classes, less to late-ranked (rare) classes.
- Classes are ranked by **native-prevalence order** (rank 1 = most frequent cancer
  type, rank 32 = least frequent).
- The power-law form is imposed to obtain controlled, monotonic severity steps; it is
  not a fit to the native TCGA-UT distribution (only moderately imbalanced, ≈28:1, and
  not well described by a single power law).

Integer allocation: raw targets `N·p_i(λ)` are floored, each class first receives one
guaranteed slide, and the remaining `N − C` slides are assigned by largest fractional
remainder. Constraints for every class `c`:

```
1 ≤ n_c ≤ a_c ,   Σ_c n_c = N
```

where `a_c` is the number of available training slides for class `c` and `N` is fixed
across all severities and seeds.

### Feasible pool size

The maximum feasible pool for a given `λ` and seed is `T*(λ) = min_c a_c / p_c(λ)`.
This continuous bound is verified against the full integer-allocation rule. The binding
regime is `λ = 0.5` (not the steepest): its flat profile requires every class —
including rare ones — to contribute a non-trivial share, saturating the rarest class's
supply at a lower `N`. For TCGA-UT this yields **N = 1132 slides** (≈18 % of the full
training pool).

The three `λ ∈ {0.5, 1.0, 1.5}` produce head-to-tail ratios of ≈6:1, ≈30:1, ≈122:1:
`λ=0.5` sits below the native ≈28:1; `λ=1.0` reproduces near-native severity; `λ=1.5`
is substantially more skewed than anything native.

## BRACS power-law fallback

BRACS was first checked as a native benchmark. Its prepared seven-subtype WSI support
spans PB (196 slides) to DCIS (87 slides), a ratio of ≈2.25:1 — below the 5:1 minimum
chosen for a "meaningful native imbalance" setting. Under the controlled-experiment
framing it therefore switched to the same power-law construction, with a feasible pool
of **108 WSIs** and constructed head-to-tail ratios ≈3:1, ≈6:1, ≈14:1 — a narrower,
milder range than TCGA-UT, reflecting the smaller label set and shallower per-class
supply.

(Note: in the *current* report this switch is dropped — BRACS is reported on its native
distribution instead. The power-law BRACS numbers remain valid for this follow-up.)

## Evaluation protocol

- Each constructed split is indexed by seed and `λ`. Hyperparameters are selected
  independently per method and per `λ`: for each candidate config, validation macro F1
  is averaged over the three seeds and the highest mean is chosen (ties broken by mean
  validation balanced accuracy). One config is fixed per method and `λ` (shared across
  seeds), not reused across `λ`.
- Because tuning is per-severity, reported figures reflect each method at its
  per-`λ` best (achievable rather than fixed-configuration performance).
- Selected configs are evaluated on the fixed test split; results are mean ± sd across
  seeds. Macro F1 is the primary metric; balanced accuracy and accuracy are secondary.
- Calibration: NLL, multiclass Brier, ECE; one temperature fit per method/seed by
  minimizing validation NLL, applied to the test scores before softmax; ECE reported
  before and after temperature scaling.

## Known findings (from existing runs)

Patch regime (TCGA-UT): balanced accuracy declines monotonically with `λ` (≈8–11 pp
from `λ=0.5` to `λ=1.5`); method rankings largely preserved. ProGAN augmentation leads
at mild severity but reverses to worst at the harshest (steepest decline, ≈10.5 pp);
OKO and CE+soft-F1 most robust. The severity effect concentrates in the tail classes;
head-tier F1 is essentially flat.

WSI-bag regime (TCGA-UT): ≈10 balanced-accuracy points above patch at equal severity.
RankMix most severity-robust; focal MIL least stable; MDE-MIL's advantage over loss
weighting erodes at `λ=1.5`.

BRACS power-law: patch severity decline ≈half of TCGA-UT (narrower range); OKO retains
lead; ProGAN does not reverse to worst. WSI-bag responses non-monotone with high
between-seed variance (±0.09–0.10), attributed to the shallow 108-slide pool — a
pool-depth requirement for interpretable severity comparisons.

## Artifacts

- TCGA-UT constructed outputs: `report/outputs/tables/` (`result_summary_*`,
  `result_tail_class_*`, `result_calibration_*`, `constructed_split_summary`) and
  `report/outputs/figures/` (`constructed_support_by_rank`, `support_vs_recall`).
- BRACS power-law outputs: `report/outputs/bracs_power_law/`.
- Code: `analysis/evaluation/tuning_run.py`, `tuning_grid.py`, `tuning_aggregate.py`;
  power-law construction in `data/full_scale/` and `data/bracs/power_law.py`.
