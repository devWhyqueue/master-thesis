# Plan 3 — TCGA-UT at K=30 and a re-anchored deficit threshold

**Scope:** a two-line threshold change that applies to every dataset, one config-level class
exclusion, and a TCGA-UT re-run at the existing three conditions. The only GPU cost in the three
plans.
**Depends on:** Plan 1 (estimator + evidence cell frozen first) and Plan 2 (banks the
natural-condition saving inside this re-run).

## Context

Plan 1 opens BRACS for inference. TCGA-UT is the second multiclass candidate — K=30 after the
exclusion below, the widest available test of the RQ2 signal axis — but two things stop it:

1. **Cholangiocarcinoma** is the single class still failing the evidence rule after Plan 1
   (m=10 pooled test participants, p2.5 kish 3.36 against a floor of 5).
2. **Its deficits sit just under the gate.** Under the pre-sync pipeline the case-macro
   balanced-minus-imbalanced balanced accuracy was ≈0.003–0.006 at ρ=10 and 0.014–0.022 at
   ρ=100, against a 0.02 threshold. One cell of six cleared, marginally.

An earlier draft of this plan answered (2) with a fourth severity condition, ρ=500, for TCGA-UT
alone. That is rejected: a severity level that exists only on the dataset that needs it is a
selection rule dressed as a design choice, and it forces a per-dataset condition-set refactor
across ten call sites plus a SLURM array-indexing hazard, for one dataset's benefit.

The honest target is instead the number the deficit is compared against. `DISCRIMINATION_THRESHOLD
= 0.02` and `CALIBRATION_THRESHOLD = 0.05` were set without a stated derivation. They are protocol
and cannot be moved to rescue a cell — but they can be *derived*, once, from constants the
protocol already fixed, applied identically to all four datasets, and frozen before any gate row
is read. That is §2.

## 1. Class exclusion: K = 31 → 30

Use the existing construction-time mechanism, exactly as DLBCL already is. No new config key, no
new estimand caveat.

- Add Cholangiocarcinoma to `dataset.excluded_classes` in `configs/tcga_ut_patch.yaml` and record
  the reason in `dataset.eligibility_rules.class_exclusions`.
- Consumed by `apply_class_exclusions` at
  [prepare.py:133-136](experiments/2_benchmark_patch/code/imbalance_benchmark/commands/prepare.py#L133-L136),
  before splitting and construction. No code change.
- **Two exclusions, two rationales — the protocol must say both.** DLBCL: no shared total is
  both tolerance- and cap-feasible for it. Cholangiocarcinoma: its held-out participant count
  cannot support split-unit-clustered inference. Do not merge them into one reason.
- Side effect to state plainly: the class leaves training too, which is stronger than the
  evidence problem requires. Accepted for a uniform 30-type task definition.

Because `assign_class_splits` is seeded per class, the 30 retained classes keep their existing
split assignments; only K and the allocation change. `shared_T` will move — Cholangiocarcinoma
is among the most cap-constrained classes, so its departure should loosen the tolerance search.

Severity targets stay `{1, 10, 100}` for TCGA-UT, identical to the other three datasets.

## 2. Deficit thresholds: derive them, once, for all datasets

### Why 0.02 is not defensible as written

Three problems, none of them about TCGA-UT:

1. **The protocol already declares a different materiality constant for the same quantity.**
   The support pilot's stability floor (§`app:construction-support`, protocol line 537) defines
   an increment of training support as immaterial when it "changes mean balanced accuracy by
   less than \num{0.01} and every class recall by less than \num{0.02}". That is the benchmark's
   own prespecified answer to *how large a balanced-accuracy change induced by changing training
   support counts as real* — 0.01. The deficit gate asks the same question about the same
   endpoint under the same kind of manipulation and answers 0.02. One of the two is wrong, and
   the pilot constant is the one that was fixed first, before any definitive fit, and used to
   set the support floors the whole design rests on.
2. **The gate is scaled above the interventions it exists to enable.** Long-tailed mitigation
   methods report gains of roughly 1–3 accuracy points over an ERM baseline on the standard
   many-class benchmarks (CIFAR-100-LT, ImageNet-LT, iNaturalist; class-balanced loss, LDAM-DRW,
   decoupled training, logit adjustment). A gate requiring 0.02 of damage before a recovery
   fraction may be computed demands damage at the top of the range of the entire effect the
   family of methods produces. The gate's job is to keep the recovery denominator away from
   zero, not to require damage larger than any method could repair.
3. **It has no relation to the measurement precision of this pipeline.** A materiality threshold
   should sit above the noise floor of the estimator; nothing about 0.02 was checked against it.

### The rule

```
DISCRIMINATION_THRESHOLD = max(0.01, 2 · σ_seed)
```

where `σ_seed` is the largest across datasets and splits of the standard deviation of
**balanced-condition CE case-macro balanced accuracy across the five confirmation seeds**.

- The 0.01 floor is the pilot stability constant, transplanted unchanged. Same endpoint, same
  kind of manipulation, a constant already locked in the protocol — no new arbitrary number is
  introduced.
- The `2 · σ_seed` term is the data-driven part: a deficit below twice the run-to-run spread of
  a *fixed* configuration is not distinguishable from re-running the same experiment. Expected
  to be the non-binding term with frozen Virchow2 features; if it binds, the threshold goes up,
  not down, and that is reported.
- Both terms are one global number, not a per-dataset one. A per-dataset threshold would
  reintroduce exactly the special-casing that killed ρ=500.

**Blinding.** `σ_seed` is a within-condition dispersion of the balanced arm only. It never touches
a balanced-minus-imbalanced contrast, a gate row, or a recovery ratio, so it can be computed
inside the Plan 1 re-analysis without unblinding it. Compute it, fix the threshold, write it into
`gates.py` and the protocol, and only then read gate output. Order matters more than the value
here: a threshold set after seeing which cells it opens is worthless regardless of how it was
derived.

### Calibration threshold: keep 0.05 nats, state what it means

No internal constant anchors the NLL scale, so 0.05 stays — but with a rationale replacing the
silence. A tail-group macro-NLL increase of 0.05 nats is a factor `exp(0.05) ≈ 1.051`, i.e. a
**5% relative reduction in the geometric-mean probability the model assigns to true tail-class
labels**. That is a statable minimum material loss of probability quality, on a scale where
absolute nats mean nothing on their own. Apply the same noise-floor sanity check
(`σ_seed` of balanced-condition tail NLL); if it exceeds 0.025, raise the threshold to twice it
and say so.

The two thresholds are not on a common scale and do not have to move together.

### Expected consequence

- TCGA-UT ρ=100: the pre-sync case-macro deficits 0.014–0.022 clear 0.01 in all six cells, at a
  1.4–2.2× margin rather than a coin flip. ρ=10 (0.003–0.006) still fails — which is the correct
  answer, and preserves the dose-response reading across ρ.
- BRACS (0.027–0.134 pre-sync): unaffected, already far clear.
- **CAMELYON16 is the cell to watch.** Plan 1 predicts it opens its evidence gate and still shows
  a deficit "well under 0.02". At 0.01 it may now open the discrimination gate, which changes
  Plan 1's planned saturation-result framing. This is a consequence to accept and report, not a
  reason to keep 0.02 — but note it before running, not after.
- PANDA: re-reported at the new threshold along with everything else.

## 3. Re-run

Only TCGA-UT, and only because of §1 — §2 is analysis-only and is carried by the Plan 1
re-analysis for the other three datasets.

**Decision point before spending anything.** Run Plan 1's re-analysis with the new threshold
first. If BRACS opens its discrimination cells at K=7, the RQ2 signal axis is already answerable
and the TCGA-UT re-run is a breadth improvement (K=30, the widest signal-profile spread in the
design) rather than a prerequisite. Decide with that in hand.

Order, if it goes ahead:

1. Quarantine TCGA-UT outputs alongside Plan 1's (same `superseded/` directory).
2. `prepare` → `pilot` → `freeze`.
3. **Stop and read the freeze report** — achieved ρ, limiting class, binding floor, new
   `shared_T`, and `is_descriptive_only: false`. `tune` is the expensive stage; do not launch it
   on an unread freeze.
4. `signals` → `tune` → `confirm` → `analyze`.
5. `match` and `combine-rq3` once all four datasets have finished `analyze`.

Compute: the observed TCGA-UT tuning bill was 763 GPU-h over four conditions; Plan 2 returns ~36
on natural and K drops by one class, so budget **~730 GPU-h for tuning** plus `confirm` — the
same order as the original run, with no fourth condition to pay for. `pilot` on gpu-5h, `tune`
and `confirm` on the gpu-2h arrays, `freeze` on cpu-2d, `signals` at freeze-sized memory (256G —
see `configs/panda_patch.yaml`'s note), `analyze` on cpu-5h escalating straight to the longest
partition on TIMEOUT since it has no checkpointing.

If tuning risks its wall, cut shards-per-task to 2 rather than moving partition. Any straggler
resubmit must reuse the `tune_shards_per_task` the array was submitted with. Cross-check `squeue`
before every resubmit.

## 4. Code

Two constants in
[gates.py:35-36](experiments/2_benchmark_patch/code/imbalance_benchmark/analysis/inference/gates.py#L35-L36),
plus the docstrings on `discrimination_gate` / `calibration_gate` that quote the numbers, plus the
comment above them pointing at the protocol section. Nothing else reads them —
`grep DISCRIMINATION_THRESHOLD` outside `gates.py` returns nothing, and no test hard-codes 0.02
as a boundary (`test_inference.py:34` uses effect 0.2, comfortably clear either way). No
per-dataset condition machinery, no `severity_targets` config key, no SLURM array change.

One config file: `configs/tcga_ut_patch.yaml`.

## 5. Protocol

- **New `tab:locked` row**: "Deficit gate thresholds — $D_{\mathrm{BA}}\geq0.01$;
  $D_{\mathrm{cal}}\geq0.05$ nats; both with CI excluding 0". The table currently locks the
  gate *aggregation unit* but not the thresholds themselves, which is part of how they drifted
  in unexamined.
- `fig:gate-routing` node label (line 349) and the gate paragraph (line 362): 0.02 → the derived
  value.
- Gate paragraph gains the derivation in two sentences: the discrimination threshold is the
  pilot stability floor's balanced-accuracy constant, held to the same value for the same
  endpoint under the same manipulation, and it is verified to exceed twice the confirmation-seed
  dispersion of balanced-condition CE. The calibration threshold is the 5%-relative-likelihood
  reading.
- §`app:construction` DLBCL paragraph (line 470): two exclusions, two rationales, resulting
  30-type target; update "The remaining 31 cancer types".
- Limitations §: the discrimination gate now sits at the smallest balanced-accuracy difference
  the design treats as real anywhere, so gate-passing means "damage above the benchmark's own
  materiality floor", not "large damage". Effect sizes carry that; the gate only licenses the
  ratio.

Needs explicit go-ahead before editing: `code/CLAUDE.md` names the protocol the authority.

## Verification

**Threshold derivation** — the one new computation, and it must run before any gate row is read:
per dataset and split, the standard deviation across the five confirmation seeds of
balanced-condition CE case-macro balanced accuracy (and tail NLL). Report the table, take the
max, apply the rule, freeze the constants. If any `σ_seed` exceeds 0.005 the pipeline is noisier
than the design assumed and that is itself a finding worth a sentence.

**Tests.** No behavioural change to gate logic, so the existing gate tests must still pass
untouched — that is the check. Add nothing; a test that asserts `DISCRIMINATION_THRESHOLD ==
0.01` restates the constant and catches nothing.

```powershell
uv run pytest experiments/2_benchmark_patch/code/tests/analysis
uv run python "$env:USERPROFILE\.codex\skills\clean-code\run.py" --scope experiments/2_benchmark_patch/code --vulture-scope experiments/2_benchmark_patch/code
```

**Freeze (TCGA-UT re-run).** `class_names` is the old list minus Cholangiocarcinoma; the 30
retained classes keep their prior split assignments; achieved ρ within [9,11] and [90,110];
`shared_T` at or above its previous 69,317.

**Preflight.** `is_descriptive_only: false` for TCGA-UT, tightest cell
Uterine_Carcinosarcoma at m=19, expected p2.5 kish ≈ 6.7.

**Gate outcome.** No prior number transfers — K, estimator, threshold and pipeline all changed.
The directional check is that deficits still increase from ρ=10 to ρ=100 on every dataset. If
TCGA-UT is under 0.01 even at ρ=100, the honest conclusion is that frozen Virchow2 features make
it robust to allocation at K=30. Report that; do not go looking for a severity level that
produces damage.

## Risks

- **The threshold change touches every dataset and every published number.** That is the point —
  it is why it must be frozen before gates are read — but it means PANDA's and BRACS's results
  text is re-reported at the new value too, and any prose citing "the 0.02 gate" needs finding.
- **CAMELYON16's saturation story may not survive.** Plan 1 plans to report it as an
  effect-size limit; at 0.01 it may open. Not a problem, but Plan 1's limitations text depends
  on the outcome and cannot be written before the re-analysis.
- **A lower gate admits smaller deficits, so recovery ratios get noisier.** `R = Δ/D` with
  D ≈ 0.012 has a wider relative interval than at D ≈ 0.03. The CI-excludes-zero condition and
  the evidence rule are still both required, and recovery CIs are reported — but expect wider
  intervals on the newly admitted cells and do not read a point recovery fraction off them.
- **Deriving a threshold after seeing which cells sit near it looks like tuning even when it is
  not.** The defence is entirely procedural: the anchor is a constant that predates every
  definitive fit, the noise floor is blind to every contrast, and both are written into the
  protocol before gate output is opened. Do it in that order or the argument is lost.
- **The TCGA-UT re-run is now the only GPU spend across the three plans** and buys breadth
  (K=30) rather than feasibility. Worth re-checking against the Plan 1 outcome before launching.
