# Experiment 36: Separate prior sensitivity from thin-support estimation error

## Aim and evidence

Explain BRACS–TCGA-UT imbalance damage under frozen Virchow2 features and logistic regression. Test whether **class separation and class-centre estimation error jointly explain the gap**, while distinguishing regularization effects.

Existing evidence supports investigation, not guaranteed success:

- Published ρ=100 damage: **7.65 versus 2.91 pp**.
- Live experiment-34 validation artifacts, matched at 10 patients/class: **6.14 versus 2.90 pp**.
- Expanding BRACS separation reduced prior-only damage **3.72 pp**, but increased support-only damage **6.50 pp**. Prior response reproduced directionally across all three splits, including reciprocal TCGA-UT contraction.
- Patient-shortage experiments support centre error as a mechanism. Its role in **patch-support imbalance remains untested**.

**Read-only precheck completed. Joint intervention has not passed an empirical precheck.** No files changed or jobs submitted.

## Controlled experiment

Create `experiments/03_class_imbalance/01_cause/cross/36_joint_mechanism`.

Keep frozen splits, features, class definitions, allocator and patient-macro balanced accuracy. Primary severity: **ρ=100**. Both datasets use **10 patients/class, 320 balanced patches/class**; TCGA-UT uses experiment 34’s nested ten-patient cohorts.

Cross two interventions:

| Setting | Separation | Training class-centre correction |
|---|---|---|
| Native | Unchanged | None |
| Separation only | Experiment 34’s frozen expansion/contraction | None |
| Centre correction only | Unchanged | Dense same-cohort target |
| Joint | Expansion/contraction | Dense same-cohort target |

For each setting, fit **B/P/S/R**: balanced, prior-only, support-only and combined imbalance.

Centre correction uses all **160 eligible patches per selected patient** to estimate a patient-balanced reference centre. Translate each training class to that target, preserving its within-class residuals. Apply correction to balanced and imbalanced arms alike. Evaluation features receive only the prescribed separation transformation.

This oracle supplies extra training information without adding classifier-training rows. It tests finite-depth first-moment error; it does not remove patient-selection error or restore missing covariance information.

Add two controls:

- **Fixed regularization:** evaluate native and joint B/P/S/R using the native balanced arm’s selected λ throughout.
- **Wrong-direction correction:** within the joint setting, replace each correction vector with its negative for S/R. Same correction magnitude, opposite direction.

Calculate prior damage, support damage, total damage and their interaction separately. Report absolute accuracy alongside damage so deterioration of the balanced reference cannot masquerade as rescue.

Bridge original and matched-budget gaps using paired stored TCGA-UT G=20 controls. Do not attribute that budget contribution to representation geometry.

## Execution and genuine precheck

**Pilot:** three splits × draws 0–1 × both datasets.

- 192 core arm evaluations.
- 24 wrong-direction controls.
- 84 additional fixed-λ evaluations, reusing fitted grid candidates.
- **300 arm evaluations total; 216 tuned training conditions.** Candidate optimization runs are counted separately.
- Freeze λ grid at powers of ten from `1e-8` through `1e2`. Retain coefficients, predictions and convergence information for every candidate.

Use validation outcomes only for new pilot decisions. Existing experiment-34 results provide historical motivation, not independent confirmation.

Full execution requires all of these prespecified gates:

1. **Integrity:** native replay within 0.01 pp; exact cohort/allocation pairing; transformations preserve intended residuals; no evaluation data enter centre estimates.
2. **Prior mechanism:** separation reduces BRACS prior damage and increases TCGA-UT prior damage by at least 1 pp pooled, with predicted direction in at least two splits each.
3. **Support mechanism:** centre correction reduces expanded-BRACS support damage by at least 1 pp and outperforms wrong-direction correction by at least 1 pp.
4. **Joint rescue:** BRACS total damage falls at least 2 pp; imbalanced-arm accuracy rises at least 2 pp; balanced accuracy falls no more than 1 pp. Rescue direction must hold in both pilot draws and at least two splits.
5. **Robustness:** joint BRACS rescue remains positive under fixed λ. All selected fits converge; boundary selections require stable endpoint behaviour rather than an automatic numerical pass.
6. **Precision:** patient-and-draw resampling projects a primary rescue interval half-width ≤1 pp for the main design. Preserve repeated patient identities across splits; never manufacture independent patients by adding draws.

**If any gate fails:** stop expansion. Record which proposed mechanism failed; do not change intervention strength or thresholds to obtain a pass.

**If gates pass:** run locked draws 2–9, excluding pilot draws from primary estimates. Same settings and controls: **1,200 arm evaluations**, including **864 tuned conditions**. No additional severity sweep.

Execute through existing Hydra/Apptainer workflow, with one canary first, resumable signed shards, and account-wide queued/running tasks ≤100. Submit main work only after validated pilot artifacts exist.

## Code, inference and checks

Reuse allocator, sampling, logistic solver, centre translation and separation code. Add a small experiment-local CLI following `precheck`, `fit`, `analyze`, `report`, and staged `submit`; no changes to existing experiment behaviour.

Persist cohort/row hashes, class order, correction targets, intervention strengths, candidate λ results, patient identities and predictions. Main submission must verify the passing pilot’s configuration and artifact hashes.

Primary estimates:

- BRACS joint rescue.
- Reduction in the matched-budget cross-dataset damage gap when BRACS receives the joint intervention and TCGA-UT remains native.

Use paired patient-and-draw bootstrap distributions, independent between datasets, with shared patient weights across repeated appearances. Report simultaneous 95% intervals for the two primary estimates. Treat splits as fixed experimental partitions; intervals remain conditional on these training pools.

Tests cover transformation identities, correction direction, unchanged allocations and priors, decomposition arithmetic, patient pairing, fixed-λ reuse, reproducibility, failed-gate submission blocking and hash-aware resumption. Run affected tests and scoped clean-code checks.

## Report and claim limits

Produce one compiled, visually inspected LaTeX report:

- Original versus support-matched gap.
- Four-setting channel decomposition with uncertainty.
- Absolute balanced/imbalanced accuracy and wrong-direction controls.
- Fixed-λ comparison and numerical diagnostics.
- Explained reduction and remaining gap, in percentage points.

Call the gap **operationally closed** only if its residual interval lies entirely within ±1 pp and absolute BRACS imbalanced accuracy improves. Otherwise report partial explanation or rejected mechanism.

Interpret positive results as a controlled, sufficient mechanism within these datasets and this classifier. Class count and dataset identity remain uncontrolled; unique biological attribution would exceed this design.

Existing test reuse makes this **internal replication**, not untouched confirmation. New draws do not erase adaptive hypothesis selection; document that limitation explicitly ([Dwork et al.](https://research.ibm.com/publications/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis)).

Thirty-five studies provide substantial constraints. They do not guarantee that the remaining explanation is identified. This design earns full execution by demonstrating actual joint rescue first.
