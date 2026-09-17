# Experiments

Controlled patch-classification studies on frozen Virchow2 features. Each study builds on the previous one. The headline question: why does patient shortage hurt accuracy, and what recovers it?

## Layout

- `NN_name/code/`: CLI (`__main__.py`) plus a study package. `configs/`: run configs. `tests/`: unit tests.
- `NN_name/report/`: `N_name.tex` (protocol), `results.tex` (results, discussion), compiled PDF.
- `environment.def`, `requirements-experiment.txt`: shared Apptainer image for Hydra runs (see `CLUSTER.md`).
- Datasets: BRACS and TCGA-UT (02–05); TCGA-UT only (06–16). Readout: logistic regression unless noted.
- Effects are patient-macro balanced accuracy in percentage points. Intervals come from a paired test-patient bootstrap. Practical threshold: 1 pp.

## Studies

| # | Study | Question | Result |
|---|---|---|---|
| 00 | datasets | Background report: CPath pipeline and datasets (CAMELYON16, TCGA-UT, BRACS, PANDA) | Descriptive, no experiment |
| 01 | methods | Taxonomy of imbalance methods by signal (prevalence, support, difficulty, diversity) and intervention level | Descriptive, no experiment |
| 02 | benchmark_patch | Controlled benchmark (MLP readout): does allocation damage accuracy, and which mitigation recovers it? | Nominal patch imbalance is recovered by prevalence/support weighting. Reduced patient coverage leaves a replicated discrimination deficit that no method recovers. Calibration needs separate treatment. |
| 03 | classifier_limitation | Is the shortage deficit a classifier limitation (MLP vs logistic vs k-NN)? | No. Logistic gains ≤0.5 pp; coverage benefit stays at 2.4–2.9 pp (BRACS) and 4.7 pp (TCGA-UT). |
| 04 | patient_influence | Does equal patient weight in the loss recover the deficit? | No. BRACS −1.4 pp (not significant), TCGA-UT ≈0. Coverage benefit unchanged (4.70 pp on TCGA-UT). |
| 05 | effective_support | Breadth vs depth grid at a fixed patch budget. Does N_eff = n/DE explain it? | 20×8 beats 5×32 by 3.7 pp (BRACS) and 9.5 pp (TCGA-UT). N_eff explains BRACS. TCGA-UT keeps a residual b = 2.98 pp per patient doubling. |
| 06 | multidirectional_redundancy | Does full-feature ICC absorb the TCGA-UT residual? | Only partly: b = 2.62 pp. Result: breadth beyond redundancy. |
| 07 | site_coverage | Do extra institutions explain the residual? (13 larger classes) | Site gain 1.06 pp [0.74, 1.37]. Inconclusive; gain mainly on seen sites. |
| 08 | patient_coverage | Nearest-neighbour cohorts vs random cohorts to isolate coverage | Manipulation check failed (leakage 0.84–0.88). No fits. |
| 09 | concentrated_coverage | Concentrate additions around one anchor | Manipulation check failed (0.67–0.73). Nested designs infeasible. |
| 10 | cohort_composition | Clustered vs random vs dispersed cohorts at 20 patients | Concentration damage 8.30 pp [7.66, 8.95]. Selection gain 1.04 pp [0.44, 1.66]. |
| 11 | coverage_redundancy | Do coverage distance and similarity-adjusted N_eff (fitted on random cohorts) explain the breadth benefit? | Residual b = 0.79 [−1.25, 2.82]. Predictions miss exp-10 contrasts by 1.5 pp (damage) and 1.1 pp (selection). Inconclusive. |
| 12 | shortage_selection | Coverage-maximizing vs random selection at 5×32 | Selection gain 5.52 pp [4.83, 6.23]. Recovers 58% of the 5→20 gap; 5 selected ≈ 10 random. |
| 13 | coverage_similarity | Matched cohorts to separate coverage from similarity | Precision gate failed (SE floor 0.38 pp). Matching infeasible. Redesign became exp-14. |
| 14 | shortage_decomposition | Designed coverage × similarity grid at 5/10 patients: decompose the gap | Gap 9.97 pp: coverage 2.23, similarity ≈0, patient count 8.08. Gap inflated by mixing 5- and 10-patient classes in one classifier (grid: 5.03). Similarity drives half of the exp-12 selection gain. |
| 15 | hull_coverage | Does hull coverage (span of patient deviations) explain the gap, at one patient count per classifier? | No. Gap 4.20 pp: coverage 1.82, hull −0.67, similarity ≈0, patient count 2.98. Hull absorption −0.27. The 10→20 gain is predicted at 0.86 pp against 4.38 observed. |
| 16 | centre_error | How much of the patient-count gap is class-centre error? Arms shift cohort centres to full-pool centres, add random centre error of size Σ/G, or also whiten along between-patient directions | Centre share 0.74 [0.70, 0.78] of the 5→20 gap (9.41 pp), stable per doubling. C5 beats R20 by 1.25 pp. Damage is class-specific (shared part 0.14 pp). Random error of size Σ/G reproduces 88% of the gap. Correct centres + whitening: combined share 0.96, residual 5→20 gap 0.40 pp. |

## Open thread

Exp-16 found the mechanism: class-centre error (first moment) carries about three quarters of the benefit of more patients, and between-patient directions (second moment) most of the rest. Pool centres and whitening use all training patients, so this is a mechanism, not a remedy. The second-moment part is not isolated: CW whitens on top of corrected centres, also transforms validation/test features, and raises accuracy at every G (2.5–4.5 pp); no arm whitens real cohorts, so the 0.96 combined share is not an additive split. Isolate it with a whitened-real-cohort arm (RW) and C5 plus population patient offsets. Next: (1) predict accuracy at untested G (e.g. 40) and on BRACS from the no-fit centre risk ρ_G = Σ_c Σ_{k≠c} (w_c − w_k)ᵀ Σ_c (w_c − w_k) / G; (2) mitigation without extra labelled patients, e.g. centre shrinkage and whitening estimated across classes or unlabelled slides. One patient count per classifier is the standard design.
