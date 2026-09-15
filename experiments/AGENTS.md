# Experiments

Controlled patch-classification studies on frozen Virchow2 features. Each study builds on the previous one. The headline question: why does patient shortage hurt accuracy, and what recovers it?

## Layout

- `NN_name/code/`: CLI (`__main__.py`) plus a study package. `configs/`: run configs. `tests/`: unit tests.
- `NN_name/report/`: `N_name.tex` (protocol), `results.tex` (results, discussion), compiled PDF.
- `environment.def`, `requirements-experiment.txt`: shared Apptainer image for Hydra runs (see `CLUSTER.md`).
- Datasets: BRACS and TCGA-UT (03–05); TCGA-UT only (06–14). Readout: logistic regression unless noted.
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
| 14 | shortage_decomposition | Designed coverage × similarity grid at 5/10 patients: decompose the gap | Gap 9.97 pp: coverage 2.23, similarity ≈0, patient count 8.08. Similarity drives half of the exp-12 selection gain. |

## Open thread

Patient-mean coverage and similarity leave most of the patient-count advantage unexplained. Next candidate signal: distances between patch distributions. Use a design where every class shares one patient count.
