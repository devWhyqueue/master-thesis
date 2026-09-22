# Experiments

Controlled patch-classification studies on frozen Virchow2 features. Each study builds on previous. Two research questions: why patient shortage hurts accuracy and what recovers it (`02_patient_shortage/`); why class imbalance hurts accuracy and what causes it (`03_class_imbalance/`).

## Layout

| Group | Subgroup | Experiments |
|---|---|---|
| `00_survey/` | — | 00_datasets, 01_methods |
| `01_benchmark/` | — | 02_benchmark_patch |
| `02_patient_shortage/` | `00_damage/` | 05_effective_support |
| `02_patient_shortage/` | `01_cause/` | 03_classifier_limitation, 04_patient_influence, 06_multidirectional_redundancy – 18_bracs_mechanism |
| `02_patient_shortage/` | `02_mitigation/` | 19_centre_shrinkage – 24_oracle_weighting |
| `03_class_imbalance/` | `00_damage/` | 25_damage_tcga_ut, 26_damage_bracs |
| `03_class_imbalance/` | `01_cause/` | 27_cause_tcga_ut, 28_cause_bracs |

- `<group>/[<subgroup>/]NN_name/code/`: CLI (`__main__.py`) + study package. `configs/`: run configs. `tests/`: unit tests.
- `<group>/[<subgroup>/]NN_name/report/`: report `.tex` (protocol), `results.tex` (results, discussion), compiled PDF.
- `environment.def`, `requirements-experiment.txt`: shared Apptainer image for Hydra runs (see `CLUSTER.md`).
- Datasets: BRACS + TCGA-UT (02–05, 18, 26, 28); TCGA-UT only (06–17, 25, 27). Readout: logistic regression unless noted.
- Effects = patient-macro balanced accuracy, percentage points. Intervals from paired test-patient bootstrap. Practical threshold: 1 pp.

## Groups

Per-experiment rows live in each group's own `AGENTS.md` (`00_survey/`, `01_benchmark/`, `02_patient_shortage/`, `03_class_imbalance/`).

| Group | Subgroup | Experiments | Question | Headline result |
|---|---|---|---|---|
| 00_survey | — | 00–01 | Background: CPath pipeline, datasets, and imbalance-method taxonomy | Descriptive, no experiment |
| 01_benchmark | — | 02 | Controlled benchmark (MLP readout): does allocation damage accuracy, which mitigation recovers it? | Nominal patch imbalance recovered by prevalence/support weighting. Reduced patient coverage leaves a replicated discrimination deficit no method recovers. Calibration needs separate treatment |
| 02_patient_shortage | 00_damage | 05 | Breadth vs. depth at fixed patch budget: does effective support N_eff = n/DE explain the shortage deficit? | Breadth beats depth by 3.7 pp (BRACS), 9.5 pp (TCGA-UT). N_eff explains BRACS; TCGA-UT keeps a residual patient-count effect (b = 2.98 pp per doubling) |
| 02_patient_shortage | 01_cause | 03–04, 06–18 | What drives the TCGA-UT patient-count residual — classifier capacity, loss weighting, redundancy, coverage, or class-centre estimation? | Not classifier limitation (exp-03) or patient loss-weighting (exp-04). Coverage/selection effects real (exp-10, 12) but partly redundant with each other (exp-11) and with patient count (exp-14, 15). Class-centre estimation is the dominant mechanism: share 0.74–0.96 of the 5→20 gap on TCGA-UT (exp-16), 0.82–1.00 on BRACS (exp-18). Between-patient whitening (RWc) is a smaller, largely overlapping channel (share 0.25 of the gap, exp-17) |
| 02_patient_shortage | 02_mitigation | 19–24 | Can shrinkage, whitening, or uncertainty-aware training recover the centre-estimation gap without more patients? | None close the gap. Directional shrinkage (exp-20, At) and centre-uncertainty training (exp-21, Ut) show real but partial recovery on TCGA-UT (share 0.09–0.22 of the gap), both below the exp-17 whitening baseline RWc. Nothing works on BRACS. Oracle weighting (exp-24) shows the shortfall is direction, not weight, on the within-patient channel |
| 03_class_imbalance | 00_damage | 25–26 | How fast does BA/NLL/ECE degrade with class prevalence alone, at fixed total patches? | TCGA-UT: slow, accelerating damage (0.18 pp/doubling to ρ10, 0.70 beyond). BRACS: early, steady, ~6–7× larger damage (1.15–1.25 pp/doubling, no acceleration). Temperature scaling removes 81–96% of the calibration cost on both |
| 03_class_imbalance | 01_cause | 27–28 | exp-25/26's ratio arm r{rho} conflates the loss's class-prior shift with the tail's reduced patch support — which one drives the damage? | No report yet (TCGA-UT run in progress; BRACS not started) |
