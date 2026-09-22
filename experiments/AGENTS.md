# Experiments

Controlled patch-classification studies on frozen Virchow2 features. Each study builds on previous. Two research questions: why patient shortage hurts accuracy and what recovers it (`02_patient_shortage/`); why class imbalance hurts accuracy and what causes it (`03_class_imbalance/`).

## Layout

| Group | Experiments |
|---|---|
| `00_survey/` | 00_datasets, 01_methods |
| `01_benchmark/` | 02_benchmark_patch |
| `02_patient_shortage/` | `00_damage/`: 05_effective_support. `01_cause/`: 03_classifier_limitation, 04_patient_influence, 06_multidirectional_redundancy – 18_bracs_mechanism. `02_mitigation/`: 19_centre_shrinkage – 24_oracle_weighting |
| `03_class_imbalance/` | `00_damage/`: 25_damage_tcga_ut, 26_damage_bracs. `01_cause/`: `tcga_ut/`: 27_prior_vs_support. `bracs/`: 28_prior_vs_support, 29_tail_assignment |

- `<group>/[<subgroup>/]NN_name/code/`: CLI (`__main__.py`) + study package. `configs/`: run configs. `tests/`: unit tests.
- `<group>/[<subgroup>/]NN_name/report/`: report `.tex` (protocol), `results.tex` (results, discussion), compiled PDF.
- `environment.def`, `requirements-experiment.txt`: shared Apptainer image for Hydra runs (see `CLUSTER.md`).
- Datasets: BRACS + TCGA-UT (02–05, 18, 26, 28); TCGA-UT only (06–17, 25, 27). Readout: logistic regression unless noted.
- Effects = patient-macro balanced accuracy, percentage points. Intervals from paired test-patient bootstrap. Practical threshold: 1 pp.

## Groups

Per-experiment rows live in each group's own `AGENTS.md` (`00_survey/`, `01_benchmark/`, `02_patient_shortage/`, `03_class_imbalance/`).

| Group | Experiments | Question | Headline result |
|---|---|---|---|
| 00_survey | 00–01 | Background: CPath pipeline, datasets, and imbalance-method taxonomy | Descriptive, no experiment |
| 01_benchmark | 02 | Controlled benchmark (MLP readout): does allocation damage accuracy, which mitigation recovers it? | Nominal patch imbalance recovered by prevalence/support weighting. Reduced patient coverage leaves a replicated discrimination deficit no method recovers. Calibration needs separate treatment |
| 02_patient_shortage | 03–24 | Why does patient shortage hurt accuracy, and can it be recovered without more patients? | Breadth beats depth by 3.7–9.5 pp at fixed patch budget (exp-05). The residual traces through coverage/selection effects (real but partly redundant) to class-centre estimation as the dominant mechanism — share 0.74–1.00 of the 5→20 gap on TCGA-UT/BRACS (exp-03–18); between-patient whitening (RWc) is a smaller, overlapping channel. Of six tested mitigations, only directional shrinkage and centre-uncertainty training show partial recovery on TCGA-UT (share 0.09–0.22); nothing works on BRACS (exp-19–24) |
| 03_class_imbalance | 25–29 | Why does class imbalance hurt accuracy, and what causes it? | TCGA-UT: slow, accelerating BA/NLL/ECE damage with prevalence ratio; BRACS: early, steady, ~6–7× larger damage; temperature scaling removes most of the calibration cost on both (exp-25–26). TCGA-UT cause (exp-27): skewed prior and thin tail support interact — at ρ 100 prior alone 1.20, support alone 0.53, interaction 1.18 of 2.91 pp; prior carries the damage up to ρ 20, support + interaction the acceleration beyond ρ 10; training-time rebalancing beats post-hoc logit adjustment. BRACS cause (exp-28): prior (4.47) and support (3.29) both damage, additive in point estimate (I −0.11, unresolved); prior larger but thin support alone exceeds TCGA-UT's full damage. BRACS tail class (exp-29): damage at ρ 100 ranges 5.7–10 pp by which class is rare, ordered by balanced recall (headroom), not by the atypical classes; six of seven classes collapse to ≤ 4% recall in the tail |
