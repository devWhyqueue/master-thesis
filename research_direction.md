# Research direction: independent-support shortage

*Master thesis · Research direction · 11 September 2026*

A zoomed-out overview of what the experiments so far have shown, the open question, and the next steps.

## Summary

Training a histopathology classifier on **fewer patients** hurts, even when the number of patches stays the same, and no tested, existing mitigation method repairs it.

On **BRACS**, the damage is almost fully explained by redundancy: patches from the same patient repeat each other. On **TCGA-UT**, redundancy explains only a small part of it. Extra patients still add something significant that we cannot measure yet.

Next: find and measure that missing ingredient, then build a mitigation method for both cases.

### Terms

- **Nominal support:** How many training patches a class has.
- **Independent support:** How many different patients those patches come from.
- **Effective support $N_{\mathrm{eff}}$:** Patch count discounted for within-patient redundancy: how many truly independent observations the patches are worth.

## Completed work

### 1. Define data-shortage signals

Named the reasons a class can be under-served: prevalence, nominal support, independent (patient) support, class difficulty, and within-class diversity. Each became a measurable score.

Sources: [Methods report](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/01_methods/report/1_methods.pdf) · [Benchmark protocol](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/02_benchmark_patch/report/2_benchmark_patch_protocol.pdf)

### 2. Manipulate shortages and measure discrimination and calibration damage

Removing patients while keeping patch counts identical costs 2.6–3.0 points of balanced accuracy on BRACS and 4.8 on TCGA-UT. On TCGA-UT that is almost three times what a severe 100:1 class imbalance costs (at most 1.8). Probability quality suffers too, and temperature scaling does not undo it.

Source: [Benchmark results, exp. 2](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/02_benchmark_patch/report/2_benchmark_patch_results.pdf)

### 3. Test whether existing methods repair the damage

Patch shortage is repairable: class weighting recovers 43–89% of its damage on BRACS. Patient shortage is not: all weighting methods recover between −6.5% and +2.5%, and the best of the wider method roster closes at most about a sixth of the TCGA-UT gap.

Source: [Benchmark results, exp. 2](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/02_benchmark_patch/report/2_benchmark_patch_results.pdf)

## Patient breadth at a fixed patch budget

Exp. 5 makes the problem concrete. Every class gets exactly 160 labelled patches. The only difference is how many patients they are spread across.

| Dataset | Deep (5 patients × 32 patches) | Broad (20 patients × 8 patches) | Gain |
| --- | ---: | ---: | ---: |
| BRACS | 37.1% | 40.8% | +3.7 |
| TCGA-UT | 62.3% | 71.8% | +9.5 |

Scores are patient-level balanced accuracy per class allocation; both gains have 95% intervals well above zero. No extra labels were used.

So the question is no longer *whether* patients matter. It is: **why do few patients hurt, even with plenty of patches?**

## Where does the damage come from?

Long-tailed learning in histopathology splits into two parts: learning a **representation** (the feature extractor) and learning a **classifier** on top of it. The first move was to check whether either was simply a bad choice.

### Did we pick the wrong classifier? — *Ruled out*

Most likely not. Swapping the MLP for logistic regression gains only +0.19 points on BRACS and +0.48 on TCGA-UT, below the one-point threshold; nearest neighbours are worse. More patients still help every classifier by 2.4 and 4.7 points.

Source: [Classifier limitation, exp. 3](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/03_classifier_limitation/report/3_classifier_limitation.pdf)

### Did we pick the wrong representation? — *Not tested*

Probably not, but unconfirmed. Features come from Virchow2, a strong pathology foundation model. The check would be to swap in other foundation models; the expected result is that none does significantly better under patient shortage. Deprioritised because it is not a likely explanation.

### Do a few patients dominate training? — *Ruled out*

Giving every patient equal weight in the loss changes accuracy by −1.36 points on BRACS and −0.004 on TCGA-UT. The patient gap stays intact.

Source: [Patient influence, exp. 4](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/04_patient_influence/report/4_patient_influence.pdf)

### Are patches from one patient redundant? — *Explains BRACS*

On BRACS, yes, and that is most of the story. Once patch counts are converted to effective support $N_{\mathrm{eff}}$, the extra effect of patient count shrinks from 2.86 to 0.13 points per log unit.

Source: [Effective support, exp. 5](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/05_effective_support/report/5_effective_support.pdf)

### What do additional patients add beyond reduced redundancy? — *Open for TCGA-UT*

TCGA-UT is even more redundant than BRACS, yet after accounting for it, each doubling of patients still adds about 3 points. Something between patients is missing from $N_{\mathrm{eff}}$.

Source: [Effective support, exp. 5](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/05_effective_support/report/5_effective_support.pdf)

## BRACS and TCGA-UT compared

Both datasets show strong within-patient redundancy. They differ in whether redundancy is the whole explanation.

| | BRACS | TCGA-UT |
| --- | --- | --- |
| Task | 7 breast-lesion subtypes (single institution) | 30 cancer types (many submitting institutions) |
| Within-patient similarity (intraclass correlation) | 0.13 | 0.46 |
| 32 patches from one patient are worth… | ≈ 6 independent patches | ≈ 2 independent patches |
| More patches per patient (8 → 32, at 20 patients) | +2.0 pts (still useful) | +0.3 pts (saturated) |
| Patient effect left after $N_{\mathrm{eff}}$ | 0.13 per log unit (95% CI −1.60 to 1.85) | 4.30 per log unit (95% CI 3.56 to 4.99; ≈ 3 pts per doubling of patients) |
| **Verdict** | **Redundancy explains it** | **Largely unexplained** |

For BRACS this makes $N_{\mathrm{eff}}$ a promising quantitative signal, both for detecting this kind of shortage and possibly for mitigating it. For TCGA-UT, extra patients seem to supply inter-patient variation or coverage that $N_{\mathrm{eff}}$ does not capture. Identifying that source is the main open question.

## Follow-up research direction

### 1. Identify the cause of the TCGA-UT deficit and make it measurable

Identify the missing source of information behind the TCGA-UT patient effect and operationalise it as a signal, the way $N_{\mathrm{eff}}$ works for BRACS. Two leads already come out of exp. 5:

- **Redundancy is underestimated.** TCGA-UT patches are rotated, rescaled, overlapping crops of a few uniform tumour regions. The current correlation looks along only one feature direction and may miss this. If a multi-direction measure absorbs the leftover effect, it was redundancy after all.
- **Real between-patient variation.** TCGA-UT slides come from many institutions with their own staining and scanning, which BRACS lacks. Extra patients may add exactly that spread.

### 2. Develop a mitigation method

Target the two shortages separately: effective-support shortage (BRACS) and *[the shortage found in step 1]* (TCGA-UT).

## Reports

0. [Datasets and background](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/00_datasets/report/0_datasets.pdf)
1. [Imbalance methods and signals](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/01_methods/report/1_methods.pdf)
2. [Patch benchmark results](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/02_benchmark_patch/report/2_benchmark_patch_results.pdf)
3. [Classifier limitation](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/03_classifier_limitation/report/3_classifier_limitation.pdf)
4. [Patient influence](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/04_patient_influence/report/4_patient_influence.pdf)
5. [Effective support](https://github.com/devWhyqueue/master-thesis/blob/main/experiments/05_effective_support/report/5_effective_support.pdf)

*Scope: patch classification on frozen Virchow2 features, BRACS and TCGA-UT, three patient-disjoint splits each. "Points" means percentage points of patient-level balanced accuracy.*
