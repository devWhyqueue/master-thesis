# Phase 01 — freeze the question and estimands

Input: experiments 25–28 and their frozen benchmark manifests. Output: `protocol_lock.json`, dataset configs, and a hashed schedule of exact training/evaluation identities. These are future artifacts, not present results.

## Design

Use frozen Virchow2 and UNI2-h, the existing multinomial logistic probe, and the three existing patient-disjoint splits. Preserve all eligible evaluation rows and class exclusions. TCGA-UT retains 30 classes, G=20 patients/class, and 640 balanced patches/class. BRACS retains seven classes, G=10, and 320 balanced patches/class. Both use eligibility depth 160, fixed total budget K·G·32, existing exponential allocator, and nested per-patient patch prefixes.

The patient count differs between datasets to preserve the established feasible regimes. Primary inference is within each dataset across encoders. Any comparison of the BRACS–TCGA-UT gap is descriptive and conditional on these unequal regimes; it cannot isolate tissue or class-count effects.

Use main draw IDs 10–19 per split and reserve draw 10000 for engineering smoke tests. First audit existing outputs for collisions; if these IDs were used, assign unused IDs and freeze them before fitting. Fresh draws do not create a fresh test cohort: existing patients and prior results informed this follow-up.

Generate the patient cohort, class permutation, and exact patch IDs once, without reference to either encoder. Each encoder consumes the same schedule. Do not regenerate patient or image order from its feature cache. Keep the same permutation at ratios 10 and 100. Store realized counts and ratios, including any allocator caps.

For each encoder and split–draw cell, fit seven arms:

| Arm | Training rows | Target loss prior |
|---|---|---|
| B | balanced allocation | uniform |
| P10, P100 | B rows | realized ratio-arm class shares |
| S10, S100 | ratio-arm rows | uniform |
| R10, R100 | ratio-arm rows | realized ratio-arm class shares |

Use per-row weights `(target class share)/(observed class share)`, summing to the number of rows. P changes risk weights without removing rows. S changes allocation at fixed class prior. S therefore measures patch-support redistribution, not isolated removal of minority rows: head classes gain rows at fixed total budget. Patient identities remain fixed across arms, but audit realized nonzero contributions at extreme ratios.

## Primary and explanatory contrasts

Let A(m,a) denote patient-macro balanced accuracy in percentage points for encoder m and arm a. Average draws within each split, then average the three splits equally.

For X in {P,S,R}, define D_X(m,rho)=A(m,B)-A(m,Xrho). Define I(m,rho)=D_R-D_P-D_S. Positive damage means worse than balanced training.

The primary contrast is delta_R=D_R(UNI2-h,100)-D_R(Virchow2,100), one endpoint per dataset. Negative values mean UNI2-h reduces imbalance damage. Also report A(UNI2-h,B)-A(Virchow2,B) and A(UNI2-h,R100)-A(Virchow2,R100), so floor effects cannot masquerade as improvement.

Secondary contrasts: delta_P, delta_S, delta_I at 100; all four at 10; balanced and imbalanced probability quality; per-class/tail recall. The decomposition identity delta_R=delta_P+delta_S+delta_I must hold numerically. Do not divide by total damage when it is small or crosses zero.

## Readout and regularization

Retain native encoder outputs and existing readout preprocessing; do not add PCA, whitening, or unit normalization to the primary design. Record feature norms and dimension. Each encoder–arm gets the same validation tuning procedure, not the same selected lambda. Primary inference concerns the complete frozen-encoder plus tuned-linear-probe pipeline, including its recommended image transform and output pooling.

Freeze a common lambda grid 10^-8 through 10^2, inclusive powers of ten, with the existing solver tolerances and strongest-regularization tie rule. This is 11 candidates per arm. Inspect convergence and grid boundaries using training/validation only. If the engineering pilot selects an endpoint, expand that boundary one decade for both encoders and datasets; repeat until the pilot optimum is interior or declare feasibility unresolved. Freeze the resulting common grid before the main run. Main boundary hits receive adjacent-decade diagnostic refits under a recorded amendment; report the frozen-grid primary analysis separately, never silently replace it.

Save every candidate's coefficients and validation score. Re-evaluate each encoder's arms at that encoder's B-selected lambda as a secondary sensitivity, using stored candidates. Never impose one encoder's numerical lambda on the other as proof of equal regularization. Fixed-B and tuned analyses answer different questions; disagreement limits mechanism claims.

Fit one positive scalar temperature per selected arm on validation patients using the existing calibration objective. Freeze objective, weighting, bounds, and ECE bins after tracing the existing implementation in phase 04. Temperature scaling must preserve argmax and BA. Report raw and scaled macro NLL and ECE. NLL is probability quality, not a pure calibration measure; ECE depends on bins and evaluation prevalence. Do not silently label existing ECE as class-balanced.

## Inference and decision rules

Use 10,000 paired bootstrap replicates, fixed seed 39000. Resample training draws within split with identical weights across all encoders and arms. Independently of draw resampling, resample evaluation patient clusters and carry each patient's multiplicity to every occurrence across arms, encoders, and split test sets. Preserve class support, accounting for BRACS patients with multiple labels; do not independently resample a patient per label. Reuse the repository bootstrap where it satisfies this contract, otherwise implement the smallest explicit shared patient-weight map. Never treat patches, 30 draws, or overlapping splits as independent patients.

Keep split membership fixed. Intervals describe uncertainty conditional on these three splits and available patients; they do not establish variability across independently collected datasets. Report per-split effects and draw dispersion. If a bootstrap replicate has no support for a class, redraw it and record the rejection count; never turn undefined recall into zero.

Report paired 95% intervals for all estimates. For the two primary dataset contrasts, additionally use two-sided 97.5% percentile intervals (Bonferroni family coverage at least 95%). Use those intervals for primary decisions: excluding zero resolves direction; wholly below -1 pp or above +1 pp resolves a practically meaningful change; wholly inside [-1,+1] pp supports practical similarity under this protocol. Otherwise label unresolved. These are approximate bootstrap decisions, not guarantees of power.

No significance claim for exploratory metrics or classwise correlations. No success-dependent stopping: after engineering gates pass, complete the frozen main design even if the expected effect looks absent. Wide intervals are a valid inconclusive result; extra draws do not repair a small independent test cohort.

Exit gate: every arm, seed, input hash, lambda rule, metric definition, and decision rule is recorded before main test results are opened.
