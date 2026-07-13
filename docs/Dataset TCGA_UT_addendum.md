# TCGA-UT Imbalanced: Experimental Addendum

## Title and purpose

This addendum complements [`Dataset TCGA_UT.pdf`](/mnt/d/Git/master-thesis/code/TCGA-UT/docs/Dataset%20TCGA_UT.pdf). The PDF remains the primary description of the dataset motivation, the balanced-to-imbalanced sampling construction, and the overall experimental framing. The purpose of this note is narrower: it captures experimental components and comparison workflows that are present in `code/tcga-ut/main` but are not described in the same level of detail in the PDF.

The emphasis here is intentionally on experiment design and supported analyses rather than on code internals.

## What the PDF already covers

The PDF already documents the core conceptual setup well:

- why controllable class imbalance is relevant in histopathology,
- how TCGA-UT is converted into a balanced base set,
- how balanced validation data are sampled before imbalanced training data are generated,
- how the imbalance parameter induces a power-law class distribution,
- why fixed dataset size and consistent validation sampling are useful for comparisons across imbalance settings,
- and why simple classifiers on frozen foundation-model features are useful for studying class difficulty and downstream robustness.

It also already establishes the main experimental logic behind varying class orderings, using per-class recall as a primary lens, and comparing results across imbalance regimes.

## Experimental components present in the code but not sufficiently covered in the PDF

The current codebase expands the experimental space beyond the core narrative in the PDF.

First, the training code supports several mitigation strategies for imbalance on top of the same feature pipeline. In addition to standard cross-entropy training, the code supports focal loss, class-frequency-based reweighting, and batch balancing. These are not just implementation alternatives; they define different hypotheses about how performance on minority classes may be improved. Focal loss emphasizes hard examples, inverse-frequency weighting increases the influence of rare classes in the objective, and batch balancing changes the sampling process so that classes are seen more uniformly during optimization.

Second, the code includes an additional training variant based on odd-k-out (OKO) set learning. Conceptually, this moves away from training only on isolated examples and instead trains the model to identify the repeated class within small sets. This is a distinct experimental direction relative to the more standard single-example classification setup. It should be read as an additional methodological probe into whether a different supervisory signal changes robustness under imbalance rather than as a mere implementation detail.

Third, the codebase operationalizes the comparison between several baseline families: a learned MLP head, KNN, and nearest-centroid classification. This matters experimentally because these models occupy different points on the spectrum between non-parametric feature probing and trained downstream adaptation. KNN and NCC can be viewed as lightweight probes of feature separability, while the MLP represents a trainable downstream classifier whose behavior can be modified by different imbalance-mitigation strategies.

## Training strategies and baselines

### MLP-based training

The main trainable baseline in the current code is an MLP on top of frozen patch-level features. This is the setting where the mitigation strategies are varied most directly. The experimental comparisons supported by the code include:

- vanilla cross-entropy training,
- cross-entropy with inverse-frequency weighting,
- focal loss in weighted and unweighted variants,
- batch-balanced training,
- and OKO-based training as an alternative objective.

This makes the MLP branch the main vehicle for studying whether better minority-class recall comes from changing the loss, changing the sampling process, or changing the supervision structure itself.

### Feature-space baselines

The code also supports KNN and NCC baselines. These baselines are important for two reasons. They provide a low-capacity reference point for how separable the frozen features already are, and they help distinguish failures caused by the feature representation from failures caused by the downstream learner. In practice, these baselines fit naturally with the PDF's discussion of class difficulty: if a class is already hard for KNN or NCC in the balanced regime, then poor recall under imbalance is easier to interpret as a feature-space limitation rather than purely a training issue.

### Class-order-dependent imbalance regimes

The PDF already motivates varying class order. The codebase makes this experimentally actionable by allowing class orders to be provided externally and reused across runs. This means the imbalance parameter does not only control severity; it can also be coupled to different notions of which classes should become rare or frequent.

This supports at least three distinct experiment types:

- realistic prevalence orderings based on the original dataset,
- orderings where easier classes become more dominant,
- and orderings where harder classes become more dominant.

That design makes it possible to ask not only whether imbalance hurts performance, but also whether it hurts differently when rarity aligns with intrinsic class difficulty.

## Evaluation views and comparison workflow

The codebase supports a more explicit comparison workflow than the PDF alone suggests.

At the single-run level, results are saved in a form that supports standard validation summaries such as accuracy, balanced accuracy, classwise recall, classwise precision, and confusion matrices. This aligns with the PDF's emphasis on per-class effects rather than only aggregate performance.

Across repeated runs, the visualization code supports several higher-level comparative views:

- averaged confusion matrices for a fixed parameter value,
- difference confusion matrices between two conditions,
- scatter plots comparing classwise recalls between two parameter settings,
- and point plots that compare multiple methods on the same class ordering.

Taken together, these plots support a workflow centered on classwise behavior. Rather than asking only which method achieves the highest mean score, the code encourages questions such as:

- which classes improve or deteriorate under stronger imbalance,
- whether a mitigation method helps mostly small classes or shifts trade-offs across all classes,
- and whether gains relative to a baseline are concentrated in specific parts of the class-size spectrum.

The method-comparison point plot is particularly useful in this regard because it places multiple training strategies on a common axis and preserves the ordering from few-shot to many-shot classes. That makes it well suited for comparing how different mitigation methods distribute benefits across the imbalance spectrum.

## Practical notes on experiment organization

The scripts in `code/tcga-ut/main/scripts` show that the intended workflow is based on parameter sweeps and repeated seeds rather than isolated one-off runs. Experiments are organized by imbalance parameter and seed, and visualization scripts aggregate results across the seeds found within a result directory. This implies that the intended unit of analysis is not a single trained model but a small collection of repeated runs per condition.

The same scripts also show the comparison sets that were most directly anticipated in current usage. In particular, the visualization workflow explicitly compares:

- weighted cross-entropy,
- batch balancing,
- weighted focal loss,
- unweighted focal loss,
- vanilla training,
- and the balanced reference setting.

This provides a concrete picture of how the codebase is meant to be used experimentally: define a family of training conditions, sweep over imbalance parameters and seeds, then compare methods using aggregated classwise plots rather than relying only on tabulated averages.

The result folders also function as lightweight experimental records. They contain the trained model or fitted baseline together with the validation summary and the arguments used for the run. At a high level, this means the directory structure is part of the reproducibility story: it preserves both the outcome and the configuration of each condition.

## Known drift or caveats

The current repository contains some drift between the PDF, the README, and the executable code.

The PDF remains the strongest source for the conceptual description of the dataset and the main imbalance-construction procedure. The README is useful for running the pipeline but is explicitly marked as partly outdated. The training code has evolved further and now includes additional experimental options, most notably OKO-style training and a broader set of imbalance-mitigation variants, that are not reflected equally clearly in the prose documentation.

There is also some drift in the visualization interface. The README describes the visualization workflow in one way, while the current script interface organizes it through a set of plot types. This does not change the experimental intent, but it is a reminder that the code should be treated as the source of truth for what comparisons are currently supported.

Finally, the repository itself flags some files as outdated or unused. Those files are better read as remnants of earlier stages of the project rather than as part of the current experimental core. The experimental core is the balanced sampling step, the imbalanced sampling step, the training and evaluation pipeline for MLP/KNN/NCC, and the comparative visualization workflow built around per-class behavior across parameters and methods.
