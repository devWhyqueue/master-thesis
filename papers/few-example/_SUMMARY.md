# Low-Data, Few-Shot, and Rare-Disease Learning in WSI Pathology

This summary assumes the basics of computational pathology, WSI preprocessing, MIL, foundation models, and weak supervision are already known. The focus is on concrete strategies for dealing with few examples in whole-slide-image (WSI) pathology, especially rare-disease or rare-subtype settings.

## Source Map

| Paper | Local file | Main use in this summary |
| :--- | :--- | :--- |
| Qu et al., path-knowledge enhanced multi-instance prompt learning | `papers/low-data/Qu et al.; Path-knowledge enhanced MI Prompt Learning.pdf` | Few-shot weakly supervised WSI classification with pathology-aware visual/text prompts |
| Hasan et al., foundation models with adapters | `papers/low-data/Hasan et al.; FMs with adapters.pdf` | Parameter-efficient adaptation of pathology FMs in very low-data regimes |
| Fu et al., FAST | `papers/low-data/Fu et al.; FAST.pdf` | Dual-level annotation and dual-branch VLM-based WSI classification |
| Zhang et al., DTFD-MIL | `papers/low-data/Zhang et al.; DTFD-MIL.pdf` | Pseudo-bag generation and double-tier MIL for limited slide counts |
| Quan et al., few-shot image classification | `papers/low-data/Quan et al.; Few-Shot Image Classification.pdf` | Metric/prototype-based few-shot learning with multi-scale features |
| Sabha et al., contrast-based learning for small datasets | `papers/low-data/Sabha et al.; Contrast based learning for small datasets.pdf` | Pairwise contrast-based training as a simple low-data augmentation/regularization idea |
| Banerjee et al., machine learning in rare disease | `papers/low-data/Banerjee et al.; ML in rare disease.pdf` | General rare-disease ML framing: label noise, small-n/high-p, regularization, transfer, knowledge graphs |
| Wojtara et al., AI in rare disease diagnosis | `papers/low-data/Wojtara et al.; AI in rare disease diagnosis.pdf` | Rare-disease AI context, clinical deployment framing, and diagnostic-support perspective |

## 1. Start From the Core Constraint

Low-data WSI learning is not only "few labeled slides." It is usually a combined constraint:

- Few patients or slides, especially for rare diseases or rare subtypes.
- Weak labels, often slide-level rather than patch-level or region-level.
- Many instances per slide, with positive morphology sometimes occupying a small fraction of tissue.
- Domain shift from staining, scanner, site, cohort composition, tissue handling, and annotation conventions.
- High label noise when disease definitions, subtype labels, or clinical standards evolve over time.

This makes naive supervised fine-tuning brittle. Banerjee et al. frame the broader rare-disease version of this problem as small-sample, high-dimensional learning with unstable generalization and high risk of learning dataset artifacts rather than disease biology. In WSI, DTFD-MIL and FAST make the same issue concrete: a slide may contain thousands of patches, but the number of independently labeled bags can still be small, and patch-level supervision is expensive.

## 2. Reuse Strong Representations Instead of Learning Them From Scratch

The most important low-data pattern is to avoid learning a full visual representation from the target dataset. Use a large pretrained histopathology model, vision-language model, or self-supervised encoder as the starting point, then adapt only the decision mechanism.

Hasan et al. evaluate this directly by freezing self-supervised histopathology foundation models and adding small adapters. The practical idea is simple: keep the expensive representation fixed and train only a lightweight MLP or convolutional adapter plus classifier. This reduces the number of task-specific parameters and lowers the risk of overfitting when only 1, 5, 10, or 20 examples per class are available.

For thesis framing, this category is best described as parameter-efficient transfer:

- Use pretrained encoders as general morphology feature extractors.
- Train a small task head, adapter, prompt, or prototype module.
- Prefer comparing adapter styles and frozen backbones over full fine-tuning when the target dataset is tiny.
- Treat backbone choice as a major experimental variable; Hasan et al. report meaningful differences between self-supervised foundation models under low-shot conditions.

**Where to read:** Hasan et al. for adapters and backbone sensitivity; Banerjee et al. for the broader transfer-learning logic in rare-disease ML.

## 3. Add Pathology Knowledge Through Prompts and Priors

Generic visual foundation models and VLMs can underperform in pathology because relevant concepts are specialized: atypia, necrosis, growth patterns, tumor-stroma interactions, tissue architecture, and disease-specific morphology. Prompting is therefore more useful when it encodes pathology knowledge rather than generic class names.

Qu et al. propose pathology-knowledge enhanced multi-instance prompt learning for few-shot weakly supervised WSI classification. The key idea is to inject pathology priors at both patch and slide levels using visual examples and textual descriptions, then align these prompts with WSI features. This is useful when only slide-level labels are available but you can still define or collect representative pathology concepts.

Fu et al.'s FAST follows a related VLM-oriented direction. It uses a prior branch with pathology-specific text descriptions, generated with GPT-4V from annotated instance images, and combines this with a cache branch for unannotated patches. The method is designed to use a small amount of expert annotation while still exploiting the rest of the WSI.

This category is strongest when:

- The class distinction can be described through recognizable histologic criteria.
- You can ask pathologists for a small number of representative patch annotations or concept descriptions.
- You want to adapt a VLM without fully retraining it.
- The target task is fine-grained and generic prompts are too weak.

The main caveat is that prompt quality becomes part of the model. Bad, vague, or incomplete pathology descriptions can become a new source of bias.

**Where to read:** Qu et al. for pathology-aware multi-instance prompt learning; Fu et al. for VLM priors and annotation-efficient WSI classification.

## 4. Use MIL Variants That Expand Supervision From Each Slide

Classic MIL gives one label per slide-sized bag. This is annotation-efficient but data-inefficient: if there are only a few WSIs, the model sees only a few bags even though each bag contains many patches. Low-data WSI methods therefore try to squeeze more learning signal out of each slide.

Zhang et al.'s DTFD-MIL does this by splitting each WSI into multiple pseudo-bags. Each pseudo-bag inherits the parent slide label, creating more training units from the same WSI. A first-tier MIL model learns from these pseudo-bags and distills features; a second-tier MIL model aggregates the distilled information for final slide-level prediction.

This is useful when:

- Slide count is the main bottleneck.
- Each WSI contains many patches.
- Patch-level labels are unavailable.
- Positive regions are sparse, making naive bag learning unstable.

The caveat is label inheritance. If pseudo-bags are assigned the parent slide label, some positive-slide pseudo-bags may contain no truly positive tissue. DTFD-MIL addresses this through double-tier feature distillation, but the false-positive pseudo-label issue should be discussed explicitly in a thesis.

FAST can also be read as a MIL supervision-expansion strategy: it combines a small amount of instance-level annotation with slide-level labels and a cache mechanism for unannotated patches. Instead of only making more pseudo-bags, it makes better use of a tiny set of stronger annotations.

**Where to read:** Zhang et al. for pseudo-bag and double-tier MIL; Fu et al. for dual-level annotation and patch/slide supervision.

## 5. Replace Dense Classifiers With Metric or Prototype-Based Decisions

Few-shot learning often avoids training a large classifier head. Instead, it learns an embedding space where classes are represented by prototypes, and new samples are classified by distance to those prototypes.

Quan et al. cover this direction through few-shot image classification with multi-scale prototype representations. The central idea is to combine local CNN-style features and global transformer-style features, refine them, and form prototypes that support 1-shot or 5-shot classification. Although this paper is broader than WSI pathology, the principle maps well to CPath: rare classes may be better represented by a few robust prototypes than by a fully trained parametric classifier.

In WSI pathology, prototype methods can be useful at multiple levels:

- Patch-level prototypes for morphological patterns.
- Slide-level prototypes from pooled or MIL-derived embeddings.
- Class prototypes based on support examples from rare subtypes.
- Hybrid prototypes using both local texture and global tissue architecture.

The caveat is that prototype quality depends on the embedding space. If the pretrained representation does not separate the disease-relevant morphology, the distance metric will look precise but encode the wrong similarity.

**Where to read:** Quan et al. for metric/prototype learning; Hasan et al. for combining few-shot algorithms with pathology foundation model features.

## 6. Create Relational Training Signal From Pairwise Contrasts

Sabha et al. introduce Contrast-Based Learning (CBL) for small datasets. The method spatially concatenates pairs of training images, expanding a dataset from `N` samples to roughly `N^2` paired inputs, while keeping the original architecture and loss function mostly unchanged. At test time, a query image is paired with a stratified control set and classified by majority voting.

For WSI pathology, this is not directly a slide-level method, but it suggests a useful patch-level or feature-level idea: when labeled examples are scarce, expose the model to explicit within-class and between-class contrasts rather than only isolated examples. In digital pathology, full-resolution spatial concatenation is likely impractical, so the more relevant adaptation would be patch-level pairing, embedding-level pairing, or pairing selected representative regions after tissue normalization.

The main caveat is computational cost and scaling. Training grows quadratically in the number of examples, and inference grows with the number of control samples per class. Sabha et al. argue this is acceptable in small-data regimes, but for WSIs it would need memory-aware design.

**Where to read:** Sabha et al. for CBL mechanics, control-set voting, Grad-CAM evidence, and the discussion of digital pathology as a future high-resolution use case.

## 7. Spend Annotation Budget Strategically

In low-data WSI work, the question is not only how many slides are labeled, but what type of labels are acquired. Fine-grained annotation is expensive, so methods increasingly combine cheap and expensive labels.

FAST is the clearest example in this set. It uses a dual-level annotation strategy: a small number of slide labels plus a tiny fraction of instance-level annotations. The annotated instances support pathology-specific prompting and prior construction, while the unannotated patches are still used through the model's cache branch.

This suggests a practical thesis design principle:

- Do not compare only "slide labels vs full patch labels."
- Include intermediate annotation regimes, such as a few patches per class, a few regions per slide, or concept-level labels.
- Measure annotation cost alongside performance.
- Report whether small instance-level supervision improves rare-class sensitivity, not only overall accuracy.

**Where to read:** Fu et al. for dual-level annotation; Qu et al. for visual/text prompts that can use expert-selected pathology examples.

## 8. Regularize, Rebalance, and Validate for Rare Classes

Rare-disease ML literature is useful because it describes failure modes that also appear in rare WSI tasks: label noise, site effects, batch effects, class imbalance, and misleading accuracy.

Banerjee et al. emphasize regularization, class rebalancing, ensemble/cascade approaches, and careful validation. In WSI, this translates to:

- Avoid using accuracy alone; prefer balanced accuracy, macro F1, sensitivity/specificity per class, and AUPRC for rare classes.
- Split by patient and, where relevant, by site or study to avoid leakage.
- Use external validation when possible, because internal cross-validation can overestimate performance in small cohorts.
- Consider class-balanced sampling, inverse-frequency weighting, or rare-class oversampling, but report the exact strategy.
- Use regularization and frozen representations to reduce memorization of site-specific artifacts.

Wojtara et al. add the clinical perspective: rare-disease AI systems should be evaluated as diagnostic-support tools under realistic constraints, not only as benchmark classifiers.

**Where to read:** Banerjee et al. for rare-disease ML pitfalls and mitigation; Wojtara et al. for clinical rare-disease AI framing.

## 9. Use External Knowledge When Examples Are Too Few

When visual data are insufficient, external knowledge can constrain the hypothesis space. This can include pathology descriptions, ontologies, known disease mechanisms, molecular data, clinical variables, or knowledge graphs.

In the WSI-specific papers, this appears as pathology-aware prompting and VLM priors. In the rare-disease ML papers, it appears as knowledge graphs, pathway-informed transfer, and multimodal integration. The common pattern is the same: add structured prior knowledge so the model does not need to infer everything from a tiny labeled image set.

For a master thesis, this can become a useful taxonomy:

- Visual priors: representative patches, morphology prototypes, concept examples.
- Textual priors: pathology descriptions, generated prompts, diagnostic criteria.
- Biological priors: pathways, genes, disease ontologies, tissue context.
- Clinical priors: symptoms, demographics, imaging context, disease prevalence.

**Where to read:** Qu et al. and Fu et al. for pathology/VLM priors; Banerjee et al. for knowledge graphs and pathway-informed transfer in rare diseases.

## Strategy Mapping Table

| Category | Core idea | When useful | Key caveat | Papers to read |
| :--- | :--- | :--- | :--- | :--- |
| Frozen FM + adapter | Freeze a pretrained pathology encoder and train only a small adapter/head | 1-20 examples per class; high overfitting risk | Depends strongly on pretrained feature quality | Hasan et al. |
| Pathology-aware prompt learning | Inject expert visual/text pathology concepts into VLM or MIL prompting | Few-shot WSI tasks with describable morphology | Prompt design can encode bias or miss relevant criteria | Qu et al.; Fu et al. |
| Dual-level annotation | Combine cheap slide labels with a tiny number of patch/instance labels | Annotation budget is small but expert patch labels are possible | Needs careful accounting of annotation cost and selection bias | Fu et al.; Qu et al. |
| Pseudo-bag MIL | Split each WSI into multiple pseudo-bags to increase bag-level training signal | Few slides, many patches per slide, weak slide labels | Parent-label inheritance creates noisy pseudo-bags | Zhang et al. |
| Double-tier MIL distillation | Learn from pseudo-bags first, then aggregate distilled features at slide level | Positive regions are sparse and bag labels are weak | More moving parts; must validate against simpler MIL baselines | Zhang et al. |
| Metric/prototype learning | Classify by distance to support-set prototypes instead of training a dense classifier | 1-shot/5-shot classification, rare classes, support-query setup | Embedding quality determines prototype quality | Quan et al.; Hasan et al. |
| Multi-scale features | Combine local texture and global structure features | Morphology depends on both cellular detail and tissue architecture | More feature fusion choices and potential overfitting | Quan et al. |
| Pairwise contrast-based learning | Train on paired examples so the model learns relational contrasts under small data | Patch-level or feature-level low-data settings where examples can be meaningfully paired | Naive spatial concatenation does not scale to full WSIs; inference adds voting overhead | Sabha et al. |
| Rare-class evaluation | Use rare-class-aware metrics and leakage-resistant validation | Imbalanced rare disease/subtype classification | Small validation sets make confidence intervals important | Banerjee et al.; Wojtara et al. |
| External knowledge | Use text, ontologies, pathology criteria, KGs, or clinical priors to constrain learning | Visual labels alone are too sparse or noisy | Knowledge sources may be incomplete or not aligned with WSI labels | Banerjee et al.; Qu et al.; Fu et al. |

## Thesis Takeaways

For low-data WSI pathology, the strongest recurring idea is not a single algorithm but a combination:

1. Start with a strong pretrained representation.
2. Adapt it with a small trainable component: adapter, prompt, prototype, or MIL head.
3. Use the WSI structure to extract more supervision from each slide.
4. Consider pairwise or prototype-style relational learning when isolated examples are too few.
5. Spend expert annotation on high-value patch/concept examples rather than trying to fully annotate slides.
6. Add pathology or biological prior knowledge when labels are too sparse.
7. Evaluate with rare-class-aware metrics and patient/site-aware validation.

The papers in this directory can therefore be grouped into three layers: representation adaptation (Hasan et al., Qu et al., Fu et al.), WSI-specific weak-supervision mechanics (Zhang et al., Fu et al., Qu et al.), and rare-disease methodological safeguards (Banerjee et al., Wojtara et al.). Quan et al. provides the general few-shot/prototype-learning background that can be adapted to patch-level or slide-level pathology embeddings, while Sabha et al. adds a lightweight pairwise-contrast idea that is most plausible for patch-level or feature-level WSI experiments.
