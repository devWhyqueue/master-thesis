# Class Imbalance Methods in Computational Pathology and Medical Imaging

This folder groups the papers according to the local taxonomy in `_TAXONOMY.md`. The emphasis is on methods that actively mitigate class imbalance. Mosquera et al. is kept separately because it is about evaluation practice, calibration, and metric choice rather than a method for rebalancing training.

## Source Map

| Taxonomy group | Papers | Main role |
| :--- | :--- | :--- |
| Reviews and taxonomy | Saini and Susan| Broad class-imbalance taxonomy and generative-model background |
| Data-level sampling and augmentation | Chen and Lu; Majeed et al.; Nallamothu et al. | Resampling, SMOTE-style methods, undersampling, feature-level WSI augmentation |
| Synthetic data generation | Ruiz-Casado et al.; Ryspayeva et al.; Yuan et al. | GAN-based minority-class image generation |
| Algorithm-level losses | Mahbub et al.; Scholz et al. | Loss functions that reshape optimization toward minority classes |
| Hybrid approaches | Guerrero et al. | Copy-paste augmentation combined with loss weighting |
| Representation and architecture | Juyal et al.; Nouyed et al. | MIL, contrastive representation learning, partitioning, and cluster-based patch sampling |
| Evaluation practices | Mosquera et al. | Metrics and calibration under class imbalance |

## Folder Layout

| Folder | Meaning |
| :--- | :--- |
| `reviews_and_taxonomy` | Survey material used to define the taxonomy and contextualize generative pathology models |
| `data_level_sampling_augmentation` | Methods that alter the effective training distribution by resampling, oversampling, undersampling, or feature/image augmentation |
| `synthetic_data_generation` | Generative augmentation, mainly GAN-based minority-class synthesis |
| `algorithm_level_losses` | Objective-function changes such as metric-derived losses, prototype/affinity losses, and minority-aware penalties |
| `hybrid_approaches` | Methods that combine data-level and algorithm-level balancing |
| `representation_and_architecture` | Methods that improve feature spaces, MIL training, or patch selection so rare classes become more separable |
| `evaluation_practices` | Evaluation-only material: how imbalance affects discrimination, precision-recall behavior, and calibration |

## 1. Taxonomy Baseline

Saini and Susan provide the broad computer-vision taxonomy used here: data-level manipulation, synthetic generation, algorithm-level learning, hybrid/ensemble approaches, and representation or architecture-level strategies. For computational pathology, this taxonomy needs one adjustment: WSI methods often operate on bags, patches, or extracted features rather than ordinary fixed-size images. A method can therefore rebalance slides, patches, feature vectors, or classifier decisions.

## 2. Data-Level Sampling and Augmentation

Data-level methods change what the model sees during training. They are the most direct response to imbalance: increase minority exposure, reduce majority dominance, or create balanced batches.

Chen and Lu propose RankMix for weakly supervised WSI classification. Instead of mixing fixed-size images, RankMix mixes ranked WSI features. This matters because WSIs differ in size and contain variable numbers of patches, so ordinary mixup is not directly applicable. The method uses pseudo-labeling and ranking to identify important WSI regions before mixing, then trains in two stages for stability. In taxonomy terms, this is data-level augmentation adapted to MIL/WSI feature bags.

Majeed et al. compare undersampling and oversampling strategies for oral cavity histopathological WSIs: Near Miss, Edited Nearest Neighbors, SMOTE, Deep SMOTE, and ADASYN, combined with transfer learning. Their main takeaway is that oversampling, especially SMOTE in their experiments, can improve performance under controlled imbalance ratios. This paper is useful as a conventional resampling baseline for pathology experiments.

Nallamothu et al. evaluate several sampling methods, including RUS, RENN, Near Miss, SVM-SMOTE, Borderline-SMOTE, and simple SMOTE, across neural models. It is less WSI-specific than the CPath papers, but useful for comparing the behavior of classical sampling methods under medical-data imbalance.

Practical takeaway: data-level methods are easy to explain and can be strong baselines, but they can also duplicate artifacts, remove useful majority variation, or create synthetic samples that do not correspond to valid pathology. Undersampling is especially risky in medical imaging because majority-class examples may still contain rare staining, scanner, tissue-preparation, or morphology variation needed for generalization. Naive oversampling can also inflate training time and overfit duplicated samples without increasing the effective number of genuinely distinct examples.

## 3. Synthetic Data Generation

Synthetic data generation is a specialized data-level strategy: create new minority-class samples rather than simply transforming or resampling existing ones.

Ryspayeva et al. use WGAN-GP to balance a breast histopathology dataset before transfer learning with ImageNet-pretrained models. The paper is a straightforward example of GAN-generated images used as a class-balancing tool.

Ruiz-Casado et al. also use GAN-generated histopathology images to improve breast cancer image classification. Their framing is especially relevant for small datasets, where downsampling the majority class may waste too much data and traditional geometric augmentation may not add enough diversity.

Yuan et al. propose a GAN-based augmentation method with Global Context Attention and UpBlock-CompRes modules to generate higher-quality minority-class histopathology images. This paper is more method-focused than the other two GAN examples because it modifies the generator architecture and evaluates generated-image quality with FID, IS, PSNR, and SSIM as well as downstream classification.

Practical takeaway: GAN augmentation is attractive when minority classes are rare, but the thesis should distinguish visual realism from clinical usefulness. A generated image that improves FID does not automatically improve rare-class decision boundaries or preserve diagnostic morphology.

## 4. Algorithm-Level Losses

Algorithm-level approaches leave the dataset mostly unchanged and alter the training objective so minority-class errors matter more or produce better-separated embeddings.

Mahbub et al. propose Center-Focused Affinity Loss (CFAL) for imbalanced histology classification. CFAL constructs class prototypes in feature space, penalizes difficult samples, reduces intra-class variation, and emphasizes minority-class features. The useful conceptual pieces are: class balancing through an effective-number-of-samples style weighting term, affinity-based comparison to class prototypes instead of only ordinary softmax logits, and a local penalty that focuses training on samples far from their class center. This sits between loss design and representation shaping: the loss is the mechanism, but the intended outcome is a more compact and discriminative embedding space for fine-grained, similarly textured histology classes.

Scholz et al. evaluate imbalance-aware loss functions derived from MCC and F1 score, alone and combined with cross-entropy, across medical image datasets. Their strongest point is practical: metric-derived losses can improve balanced accuracy and minority-class F1, especially when combined with conventional cross-entropy rather than replacing it entirely.

Practical takeaway: loss-based methods are attractive because they do not require generating or deleting data. They are especially relevant when deleting majority samples would discard useful pathology variation or when synthetic images are hard to validate. They should be evaluated with minority-aware metrics, not only accuracy or macro averages, because loss changes can shift calibration and class trade-offs. For fine-grained histology, the most interesting loss designs are not just class weights; they also explicitly shape the embedding space through margins, prototypes, affinities, or metric-derived objectives.

## 5. Hybrid Approaches

Hybrid methods combine data-level changes with algorithm-level or model-level changes.

Guerrero et al. propose a modified copy-paste augmentation for nuclei detection and combine it with loss weight balancing. The method targets instance-level class imbalance in dense histopathology object detection, where naive copy-paste can create harmful overlaps. This is a clear hybrid case: synthetic placement changes the training data, while loss weighting changes the learning signal.

Practical takeaway: hybrid methods often work because imbalance appears at multiple levels. In pathology detection, there can be both foreground-background imbalance and class imbalance among object types. Treating only one level may leave the model biased.

## 6. Representation and Architecture-Level Methods

Representation-level methods try to make rare classes easier to separate rather than only changing sample counts or class weights.

Juyal et al. propose SC-MIL, a supervised contrastive MIL framework for imbalanced pathology classification. The method progressively transitions from bag-level representation learning to classifier learning, using supervised contrastive learning to improve decision boundaries under label imbalance. This is especially relevant for WSI pathology because positive evidence can occupy only a small fraction of a positive slide.

Nouyed et al. address patch-level imbalance in histopathology classification with a divide-and-conquer strategy. They partition data into subgroups, define three classification subproblems, sample discriminative patches using cluster-based feature sampling, train separate models, and integrate their outputs. This is not simple undersampling; it is architecture plus sampling designed around the WSI patch-distribution problem.

Practical takeaway: for WSI tasks, representation and architecture choices may be as important as classical rebalancing. Rare-class failure can come from sparse positive tissue, noisy patch labels, and weak slide labels, not only from fewer class examples.

## 7. Evaluation Practices

Mosquera et al. should be used as evaluation guidance, not as a mitigation-method paper. The paper shows that common metrics can conceal poor minority-class performance under class imbalance and argues for complementary metrics such as AUC-PR, adjusted AUC-PR, and balanced Brier score. It also emphasizes calibration, which is often missing in medical image classifier evaluation.

For thesis experiments, this means:

- Report class-wise sensitivity, specificity, precision, F1, and support.
- Prefer balanced accuracy, macro F1, and AUC-PR when minority-class behavior matters.
- Include calibration metrics or reliability analysis if model scores are interpreted as probabilities.
- Avoid using AUC-ROC or accuracy alone, especially when deployment prevalence differs from test-set prevalence.

## Method Selection Guide

| Problem pattern | Most relevant method family | Papers to start with |
| :--- | :--- | :--- |
| Few minority slides or images | Oversampling, SMOTE-style methods, GAN augmentation | Majeed et al.; Ruiz-Casado et al.; Yuan et al. |
| Variable-size WSI bags | Feature-level augmentation or MIL-aware representation learning | Chen and Lu; Juyal et al. |
| Sparse positive tissue inside positive WSIs | MIL architecture, contrastive learning, patch selection | Juyal et al.; Nouyed et al. |
| Fine-grained classes with similar morphology | Prototype/affinity losses and representation shaping | Mahbub et al. |
| Medical image classifiers with biased objectives | Metric-derived or imbalance-aware losses | Scholz et al. |
| Dense nuclei/object detection imbalance | Copy-paste plus loss weighting | Guerrero et al. |
| Need to evaluate fairly under prevalence shift | Precision-recall and calibration metrics | Mosquera et al. |

## Thesis Takeaways

The papers suggest four practical layers for handling class imbalance in computational pathology:

1. Start with simple baselines: class-balanced sampling, SMOTE-style oversampling, and class-weighted or focal-style losses.
2. For WSI classification, add MIL-aware methods because imbalance is often inside the bag, not only across slide labels.
3. Treat GAN or synthetic augmentation as a separate experimental condition and validate both image quality and downstream rare-class performance.
4. Evaluate with minority-aware and calibration-aware metrics; otherwise, improvements may be invisible or misleading.

The strongest CPath-specific direction is likely a combination of representation-level MIL methods and imbalance-aware evaluation. Classical resampling is still useful as a baseline, but WSI pathology has additional structure: sparse positive regions, weak labels, site shift, and class-dependent morphology. Methods that account for that structure are more defensible than generic tabular resampling alone.
