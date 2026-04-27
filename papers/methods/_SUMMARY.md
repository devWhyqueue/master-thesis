# Machine Learning Methodologies in Computational Pathology

## 1. Theoretical Foundations and the MIL Problem Formulation

The evolution of **Computational Pathology (CPATH)** is characterized by a strategic departure from fully supervised learning toward **Weakly Supervised Learning (WSL)**.

This transition is a pragmatic response to the **prohibitive cost of pixel-level annotations**, which require expert pathologists to perform labor-intensive delineation of regions of interest (ROIs) across gigapixel **whole-slide images (WSIs)**.

To bypass this bottleneck, **Multiple Instance Learning (MIL)** has been adopted as the framework of choice. MIL shifts the diagnostic burden from dense annotations to coarse-grained, slide-level labels.

Methodologically, the MIL problem can be stated as learning the Bernoulli distribution of the bag label, where the bag-label probability is parameterized by neural networks and trained end-to-end by optimizing the log-likelihood function.

---

## 1.1 Structural Definitions

As formalized by **Ilse et al. (2018)**, the MIL framework organizes data into:

* **Bags**: (X)
* **Instances**: (x_1, \dots, x_K)

In CPATH:

* A **bag** represents a WSI.
* **Instances** are the constituent patches extracted from the digitized slide.

These instances are generally assumed to be unordered and independent.

The mapping of instance-level labels (y_k) to the bag-level label (Y) is defined by the **Standard Multiple Instance (SMI) assumption**:

### Standard MIL Assumption

[
Y = \max_k {y_k}
]

A bag is labeled negative if and only if all instances within it are negative. A single positive instance is sufficient to determine a positive bag label.

---

## 1.2 The Fundamental Theorem of Symmetric Functions

Since patch order is arbitrary, any MIL scoring function (S(X)) must exhibit **permutation invariance**.

Theoretical contributions from **Zaheer et al.** and **Qi et al.** provide the mathematical blueprint for this requirement, though they differ in their rigor.

### Theorem 1: Universal Decomposition

Zaheer et al. established that any symmetric function can be decomposed into:

[
g\left(\sum f(x)\right)
]

This represents a universal mathematical requirement consisting of:

1. An instance-level transformation (f)
2. A permutation-invariant aggregation operation
3. A final transformation (g)

### Theorem 2: Arbitrary Approximation

Qi et al. demonstrated that continuous symmetric functions can be arbitrarily approximated using the **max-pooling operator**.

Together, these results motivate the canonical three-step MIL structure:

1. **Transformation**: (f)
2. **Aggregation**: (\sigma)
3. **Final Transformation**: (g)

This structure serves as the foundation for modern MIL architectures and ensures that models remain robust regardless of the sequence in which patches are processed.

---

## 1.3 Transition Statement

These theoretical constraints require architectures that can effectively parameterize (f) and (g).

However, the specific implementation of these functions creates a fundamental dichotomy in CPATH:

> **Should the model learn from instance-level scores or from high-dimensional embeddings?**

---

# 2. Taxonomy of MIL Approaches: Instance-Based vs. Embedding-Based

Deciding between instance-level and embedding-level approaches is one of the most critical design decisions in a CPATH pipeline.

This choice determines the depth of feature extraction and directly affects the model’s ability to provide both high-order predictive accuracy and clinically actionable interpretability.

---

## 2.1 Comparative Analysis

| Feature              | Instance-Based MIL                                                | Embedding-Based MIL                                          |
| -------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------ |
| **Definition**       | Each patch is reduced to a single score ((m = 1)) before pooling. | Each patch is mapped to a high-dimensional vector ((m > 1)). |
| **Interpretability** | High; directly maps scores to morphology.                         | Moderate; relies on attention weights for visualization.     |
| **Feature Richness** | Low; risks losing morphological nuance.                           | High; captures complex cellular heterogeneity.               |
| **Primary Use Case** | ROI detection, spatial quantification, and segmentation.          | WSI classification, prognosis, and biomarker prediction.     |

---

## 2.2 The Rise and Re-Emergence of Methods

Between **2019 and 2021**, the field saw a significant pivot toward **embedding-based MIL**.

Researchers increasingly recognized that single patch-level scores often failed to capture the morphological heterogeneity of pathology slides, including:

* Staining variation
* Cellular architecture
* Tumor microenvironment structure
* Subtle tissue-level patterns

As a result, low-dimensional and high-dimensional embeddings became the standard for preserving informational richness.

Interestingly, while instance-based methods were almost entirely neglected around 2021, they have recently re-emerged. This rebound is driven by improvements in feature extraction technology. As encoders have become more reliable, the scores assigned to individual patches have gained enough validity to support robust diagnostic quantification again.

---

## 2.3 Transition Statement

Whether using scores or embeddings, the primary performance gains in modern CPATH are realized at the **aggregation stage**, where static pooling has increasingly given way to trainable mechanisms.

---

# 3. The Evolution of MIL Aggregation: From Heuristic Pooling to Trainable Attention

The transition from heuristic operators such as **max pooling** and **mean pooling** to adaptive attention modules marks the professionalization of CPATH diagnostics.

Traditional pooling was often mathematically insufficient for identifying subtle diagnostic evidence distributed unevenly across a gigapixel slide.

---

## 3.1 Attention-Based Deep MIL: ABMIL

**Attention-Based Multiple Instance Learning (ABMIL)**, introduced by **Ilse et al. (2018)**, replaced static pooling with trainable attention mechanisms.

ABMIL introduced:

* **Linear attention**
* **Gated attention**

Unlike rigid max pooling, ABMIL assigns learnable weights (a_k) to every instance.

This allows the model to:

* Prioritize diagnostically relevant subregions
* Suppress irrelevant background tissue
* Learn the importance of each patch during training

In effect, ABMIL parameterizes the diagnostic relevance of each patch.

---

## 3.2 Clustering and Dual-Stream Innovations

The field was further refined by **CLAM** and **DSMIL**.

### CLAM: Clustering-Constrained Attention MIL

**CLAM** was introduced by **Lu et al. (2021)**.

It uses instance-level clustering to generate pseudo-labels. These pseudo-labels provide auxiliary supervision that helps refine the feature space.

This enables the model to better distinguish between:

* Positive diagnostic evidence
* Negative diagnostic evidence
* Ambiguous or weakly informative tissue regions

CLAM is particularly useful when only bag-level labels are available.

---

### DSMIL: Dual-Stream Multiple Instance Learning

**DSMIL**, introduced by **Li et al. (2021)**, introduced the concept of a **critical instance**.

This critical instance anchors the attention mechanism through a form of **quasi-self-attention** or locally constrained multiscale attention.

The model performs query-key matching between:

* The critical diagnostic patch
* All other patches in the slide

This transforms MIL from a simple pooling problem into a more nuanced feature-search problem based on diagnostic relevance.

---

## 3.3 Transition Statement

These attention mechanisms weakened the traditional **independent and identically distributed (i.i.d.)** assumption and paved the way for models that understand spatial and contextual relationships between patches.

---

# 4. Modeling Inter-Instance Relationships: Transformers and Sequence Models

The **Vision Transformer (ViT)** revolution catalyzed a shift from local patch analysis toward global context modeling.

Pathologists do not view cells in isolation. They interpret relationships between:

* Cellular clusters
* Stromal architecture
* Tumor boundaries
* Tissue compartments
* Spatial growth patterns

Modern sequence models attempt to mirror this process computationally.

---

## 4.1 TransMIL and Global Context

**TransMIL**, introduced by **Shao et al. (2021)**, was one of the first successful applications of Transformers to CPATH.

Its **TPT module** models pairwise relationships across the entire patch sequence.

To make this computationally feasible, TransMIL uses the **Nyström method** to approximate self-attention.

This allows the model to capture long-range spatial correlations that traditional attention-based MIL methods often ignore.

---

## 4.2 Addressing the Long-Sequence Challenge

WSIs present an **ultra-long token problem**.

A single slide can contain thousands to hundreds of thousands of patches, which makes standard Transformer architectures computationally expensive or infeasible.

Modern models address this challenge through specialized bottlenecks and efficient sequence modeling strategies.

### Prov-GigaPath

**Prov-GigaPath** by **Xu et al. (2024)** represents a major milestone in the field.

It was pretrained on:

* **1.3 billion patches**
* **171,189 WSIs**

It uses **LongNet** with dilated attention to maintain approximately linear computational complexity across very long patch sequences.

This enables large-scale whole-slide representation learning.

---

### State Space MIL

**State Space MIL (SSM)** offers an alternative to Transformer-based modeling.

Instead of quadratic self-attention, SSM-based models use mechanisms such as:

* **Bi-SSM**
* **2D-CAB blocks**

These architectures model long sequences efficiently and can achieve competitive performance without the quadratic computational cost of standard Transformers.

---

## 4.3 Transition Statement

However, the effectiveness of relationship modeling is fundamentally limited by the quality of the initial patch features.

This has motivated a shift from natural-image encoders toward pathology-specific foundation models.

---

# 5. Feature Representation: From ImageNet to Foundation Models

The domain gap between **ImageNet** and histopathology is substantial.

ImageNet models are trained on natural images, while pathology requires representations that understand:

* Cellular morphology
* Tissue architecture
* Staining patterns
* Nuclear atypia
* Tumor microenvironments
* Histological subtypes

This has led to the rise of domain-specific pretraining.

---

## 5.1 Self-Supervised Learning and Domain Nuance

Self-supervised learning methods such as:

* **SimCLR**
* **MoCo**
* **DINO**

have been adapted into pathology-specific models such as:

* **RetCCL**
* **CTransPath**

Contrastive learning is particularly useful in CPATH because it can learn robust semantic biological features while becoming less sensitive to irrelevant variation, such as stain variability across laboratories.

In this sense, contrastive learning can provide a degree of domain robustness by encouraging the model to focus on meaningful morphology rather than superficial acquisition artifacts.

---

## 5.2 The Dawn of Foundation Models

The field has entered a new phase with large-scale pathology foundation models such as **UNI** and **Prov-GigaPath**.

### UNI

**UNI**, introduced by **Chen et al. (2024)**, uses **DINOv2** as a task-agnostic, general-purpose pathology encoder.

It was trained on more than **100 million images** across diverse tissue types.

UNI aims to provide a reusable pathology representation that can support many downstream tasks.

---

### Prov-GigaPath

**Prov-GigaPath** combines:

* **DINOv2** for local patch-level features
* **Masked Autoencoder (MAE)** objectives for global aggregation
* **LongNet-style sequence modeling** for whole-slide representations

It has set new benchmarks for tasks such as:

* Mutation prediction
* Cancer subtyping
* Whole-slide classification

---

## 5.3 Transition Statement

These unified representations are now enabling more **holistic understanding**, where image data is no longer analyzed in isolation but aligned with genomic, transcriptomic, and clinical information.

---

# 6. Multimodal Fusion and Clinical Frontiers

The frontier of CPATH lies in integrating WSIs with additional biomedical data streams, including:

* Transcriptomics
* Genomics
* Clinical text
* Pathology reports
* Survival outcomes
* Treatment metadata

The goal is to improve prognostic accuracy and clinical decision support.

---

## 6.1 Modality Alignment vs. Modality Aggregation

Modern multimodal CPATH methods can be broadly divided into two categories.

---

### Modality Alignment

Examples include:

* **PLIP**
* **CONCH**

These models align pathology images with text, such as pathology reports or natural-language descriptions.

**PLIP** notably used **OpenPath**, a dataset curated from Medical Twitter, to build image-text pairs.

This enables:

* Zero-shot transfer
* Text-guided retrieval
* Rare pathology recognition
* Natural-language anchoring of visual concepts

---

### Modality Aggregation

Examples include:

* **SurvPath**
* **HECTOR**

These models fuse multiple biomedical data streams, such as:

* WSI patches
* RNA-seq pathways
* Clinical covariates

The key value lies in the interaction between modalities.

Images capture spatial heterogeneity, while transcriptomics captures gene-expression programs. Models such as SurvPath analyze patch-to-pathway interactions to improve survival prediction.

---

## 6.2 Key Challenges in Clinical Applications

### 1. Interpretability

Models must move beyond heatmaps toward richer explanations, including natural-language rationales and clinically meaningful concepts.

This is especially important for regulatory and legal requirements, including explainability expectations under frameworks such as GDPR.

---

### 2. Workflow Integration

CPATH models should assist rather than replace pathologists.

Useful systems should provide multi-level outputs such as:

* Slide-level scores
* Region-level annotations
* Patch-level evidence
* Diagnostic confidence
* Suggested follow-up regions

---

### 3. Privacy

Clinical deployment requires methods that can train across institutions without centralizing sensitive patient data.

This motivates techniques such as:

* Federated learning
* Privacy-preserving training
* Secure multi-site validation

---

### 4. Rare Pathologies

Many diseases suffer from insufficient database completeness and limited annotated examples.

Promising strategies include:

* Few-shot learning
* Domain generalization
* Zero-shot image-text models
* Foundation-model-based transfer learning

---

## 6.3 Transition Statement

MIL has matured from an academic curiosity into a foundational technology for CPATH.

It is now positioned to serve as the backbone of integrated clinical decision-support systems that combine image analysis, molecular data, and clinical context.

---

# 7. Master Thesis Reference Compendium

| Reference                                                                             | Contribution                                                                                                              |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Ilse et al. (2018)** — *Attention-Based Deep Multiple Instance Learning*            | Introduced linear and gated attention mechanisms to replace static pooling with trainable attention weights.              |
| **Lu et al. (2021)** — *Data-Efficient and Weakly Supervised Computational Pathology* | Introduced CLAM, using instance-level clustering and pseudo-labels to refine features and support WSI-level segmentation. |
| **Shao et al. (2021)** — *TransMIL*                                                   | Pioneered Transformer-based CPATH by using the TPT module to capture spatial correlations between patches.                |
| **Li et al. (2021)** — *Dual-Stream MIL*                                              | Introduced DSMIL, using quasi-self-attention anchored by a critical instance and self-supervised contrastive learning.    |
| **Chen et al. (2024)** — *Towards a General-Purpose Foundation Model*                 | Introduced UNI, a foundation encoder trained on more than 100 million images across 20 tissue types.                      |
| **Xu et al. (2024)** — *Prov-GigaPath*                                                | Developed a whole-slide foundation model using LongNet, pretrained on 1.3 billion patches from 171,189 WSIs.              |
| **Zhang et al. (2025)** — *From Patches to WSIs: A Systematic Review*                 | Provided a systematic roadmap and analysis of MIL development trends from 2019 to 2024.                                   |
