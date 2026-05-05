## 1. Data-level approaches

These change the **training data distribution** before or during training.

Core idea:

> Make the minority class more visible to the model, or reduce dominance of the majority class.

Subcategories:

| Approach                | What it does                                           | Search keywords                                                                                                        |
| ----------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Oversampling**        | Repeats or synthesizes minority-class samples          | `minority oversampling class imbalance`, `random oversampling imbalanced learning`                                     |
| **Undersampling**       | Removes majority-class samples                         | `random undersampling class imbalance`, `majority class undersampling`                                                 |
| **Hybrid sampling**     | Combines over- and undersampling                       | `hybrid sampling imbalanced learning`, `over-under sampling class imbalance`                                           |
| **SMOTE-style methods** | Generates synthetic minority examples in feature space | `SMOTE computer vision`, `Borderline-SMOTE image classification`, `ADASYN class imbalance`                             |
| **Image augmentation**  | Applies transformations to minority-class images       | `minority class data augmentation`, `class-balanced data augmentation`, `imbalanced image classification augmentation` |

The review notes that classic resampling is common in data mining, but less straightforward in computer vision because images have spatial structure and high dimensionality; SMOTE-style approaches are therefore more limited in vision than in tabular data. 

Good search strings:

```text
data-level methods for class imbalance computer vision
minority class data augmentation imbalanced image classification
SMOTE ADASYN Borderline-SMOTE computer vision class imbalance
class-balanced sampling deep learning image classification
```

## 2. Synthetic data generation / generative augmentation

This is still data-level, but important enough to search separately.

Core idea:

> Generate new minority-class samples, often with GANs or other generative models.

Subcategories:

| Approach                                 | Search keywords                                                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **GAN-based augmentation**               | `GAN class imbalance image classification`, `GAN minority class augmentation`, `imbalanced learning GAN computer vision` |
| **Conditional GANs**                     | `conditional GAN imbalanced classification`, `cGAN minority class image generation`                                      |
| **VAE / diffusion augmentation**         | `VAE class imbalance image classification`, `diffusion model data augmentation rare classes`                             |
| **Medical-image synthetic augmentation** | `GAN class imbalance medical imaging`, `synthetic histopathology image generation class imbalance`                       |

The review explicitly lists GAN-generated images as a popular data-level approach and discusses generative models as augmentation tools for minority classes. 

Good search strings:

```text
GAN-based data augmentation for imbalanced image classification
synthetic minority image generation class imbalance
generative models for long-tailed visual recognition
diffusion augmentation rare class classification medical imaging
```

## 3. Algorithm-level approaches

These change the **learning objective or classifier behavior**, not the dataset itself.

Core idea:

> Keep the imbalanced dataset, but make the model care more about minority-class errors.

Subcategories:

| Approach                                | What it does                                            | Search keywords                                                                                                |
| --------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Cost-sensitive learning**             | Penalizes minority-class errors more strongly           | `cost-sensitive learning class imbalance`, `cost-sensitive deep learning imbalanced classification`            |
| **Class weighting**                     | Gives higher loss weight to minority classes            | `class-weighted loss imbalanced image classification`, `weighted cross entropy class imbalance`                |
| **Focal loss**                          | Down-weights easy examples and focuses on hard examples | `focal loss class imbalance`, `focal loss long-tailed classification`, `focal loss medical image segmentation` |
| **Class-balanced loss**                 | Uses effective number of samples / reweighting          | `class-balanced loss effective number of samples`, `class-balanced loss long-tailed recognition`               |
| **Logit adjustment / prior correction** | Corrects logits using class priors                      | `logit adjustment long-tailed recognition`, `balanced softmax long-tailed classification`                      |
| **Margin-based losses**                 | Changes decision margins for minority classes           | `LDAM loss long-tailed recognition`, `large margin loss class imbalance`                                       |

The review highlights cost-sensitive learning, class weights, and loss-function design as central algorithm-level strategies; it also says loss functions are among the most effective deep-learning remedies, with focal loss especially popular.  

Good search strings:

```text
cost-sensitive deep learning for imbalanced image classification
weighted cross entropy class imbalance computer vision
focal loss imbalanced classification object detection segmentation
class-balanced loss long-tailed recognition
logit adjustment long-tailed classification
balanced softmax long-tailed recognition
LDAM loss class imbalance
```

## 4. Hybrid approaches and ensembles

These combine **data-level balancing** with **algorithm/model-level changes**.

Core idea:

> Use multiple learners, resampling, boosting, or cascades so that minority classes get better represented during training.

Subcategories:

| Approach                         | Search keywords                                                                              |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| **Bagging-based ensembles**      | `bagging imbalanced learning`, `balanced random forest class imbalance`                      |
| **Boosting-based methods**       | `AdaBoost class imbalance`, `RUSBoost imbalanced learning`, `SMOTEBoost class imbalance`     |
| **Balanced cascades**            | `balanced cascade imbalanced learning`, `cascade forest class imbalance`                     |
| **Deep ensembles for imbalance** | `ensemble deep learning imbalanced image classification`, `long-tailed recognition ensemble` |
| **Sampling + loss combinations** | `class-balanced sampling focal loss`, `oversampling weighted loss imbalanced classification` |

The review gives RUSBoost as an example of a hybrid of resampling and boosting, and mentions balance cascade as another hybrid/cascade-style method. 

Good search strings:

```text
hybrid approaches class imbalance computer vision
RUSBoost imbalanced learning
SMOTEBoost imbalanced classification
ensemble learning imbalanced image classification
balanced cascade imbalanced learning
class-balanced sampling with focal loss
```

## 5. Representation-learning and architecture-level approaches

This category is less “classic imbalance learning,” but very relevant in modern computer vision and computational pathology.

Core idea:

> Improve the learned feature space so rare classes become separable even with few examples.

Subcategories:

| Approach                                         | What to search                                                                                                                               |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Transfer learning / pre-trained CNNs**         | `transfer learning imbalanced image classification`, `pretrained CNN class imbalance medical imaging`                                        |
| **Foundation models / self-supervised learning** | `self-supervised learning class imbalance`, `foundation models rare disease classification histopathology`                                   |
| **Metric learning**                              | `metric learning class imbalance`, `supervised contrastive learning long-tailed recognition`, `prototype learning imbalanced classification` |
| **Decoupled representation/classifier learning** | `decoupled learning long-tailed recognition`, `classifier re-training long-tailed classification`                                            |
| **Vision transformers**                          | `vision transformer class imbalance`, `transformer long-tailed recognition`                                                                  |

The review mentions that distance metric learning is another imbalance-treatment strategy, although less common in vision than loss-based and augmentation-based methods. 

Good search strings:

```text
metric learning for imbalanced image classification
contrastive learning long-tailed recognition
self-supervised learning for class imbalance computer vision
transfer learning imbalanced medical image classification
decoupled representation learning long-tailed recognition
vision transformers for long-tailed recognition
```

## A useful search map

Use these terms depending on the level of specificity:

| You want papers on...        | Search phrase                                                 |
| ---------------------------- | ------------------------------------------------------------- |
| General overview             | `class imbalance computer vision survey`                      |
| Modern deep-learning methods | `deep learning class imbalance image classification`          |
| Long-tail setting            | `long-tailed visual recognition survey`                       |
| Sampling methods             | `resampling methods imbalanced image classification`          |
| Augmentation                 | `minority class data augmentation computer vision`            |
| GANs                         | `GAN augmentation imbalanced image classification`            |
| Loss functions               | `loss functions for class imbalance deep learning`            |
| Focal loss                   | `focal loss imbalanced classification computer vision`        |
| Class-balanced loss          | `class-balanced loss long-tailed recognition`                 |
| Ensembles                    | `ensemble learning for imbalanced image classification`       |
| Medical imaging              | `class imbalance medical image classification deep learning`  |
| Histopathology               | `class imbalance histopathology classification`               |
| WSI classification           | `class imbalance whole slide image classification`            |
| Segmentation                 | `class imbalance medical image segmentation loss function`    |
| Object detection             | `foreground background imbalance object detection focal loss` |

## For your topic, I would start with these paper-search clusters

For computational pathology / WSI classification, the most relevant clusters are probably:

```text
class imbalance histopathology classification
long-tailed recognition histopathology
rare class classification whole slide images
class-balanced loss medical image classification
focal loss histopathology classification
self-supervised learning rare disease histopathology
foundation model long-tailed histopathology classification
few-shot and long-tailed learning computational pathology
```

A clean mental model:

> **Data-level methods** change what the model sees.
> **Algorithm-level methods** change how errors are penalized.
> **Hybrid methods** combine sampling and learning changes.
> **Representation-learning methods** try to make rare classes separable in feature space.
> **Generative methods** create additional rare-class examples.
