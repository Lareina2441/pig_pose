# Pig Posture Recognition (Kaggle)

**Macro-F1: 0.907**

This repository contains a complete, high-performance solution for the Kaggle competition **Pig Posture Recognition**, achieving a **macro-averaged F1 score of 0.907** on the public leaderboard.

The task is to classify the posture of individual pigs from real farm images, given pre-annotated bounding boxes, under challenging conditions such as occlusion, cluttered backgrounds, and varying illumination.

---

## 🧠 Key Ideas

* **Instance-level classification** using provided bounding boxes (no detection required)
* **Aspect-ratio preserving cropping (letterbox)** to avoid geometric distortion
* **Strong CNN backbone (EfficientNet-B3)** for visual feature extraction
* **Explicit geometric metadata fusion** (bounding box shape priors)
* **Class imbalance handling** via *class-weighted Focal Loss*
* **Test-Time Augmentation (TTA)** for more robust inference
* **Careful data splitting** to avoid image-level leakage

---

## 📁 Dataset Overview

Each pig instance is annotated with:

* An **axis-aligned bounding box** `[x, y, w, h]`
* One of **five posture classes**:

  * `0`: Lateral lying (left)
  * `1`: Lateral lying (right)
  * `2`: Sitting
  * `3`: Standing
  * `4`: Sternal lying

Multiple pigs may appear in a single image.

The evaluation metric is **macro-averaged F1 score**, which places equal importance on all classes and is sensitive to class imbalance.

---

## 🏗️ Pipeline Overview

```
Original Image
   └── Bounding Box Crop (+ Context Padding)
         └── Letterbox Padding (Preserve Aspect Ratio)
               └── Data Augmentation
                     └── CNN Backbone (EfficientNet-B3)
                           └── Feature + Metadata Fusion
                                 └── Focal Loss Training
                                       └── TTA Inference
                                             └── submission.csv
```

---

## 🖼️ Image Preprocessing (No Distortion)

A **critical design choice** in this project is to *never stretch cropped pig images*.

### Steps:

1. Crop the pig using the provided bounding box
2. Add **15% context padding** to include full body contours
3. Pad the crop to a square canvas (letterbox)
4. Resize to a fixed input size (`384 × 384`)

This preserves:

* Body proportions
* Limb orientation
* Posture geometry (crucial for lying vs sitting vs standing)

---

## 🧬 Metadata Fusion (Geometric Priors)

In addition to CNN features, we explicitly inject **geometric information** derived from bounding boxes:

* **Aspect ratio**: `bbox_width / bbox_height`
* **Relative area**: `(bbox_area) / (image_area)`

These features are concatenated with CNN embeddings before classification.

> Intuition:
> Standing pigs tend to be taller and narrower, while lying pigs are wider and flatter.
> Explicit geometry helps disambiguate visually similar textures.

---

## 🧠 Model Architecture

### Backbone

* **EfficientNet-B3 (timm)**
* Pretrained weights loaded manually (offline-friendly)
* Classification head removed (`num_classes=0`)

### Classification Head

```text
[ CNN Features ]
        ⊕
[ Aspect Ratio, Area Ratio ]
        ↓
BatchNorm
Linear (512)
ReLU
Dropout (0.4)
Linear (5 classes)
```

This lightweight fusion significantly improves robustness with minimal overhead.

---

## ⚖️ Loss Function: Class-Weighted Focal Loss

The dataset exhibits **class imbalance**, which hurts macro-F1 if treated naively.

To address this:

* Class weights are computed using `sklearn.compute_class_weight`
* **Focal Loss** is used to:

  * Down-weight easy samples
  * Focus learning on hard and underrepresented postures

This combination stabilizes training and improves minority-class recall.

---

## 🔁 Data Augmentation

Training-time augmentations (Albumentations):

* Horizontal flip
* Vertical flip (valid due to lying postures)
* Rotation (±15°)
* Color jitter (brightness / contrast / saturation)
* ImageNet normalization

Validation and test sets use **normalization only**.

---

## 🧪 Train / Validation Split (Leakage-Safe)

To prevent data leakage:

* Splitting is done using **GroupShuffleSplit**
* Group key: `image_id`
* Ensures pigs from the same image never appear in both train and validation sets

Only **5%** of data is held out for validation to maximize training signal.

---

## 🚀 Training Strategy

* Optimizer: **AdamW**
* Learning rate schedule: **Cosine Annealing**
* Mixed precision training via `torch.cuda.amp`
* Best model selected based on **validation macro-F1**

---

## 🔍 Inference & Test-Time Augmentation (TTA)

During inference:

* Original image prediction
* Horizontally flipped prediction
* Softmax probabilities are averaged

TTA improves stability without increasing model size.

---

## 📊 Final Performance

* **Macro-F1 score: 0.907**
* Strong performance across all five posture classes
* Robust to occlusion, scale variation, and lighting changes

---

## 📦 Output

The final output is a CSV file in the required Kaggle format:

```csv
row_id,class_id
test_00001001_0000,3
test_00001001_0001,2
```

---

## 🔧 Environment

* PyTorch
* timm
* Albumentations
* OpenCV
* scikit-learn
* Kaggle GPU (16GB compatible)

---

## ✨ Takeaways

This project demonstrates that:

* **Careful preprocessing** matters more than model size
* **Geometry-aware features** complement CNNs effectively
* **Macro-F1–oriented design** (loss, split, TTA) is critical for imbalanced datasets

---

## 📬 Contact

If you have questions, ideas for improvement, or would like to discuss extensions (pose-based methods, ensembling, or self-supervised pretraining), feel free to open an issue or reach out.

---
