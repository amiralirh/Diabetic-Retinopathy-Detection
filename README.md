# Diabetic Retinopathy Detection using Deep Learning 👁️🧠

An end-to-end automated machine learning pipeline for detecting and grading the severity of Diabetic Retinopathy (DR) from high-resolution retinal fundus images, utilizing the APTOS 2019 Blindness Detection dataset.

## Project Overview

Diabetic Retinopathy is a leading cause of preventable blindness worldwide. Early detection is critical, but manual screening by ophthalmologists is resource-intensive and prone to subjective variance. This project builds a robust, scalable deep learning solution to automatically classify retinal images into five severity grades. 

This repository contains the complete pipeline—from advanced domain-specific image preprocessing and `tf.data` optimization to transfer learning with EfficientNetB0, latent space visualization (t-SNE), and model explainability (Grad-CAM).

## Dataset

The dataset used is the **APTOS 2019 Blindness Detection Dataset** (provided by Aravind Eye Hospital). Images are clinically rated on a scale of 0 to 4:

| Grade | Diagnosis | Description |
| :--- | :--- | :--- |
| **0** | No DR | Healthy retina |
| **1** | Mild | Microaneurysms present |
| **2** | Moderate | Multiple microaneurysms, dot-and-blot hemorrhages |
| **3** | Severe | Significant hemorrhages, IRMA |
| **4** | Proliferative DR | Neovascularization, preretinal hemorrhage |

*Note: The dataset exhibits severe class imbalance, which is addressed programmatically via cost-sensitive learning (custom class weights).*

## Technical Pipeline & Methodology

### 1. Advanced Retinal Image Preprocessing
Raw fundus images contain lighting artifacts, noise, and uninformative black borders. To expose clinical biomarkers, an offline preprocessing pipeline was engineered using OpenCV:
* **Automated Contour Cropping:** Detects and crops excess black backgrounds.
* **Targeted Green Channel Enhancement:** Isolates the green RGB channel (which provides the highest contrast for blood vessels and hemorrhages), applies a Median Filter to remove noise, and utilizes **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to locally amplify subtle lesions.
* **Aspect-Ratio Preserving Padding:** Resizes images to `224x224` while maintaining anatomical geometry by padding remaining space, preventing structural distortion.

### 2. High-Performance ETL Pipeline (`tf.data`)
To prevent GPU starvation during training, the data pipeline is optimized using TensorFlow's `tf.data.Dataset` API:
* Parallelized data loading and augmentation (`AUTOTUNE`).
* Memory caching to eliminate disk I/O bottlenecks after the first epoch.
* **Clinical Augmentation:** Applies random flips, rotations, and subtle brightness/contrast shifts. (Color shifting like hue/saturation is strictly avoided to preserve the diagnostic integrity of lesion colors).

### 3. Model Architecture
The core model leverages **Transfer Learning** using **EfficientNetB0** (pre-trained on ImageNet).
To combat the high risk of overfitting on medical data, the custom classification head is heavily regularized:
* **Global Average Pooling 2D:** For spatial dimension reduction.
* **Dense Layers with L2 Regularization:** Penalty `0.005` applied to weights.
* **Aggressive Dropout:** `65%` dropout rate forces the network to learn robust, redundant biomarkers.

### 4. Two-Phase Training Strategy
* **Phase 1 (Warm-up):** The EfficientNetB0 base is frozen. Only the heavily regularized dense layers are trained for 30 epochs to stabilize the classification head.
* **Phase 2 (Fine-Tuning):** The entire network is unfrozen and trained for 40 epochs at a highly constrained learning rate (`1e-05`) to adapt deep convolutional filters to retinal pathologies.
* **Callbacks:** Governed by `EarlyStopping` (patience: 20) and `ReduceLROnPlateau` (patience: 15).

## Evaluation & Explainability

Accuracy is insufficient for imbalanced medical data. The model is evaluated across several strict clinical and mathematical dimensions:
* **Quadratic Weighted Kappa (QWK):** The official competition metric, which heavily penalizes predictions that diverge significantly from the true clinical grade.
* **Multi-Class ROC-AUC:** Measures the model's ability to cleanly separate individual DR stages.
* **Latent Space t-SNE:** Extracts features from the top convolutional layer and compresses them into 2D space, confirming that the model learns distinct biological clusters corresponding to disease severity.
* **Grad-CAM:** Provides visual interpretability by generating heatmaps over the original retinas, proving the CNN is focusing on actual lesions (exudates, hemorrhages) rather than background noise.

## Getting Started

### Prerequisites
* Python 3.8+
* TensorFlow 2.x
* OpenCV, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

