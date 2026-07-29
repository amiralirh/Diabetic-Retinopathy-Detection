### Diabetic Retinopathy Detection using Deep Learning 👁️🧠
# Diabetic Retinopathy Classification: ResNet50 and EfficientNetB0 Comparison

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

This repository contains a machine learning pipeline for classifying the severity of Diabetic Retinopathy (DR) from retinal fundus images. It compares two transfer learning architectures (ResNet50 and EfficientNetB0) on the APTOS 2019 dataset, focusing on clinical image preprocessing, cost-sensitive training, and model interpretability.

## Dataset

The project uses the **APTOS 2019 Blindness Detection Dataset**. Images are rated on a severity scale of 0 to 4:

| Grade | Diagnosis | Description |
| :--- | :--- | :--- |
| **0** | No DR | Healthy retina |
| **1** | Mild | Microaneurysms present |
| **2** | Moderate | Multiple microaneurysms, dot-and-blot hemorrhages |
| **3** | Severe | Significant hemorrhages, IRMA |
| **4** | Proliferative DR | Neovascularization, preretinal hemorrhage |

*Note: The dataset has a natural class imbalance, which is addressed during training using calculated class weights.*

## Methodology

### 1. Image Preprocessing
Fundus images vary in lighting and border size. The offline preprocessing pipeline standardizes the images using OpenCV:
* **Contour Cropping:** Detects and removes uninformative black background borders.
* **Green Channel Enhancement:** Isolates the green channel (which provides better contrast for blood vessels), applies a median filter for noise reduction, and uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve local contrast.
* **Aspect-Ratio Padding:** Scales the longest dimension to `224` pixels while padding the remainder with black space to prevent anatomical distortion, resulting in a `224x224` image.

### 2. Data Pipeline
Data loading is managed via TensorFlow's `tf.data.Dataset` API:
* Implements parallelized loading, memory caching, and prefetching to optimize I/O.
* **Augmentation:** Applies random flips, rotations, and minor brightness/contrast adjustments. Color transformations (such as hue and saturation) are excluded to preserve the diagnostic integrity of lesion colors.

### 3. Model Architectures
The project evaluates two ImageNet-pretrained base models, adapted with custom classification heads:
* **ResNet50:** Includes Global Average Pooling (GAP), a 256-unit Dense layer, and 50% Dropout.
* **EfficientNetB0:** Includes GAP, a 256-unit Dense layer, 65% Dropout, and L2 Regularization (weight decay of 0.005).

### 4. Training Strategy
Both models are trained using a two-phase approach:
* **Phase 1 (Warm-up):** The base layers are frozen. The custom top layers are trained with an Adam optimizer to initialize the classification head.
* **Phase 2 (Fine-Tuning):** The entire network is unfrozen and trained with a reduced learning rate (1e-05) to refine the convolutional filters for specific retinal features.
* Training is managed by `EarlyStopping` and `ReduceLROnPlateau` callbacks to monitor validation accuracy.

## Evaluation & Interpretability

The models are evaluated using metrics suitable for medical classification and class imbalance:
* **Quadratic Weighted Kappa (QWK):** The primary metric, which calculates inter-rater agreement and penalizes predictions based on their distance from the true severity grade.
* **ROC-AUC:** Assesses the model's capacity to distinguish between specific disease classes.
* **t-SNE:** Visualizes the high-dimensional feature representations in a 2D space to observe class clustering.
* **Grad-CAM:** Generates heatmaps over the input images to indicate which spatial regions (e.g., specific lesions) most influenced the model's predictions.

## Getting Started

### Prerequisites
* Python 3.8+
* TensorFlow 2.x
* OpenCV, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

### Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/YourUsername/APTOS-Diabetic-Retinopathy.git](https://github.com/YourUsername/APTOS-Diabetic-Retinopathy.git)
   cd APTOS-Diabetic-Retinopathy
