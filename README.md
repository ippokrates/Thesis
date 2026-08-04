# Medical AI & Explainable AI (XAI) Thesis Repository

This repository contains the complete codebase, data preprocessing pipelines, model architectures (CNN, ViT, DNN, Random Forest), explainable AI (XAI) frameworks (Grad-CAM, SHAP, LIME), and evaluation scripts for the Master's Thesis on **Explainable AI in Medical Image and Tabular Diagnostics**.

---

## 📂 Repository Structure & Overview

```
Thesis/
├── app.py                         # Web application / dashboard entry point
├── compare_cnn_vit.py             # Comparative analysis script for CNN vs ViT models
├── mappings.py                    # Feature names mapping and Greek translations for tabular data & UI
├── requirements.txt               # Python package dependencies
├── thesis_structure.json          # JSON schema of thesis document outline & chapters
├── changelog.md                   # Log of updates, bug fixes, and development progress
├── improvements_guide.md          # Guide detailing dataset, model, and XAI enhancements
│
├── data/                          # Datasets (Image & Tabular)
│   ├── HAM10000/                  # Skin lesion image dataset pipeline
│   │   ├── image sort.py          # Sorts raw HAM10000 images into benign/malignant folders
│   │   ├── all_ham_images/        # Raw ISIC HAM10000 image files
│   │   ├── skin_cancer_data/      # Sorted binary image dataset
│   │   ├── balanced_data/         # Undersampled balanced dataset
│   │   └── augmented_data/        # Augmented balanced dataset
│   └── HDD/                       # Heart Disease Dataset (Tabular) pipeline
│       ├── data_preprocessing.py  # Data cleaning, encoding, train/test split & scaling
│       ├── heart.csv              # Raw tabular dataset
│       ├── X_train_ready.csv      # Scaled training feature matrix
│       ├── X_test_ready.csv       # Scaled test feature matrix
│       ├── y_train_ready.csv      # Training labels
│       ├── y_test_ready.csv       # Test labels
│       └── scaler.pkl             # Serialized StandardScaler object
│
├── training_scripts/              # Model training & XAI scripts
│   ├── CNN/                       # Convolutional Neural Network pipeline
│   │   ├── balance_dataset.py     # Undersampling script for class balancing
│   │   ├── augment_malignant.py   # Offline image augmentation for minority class
│   │   ├── phase1_cnn.py          # CNN model creation, training & evaluation
│   │   └── phase2_gradcam.py      # Grad-CAM visual explanation generator
│   ├── ViT/                       # Vision Transformer pipeline
│   │   └── vit_model.py           # Fine-tuning ViT-B/16 from TensorFlow Hub
│   ├── DNN/                       # Deep Neural Network for tabular data
│   │   ├── dnn.py                 # Multi-Layer Perceptron (MLP) training
│   │   ├── dnn_lime.py            # Local Interpretable Model-agnostic Explanations (LIME)
│   │   └── dnn_shap.py            # DeepExplainer SHAP attribution analysis
│   ├── Random Forest/             # Machine Learning baseline
│   │   ├── rf_model.py            # Random Forest Classifier training & cross-validation
│   │   └── rf_shap.py            # TreeExplainer SHAP attribution analysis
│   ├── compare_shap.py            # Comparative SHAP analysis between RF and DNN
│   ├── shap_waterfall.py          # Individual patient SHAP waterfall plot generator
│   └── xai_outputs/               # Generated heatmaps, HTML reports, and SHAP plots
│
├── saved_models/                  # Serialized trained model weights
│   ├── cnn_skin_cancer.h5         # Trained CNN model checkpoint
│   ├── vit_model.h5               # Trained Vision Transformer weights
│   ├── dnn_model.keras            # Trained tabular Deep Neural Network
│   └── rf_model.pkl               # Serialized Random Forest classifier
│
└── Word/                          # Thesis write-ups & literature references
    ├── Lecun2015.pdf              # Deep learning foundational literature paper
    └── phges.docx                 # Draft thesis notes and documents
```

---

## 📜 Detailed File Descriptions

### 🟢 Root Directory Scripts & Files

| File | Description |
| :--- | :--- |
| [app.py](file:///home/ippo/Desktop/Thesis/app.py) | Streamlit dashboard web interface entry point for real-time model inference and XAI visualization. |
| [compare_cnn_vit.py](file:///home/ippo/Desktop/Thesis/compare_cnn_vit.py) | Loads trained CNN (128x128) and Vision Transformer (224x224) models, evaluates random test samples from HAM10000, and displays side-by-side predictions with confidence scores. |
| [mappings.py](file:///home/ippo/Desktop/Thesis/mappings.py) | Contains translation dictionaries mapping raw tabular dataset feature codes (e.g., `cp`, `thal`, `exang`) to user-friendly Greek labels and dropdown options for UI display. |
| [thesis_structure.json](file:///home/ippo/Desktop/Thesis/thesis_structure.json) | Structured outline defining thesis chapters, sub-sections, and progress tracking. |
| [improvements_guide.md](file:///home/ippo/Desktop/Thesis/improvements_guide.md) | Technical documentation detailing iterative enhancements applied to datasets, model hyperparameters, and XAI interpretability workflows. |
| [changelog.md](file:///home/ippo/Desktop/Thesis/changelog.md) | Project commit and execution history tracking model training runs and data preprocessing updates. |

---

### 🖼️ Skin Lesion Diagnosis (Image Pipeline: HAM10000)

Located in `data/HAM10000/` and `training_scripts/CNN/` / `training_scripts/ViT/`:

#### 1. Data Processing Scripts
* **[data/HAM10000/image sort.py](file:///home/ippo/Desktop/Thesis/data/HAM10000/image%20sort.py)**
  Reads metadata from `HAM10000_metadata.csv` and categorizes images from `all_ham_images/` into binary folders: `malignant` (`mel`, `bcc`, `akiec`) and `benign` (`nv`, `bkl`, `df`, `vasc`), outputting to `data/HAM10000/skin_cancer_data/`.

* **[training_scripts/CNN/balance_dataset.py](file:///home/ippo/Desktop/Thesis/training_scripts/CNN/balance_dataset.py)**
  Performs random undersampling on the majority `benign` class to construct a 1:1 balanced dataset in `data/HAM10000/balanced_data/`.

* **[training_scripts/CNN/augment_malignant.py](file:///home/ippo/Desktop/Thesis/training_scripts/CNN/augment_malignant.py)**
  Addresses class imbalance via targeted data augmentation (rotation, flipping, zooming, and brightness adjustments) on `malignant` images until the dataset count matches `benign` images, storing outputs in `data/HAM10000/augmented_data/`.

#### 2. Model Architectures & Training
* **[training_scripts/CNN/phase1_cnn.py](file:///home/ippo/Desktop/Thesis/training_scripts/CNN/phase1_cnn.py)**
  Constructs and trains a 3-layer Convolutional Neural Network (Conv2D -> MaxPooling -> Dropout -> Dense) with online data augmentation and EarlyStopping on 128x128 images. Saves model weights to `saved_models/cnn_skin_cancer.h5` and training loss/accuracy curves.

* **[training_scripts/ViT/vit_model.py](file:///home/ippo/Desktop/Thesis/training_scripts/ViT/vit_model.py)**
  Leverages Google's pre-trained Vision Transformer (`ViT-B/16`) via TensorFlow Hub. Configures a custom classification head on 224x224 input images and saves trained weights to `saved_models/vit_model.h5`.

#### 3. Explainable AI (Visual Explanations)
* **[training_scripts/CNN/phase2_gradcam.py](file:///home/ippo/Desktop/Thesis/training_scripts/CNN/phase2_gradcam.py)**
  Implements Grad-CAM (Gradient-weighted Class Activation Mapping) on the final convolutional layer of the trained CNN. Computes feature activation maps to generate heatmaps highlighting image regions driving model predictions.

---

### ❤️ Heart Disease Risk Prediction (Tabular Pipeline: HDD)

Located in `data/HDD/`, `training_scripts/DNN/`, and `training_scripts/Random Forest/`:

#### 1. Data Processing Scripts
* **[data/HDD/data_preprocessing.py](file:///home/ippo/Desktop/Thesis/data/HDD/data_preprocessing.py)**
  Cleans raw `heart.csv` data (handles missing values `?`, binarizes target label to healthy `0` vs heart disease `1`), performs stratified 80/20 train/test splitting, applies `StandardScaler` to prevent data leakage, and outputs preprocessed CSV matrices and `scaler.pkl`.

#### 2. Machine Learning & Deep Learning Models
* **[training_scripts/DNN/dnn.py](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/dnn.py)**
  Builds and trains a Deep Neural Network (16-8-1 dense layers with ReLU, Dropout, and Sigmoid output) using Keras. Evaluates model performance via classification reports and confusion matrices, saving the trained model to `saved_models/dnn_model.keras`.

* **[training_scripts/Random Forest/rf_model.py](file:///home/ippo/Desktop/Thesis/training_scripts/Random%20Forest/rf_model.py)**
  Trains a regularized `RandomForestClassifier` (100 estimators, max depth 5) using `scikit-learn`. Runs 5-Fold Stratified Cross-Validation across metrics (Accuracy, F1, Recall, Precision, ROC-AUC) and saves the model to `saved_models/rf_model.pkl`.

#### 3. Explainable AI (Tabular Feature Attribution)
* **[training_scripts/DNN/dnn_shap.py](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/dnn_shap.py)**
  Computes exact SHAP values for the DNN model using `shap.DeepExplainer` (with `GradientExplainer` fallback), generating global feature importance summary plots (`dnn_shap_summary.png`) and saving raw arrays (`dnn_shap_values.npy`).

* **[training_scripts/Random Forest/rf_shap.py](file:///home/ippo/Desktop/Thesis/training_scripts/Random%20Forest/rf_shap.py)**
  Uses `shap.TreeExplainer` on the Random Forest model to calculate Shapley values and visualize global feature attributions.

* **[training_scripts/DNN/dnn_lime.py](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/dnn_lime.py)**
  Applies `LimeTabularExplainer` to compare local feature contributions between the DNN and Random Forest for specific patient cases (healthy vs sick), exporting interactive HTML explanation reports to `training_scripts/xai_outputs/`.

* **[training_scripts/compare_shap.py](file:///home/ippo/Desktop/Thesis/training_scripts/compare_shap.py)**
  Executes side-by-side SHAP evaluation comparing TreeExplainer (RF) and DeepExplainer (DNN) global feature rankings and mean absolute SHAP values.

* **[training_scripts/shap_waterfall.py](file:///home/ippo/Desktop/Thesis/training_scripts/shap_waterfall.py)**
  Generates individual patient SHAP Waterfall plots, mapping unscaled real feature values (e.g. actual blood pressure, cholesterol, age) to positive/negative prediction attributions.

---

## 🛠️ Requirements & Setup

### Prerequisites
* Python 3.10+
* Virtual Environment (recommended)

### Installation
```bash
# Clone the repository
git clone <repo_url>
cd Thesis

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1. Data Preprocessing
```bash
# Prepare tabular Heart Disease dataset
python data/HDD/data_preprocessing.py

# Sort raw HAM10000 image dataset
python "data/HAM10000/image sort.py"

# Apply offline data augmentation for malignant images
python training_scripts/CNN/augment_malignant.py
```

### 2. Training Models
```bash
# Train CNN on skin lesion images
python training_scripts/CNN/phase1_cnn.py

# Train Vision Transformer (ViT)
python training_scripts/ViT/vit_model.py

# Train Tabular Deep Neural Network (DNN)
python training_scripts/DNN/dnn.py

# Train Random Forest baseline
python "training_scripts/Random Forest/rf_model.py"
```

### 3. Model Comparisons & XAI Interpretability
```bash
# Compare CNN vs ViT prediction outputs
python compare_cnn_vit.py

# Generate Grad-CAM heatmaps for CNN
python training_scripts/CNN/phase2_gradcam.py

# Generate SHAP attributions & comparison plots
python training_scripts/compare_shap.py
python training_scripts/shap_waterfall.py

# Generate LIME HTML explanation reports
python training_scripts/DNN/dnn_lime.py
```

---

## 🧠 Models & Explainability Summary

| Domain | Model Architecture | XAI Technique | Output Target |
| :--- | :--- | :--- | :--- |
| **Image Analysis** | Custom CNN (128x128) | Grad-CAM | Malignant vs Benign Skin Lesion |
| **Image Analysis** | Vision Transformer ViT-B/16 (224x224) | Attention / Comparative Inference | Malignant vs Benign Skin Lesion |
| **Tabular Diagnostics** | Multi-Layer Perceptron (DNN: 16-8-1) | SHAP (DeepExplainer), LIME | Heart Disease Risk Prediction |
| **Tabular Diagnostics** | Random Forest Classifier (100 Trees) | SHAP (TreeExplainer), LIME | Heart Disease Risk Prediction |
