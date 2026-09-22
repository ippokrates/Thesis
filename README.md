# Explainable AI in Medical Diagnosis
### Πτυχιακή Εργασία - Επεξηγήσιμη Τεχνητή Νοημοσύνη στην Ιατρική Διάγνωση

---

Η παρούσα πτυχιακή εργασία διερευνά την εφαρμογή της **Επεξηγήσιμης Τεχνητής Νοημοσύνης (XAI)** σε δύο διαφορετικά ιατρικά προβλήματα διάγνωσης:

1. **Πρόβλεψη Καρδιοπάθειας** (δεδομένα πίνακα) με την χρήση του Cleveland Heart Disease Dataset (HDD)
2. **Ταξινόμηση Δερματικών Αλλοιώσεων** (δεδομένα εικόνας) με την χρήση του HAM10000

Για κάθε εργασία, εκπαιδεύονται και αξιολογούνται πολλαπλά μοντέλα μηχανικής μάθησης, τα οποία στη συνέχεια ερμηνεύονται με τεχνικές XAI, με στόχο να γίνουν οι αποφάσεις τους διαφανείς και κλινικά κατανοητές. Τα αποτελέσματα παρουσιάζονται μέσα από ένα διαδραστικό εργαλείο Streamlit.

---

### Σύνολα Δεδομένων

| Σύνολο Δεδομένων | Εργασία | Τύπος | Κλάσεις | Δείγματα |
|---|---|---|---|---|
| Cleveland Heart Disease Dataset (HDD) | Πρόβλεψη Καρδιοπάθειας | Πίνακας | Υγιής / Καρδιοπάθεια | ~303 |
| HAM10000 (ISIC) | Ταξινόμηση Δερματικών Αλλοιώσεων | Εικόνα | Καλοήθεις / Κακοήθεις | ~10.000 |

- **HDD:** 13 κλινικά χαρακτηριστικά (ηλικία, φύλο, τύπος θωρακικού άλγους, χοληστερόλη κ.ά.). Προεπεξεργασία με StandardScaler. Διαχωρισμός: 80% εκπαίδευση / 20% δοκιμή, `random_state=42`.
- **HAM10000:** 7 αρχικές διαγνωστικές κατηγορίες ομαδοποιήθηκαν σε δύο κλάσεις: Καλοήθεις (`nv`, `bkl`, `df`, `vasc`) και Κακοήθεις (`mel`, `bcc`, `akiec`). Στρωματοποιημένος διαχωρισμός 80/20.

---

### Μοντέλα

#### Εργασία 1 Καρδιοπάθεια (Πίνακας)

| Μοντέλο | Ακρίβεια | AUC | F1 (macro) |
|---|---|---|---|
| Random Forest (RF) | 80% | 0.8907 | 0.80 |
| Deep Neural Network (DNN) | 84% | 0.8712 | 0.83 |

- **Random Forest:** Σύνολο δέντρων απόφασης με 5-Fold Stratified Cross-Validation (ROC-AUC: 0.919 +/- 0.027).
- **DNN:** Πλήρως συνδεδεμένο νευρωνικό δίκτυο που εκπαιδεύτηκε με βελτιστοποιητή Adam και Early Stopping.

#### Εργασία 2 Δερματικές Αλλοιώσεις (Εικόνα)

| Μοντέλο | Ακρίβεια | Backbone |
|---|---|---|
| CNN (Μεταφορά Μάθησης) | 85% | MobileNetV2 (ImageNet) |
| Vision Transformer (ViT) | 87% | ViT-B/16 (HuggingFace) |

- **CNN:** Αρχιτεκτονική MobileNetV2 με Global Average Pooling και Dense κεφαλή ταξινόμησης. Εκπαίδευση σε εικόνες 128x128.
- **ViT:** HuggingFace `TFViTModel` με εξαγωγή CLS token και Dense κεφαλή. Εκπαίδευση σε εικόνες 224x224 με πλήρη υποστήριξη gradient για XAI.

---

### Τεχνικές XAI

| Τεχνική | Μοντέλα | Τύπος Δεδομένων | Έξοδος |
|---|---|---|---|
| SHAP (TreeExplainer / DeepExplainer) | RF, DNN | Πίνακας | Σύνοψη σημαντικότητας χαρακτηριστικών & διαγράμματα waterfall |
| LIME | RF, DNN | Πίνακας | Τοπικές επεξηγήσεις ανά ασθενή |
| Grad-CAM | CNN | Εικόνα | Χάρτης θερμότητας βαθμίδας επί της δερματικής αλλοίωσης |
| Attention Rollout | ViT | Εικόνα | Χάρτης διάδοσης πολυ-κεφαλικής προσοχής |

---

### Διαδραστικό Εργαλείο (Streamlit)

Το `app/app.py` είναι το κεντρικό διαδραστικό εργαλείο XAI. Εκτελείται από τον κεντρικό φάκελο:

```bash
streamlit run app/app.py
```

Περιλαμβάνει δύο καρτέλες:

- **Δερματολογία:** Ανέβασμα εικόνας δερματικής βλάβης, εκτέλεση CNN + ViT, εμφάνιση Grad-CAM και Attention Rollout side-by-side, consensus badge και λήψη αποτελέσματος.
- **Καρδιολογία:** Εισαγωγή 13 κλινικών χαρακτηριστικών, πρόβλεψη RF + DNN, SHAP waterfall plots και LIME bar charts για τον συγκεκριμένο ασθενή.

Για τη δερματολογική ενότητα, το CNN και το ViT εκτελούνται σε ξεχωριστές υποδιεργασίες λόγω ασυμβατότητας εκδόσεων της βιβλιοθήκης Keras (Keras 2 για το ViT, Keras 3 για το CNN).

| Αρχείο | Ρόλος |
|---|---|
| `app/app.py` | Κεντρικό Streamlit interface |
| `diagnose.py` | Pipeline δερματολογικής διάγνωσης (ViT + Grad-CAM + Attention Rollout) |
| `cnn_worker.py` | Subprocess worker για φόρτωση CNN και υπολογισμό Grad-CAM |
| `mappings.py` | Ελληνικές ετικέτες για τα 13 χαρακτηριστικά του Cleveland dataset |

---

### Δομή

```
Thesis/
├── app/
│   └── app.py                        # Streamlit XAI εργαλείο (2 καρτέλες)
│
├── data/
│   ├── HDD/                          # Cleveland Heart Disease Dataset (CSV + scaler)
│   └── HAM10000/                     # Μεταδεδομένα HAM10000 + φάκελοι εικόνων
│
├── saved_models/
│   ├── rf_model.pkl                  # Εκπαιδευμένο Random Forest
│   ├── dnn_model.keras               # Εκπαιδευμένο DNN
│   ├── cnn_tl_skin_cancer.keras      # Εκπαιδευμένο CNN (MobileNetV2)
│   └── vit_model_hf.h5               # Βάρη ViT (HuggingFace)
│
├── training_scripts/
│   ├── Random Forest/
│   │   └── rf_model.py               # Εκπαίδευση & αξιολόγηση RF
│   ├── DNN/
│   │   └── dnn.py                    # Εκπαίδευση & αξιολόγηση DNN
│   ├── CNN/
│   │   ├── cnn_imagenet.py           # Εκπαίδευση & αξιολόγηση CNN
│   │   └── cnn_imagenet_gracam.py    # Παραγωγή Grad-CAM
│   ├── ViT/
│   │   ├── vit_model_hf.py           # Εκπαίδευση & αξιολόγηση ViT
│   │   └── vit_attention_rollout_hf.py.py  # Παραγωγή Attention Rollout
│   ├── compare_shap.py               # SHAP: συγκριτικά διαγράμματα RF vs DNN
│   ├── compare_lime.py               # LIME: αναφορές RF vs DNN
│   ├── shap_waterfall.py             # SHAP waterfall ανά ασθενή
│   └── correlation_heatmap.py        # EDA: heatmap συσχέτισης χαρακτηριστικών
│
├── outputs/
│   ├── eda/                          # Διαγράμματα Ανάλυσης Εξερεύνησης Δεδομένων
│   ├── evaluation/
│   │   ├── training_curves/          # Καμπύλες εκπαίδευσης (DNN, CNN, ViT)
│   │   ├── confusion_matrices/       # Πίνακες σύγχυσης (4 μοντέλα)
│   │   └── roc_curves/               # Καμπύλες ROC (RF, DNN + σύγκριση)
│   └── xai/
│       ├── tabular/
│       │   ├── shap/                 # SHAP διαγράμματα + .npy cache
│       │   └── lime/                 # LIME αναφορές (RF & DNN, ανά ασθενή)
│       └── vision/
│           ├── grad_cam/             # Grad-CAM χάρτες (TP, TN, FN)
│           └── attention_rollout/    # Attention Rollout χάρτες (TP, TN, FP, FN)
│
├── diagnose.py                       # Pipeline δερματολογικής διάγνωσης
├── cnn_worker.py                     # Subprocess worker για CNN & Grad-CAM
├── mappings.py                       # Ελληνικές ετικέτες Cleveland dataset
├── requirements.txt
└── README.md
```

---

### Εγκατάσταση

**1. Δημιουργία και ενεργοποίηση εικονικού περιβάλλοντος**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Εγκατάσταση εξαρτήσεων**
```bash
pip install -r requirements.txt
```

> **Σημείωση:** Οι εικόνες HAM10000 δεν συμπεριλαμβάνονται λόγω μεγέθους. Κατεβάστε τις από το [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) και τοποθετήστε τις στους φακέλους:
> - `data/HAM10000/skin_cancer_data/benign/`
> - `data/HAM10000/skin_cancer_data/malignant/`

---

### Εκτέλεση

**Streamlit εργαλείο (κύρια εφαρμογή)**
```bash
streamlit run app/app.py
```

**Δερματολογική διάγνωση (CLI)**
```bash
python3 diagnose.py --image path/to/image.jpg --output_dir diagnosis/
```

**Scripts εκπαίδευσης** (εκτελούνται από τον κεντρικό φάκελο `Thesis/`)
```bash
# Εκπαίδευση Random Forest
python3 training_scripts/Random\ Forest/rf_model.py

# Εκπαίδευση DNN
python3 training_scripts/DNN/dnn.py

# Εκπαίδευση CNN
python3 training_scripts/CNN/cnn_imagenet.py

# Εκπαίδευση ViT
python3 training_scripts/ViT/vit_model_hf.py

# Δημιουργία SHAP διαγραμμάτων σύγκρισης (RF vs DNN)
python3 training_scripts/compare_shap.py

# Δημιουργία LIME αναφορών (RF vs DNN)
python3 training_scripts/compare_lime.py

# Δημιουργία SHAP waterfall διαγραμμάτων
python3 training_scripts/shap_waterfall.py

# Δημιουργία Grad-CAM χαρτών (CNN)
python3 training_scripts/CNN/cnn_imagenet_gracam.py

# Δημιουργία Attention Rollout χαρτών (ViT)
python3 "training_scripts/ViT/vit_attention_rollout_hf.py.py"

# Δημιουργία Heatmap συσχέτισης χαρακτηριστικών (EDA)
python3 training_scripts/correlation_heatmap.py
```
---

### Εξαρτήσεις

| Library | Version |
|---|---|
| TensorFlow | 2.21.0 |
| scikit-learn | 1.9.0 |
| transformers | 4.57.6 |
| streamlit | 1.61.1 |
| shap | 0.52.0 |
| lime | 0.2.0.1 |
| pandas | 3.0.3 |
| numpy | 2.5.1 |
| matplotlib | 3.11.0 |
| seaborn | 0.13.2 |
| joblib | 1.5.3 |
| opencv-python | 5.0.0.93 |
| Pillow | 12.2.0 |

---

---

This thesis investigates the application of **Explainable Artificial Intelligence (XAI)** in two distinct medical diagnostic tasks:

1. **Heart Disease Prediction** (tabular data) - using the Cleveland Heart Disease Dataset (HDD)
2. **Skin Lesion Classification** (image data) - using the HAM10000 dermatology dataset

For each task, multiple machine learning models are trained, evaluated, and interpreted using XAI techniques, aiming to make model decisions transparent and clinically meaningful. Results are presented through an interactive Streamlit tool.

---

### Datasets

| Dataset | Task | Type | Classes | Samples |
|---|---|---|---|---|
| Cleveland Heart Disease Dataset (HDD) | Heart Disease Prediction | Tabular | Healthy / Heart Disease | ~303 |
| HAM10000 (ISIC) | Skin Lesion Classification | Image | Benign / Malignant | ~10,000 |

- **HDD:** 13 clinical features (age, sex, chest pain type, cholesterol, etc.). Preprocessed with StandardScaler. Data split: 80% train / 20% test, `random_state=42`.
- **HAM10000:** 7 original diagnostic categories regrouped into binary classes: Benign (`nv`, `bkl`, `df`, `vasc`) and Malignant (`mel`, `bcc`, `akiec`). Stratified 80/20 split.

---

### Models

#### Task 1 Heart Disease (Tabular)

| Model | Accuracy | AUC | F1 (macro) |
|---|---|---|---|
| Random Forest (RF) | 80% | 0.8907 | 0.80 |
| Deep Neural Network (DNN) | 84% | 0.8712 | 0.83 |

- **Random Forest:** Ensemble of decision trees with 5-Fold Stratified Cross-Validation (ROC-AUC: 0.919 +/- 0.027).
- **DNN:** Fully connected neural network trained with Adam optimizer and Early Stopping.

#### Task 2 Skin Lesion (Image)

| Model | Accuracy | Backbone |
|---|---|---|
| CNN (Transfer Learning) | 85% | MobileNetV2 (ImageNet) |
| Vision Transformer (ViT) | 87% | ViT-B/16 (HuggingFace, google/vit-base-patch16-224) |

- **CNN:** MobileNetV2 backbone with Global Average Pooling + Dense classification head. Fine-tuned on 128x128 images.
- **ViT:** HuggingFace `TFViTModel` with CLS token extraction + Dense head. Fine-tuned on 224x224 images. Full gradient support for XAI.

---

### XAI Techniques

| Technique | Models | Data Type | Output |
|---|---|---|---|
| SHAP (TreeExplainer / DeepExplainer) | RF, DNN | Tabular | Feature importance summary & waterfall plots |
| LIME | RF, DNN | Tabular | Local per-instance explanation reports |
| Grad-CAM | CNN | Image | Gradient-based heatmap over lesion region |
| Attention Rollout | ViT | Image | Multi-head attention propagation map |

---

### Interactive Tool (Streamlit)

`app/app.py` is the main interactive XAI tool. Run it from the project root:

```bash
streamlit run app/app.py
```

It includes two tabs:

- **Dermatology:** Upload a skin lesion image, run CNN + ViT inference, display Grad-CAM and Attention Rollout side-by-side, consensus badge, and PNG download.
- **Cardiology:** Enter 13 clinical features, get RF + DNN predictions, SHAP waterfall plots and LIME bar charts for the specific patient.

For the dermatology tab, the CNN and ViT run in separate subprocesses due to a Keras version conflict (Keras 2 required by ViT via transformers, Keras 3 used by the saved CNN model).

| File | Role |
|---|---|
| `app/app.py` | Main Streamlit interface |
| `diagnose.py` | Dermatology diagnosis pipeline (ViT + Grad-CAM + Attention Rollout) |
| `cnn_worker.py` | Subprocess worker for CNN loading and Grad-CAM computation |
| `mappings.py` | Greek labels for the 13 Cleveland dataset features |

---

### Project Structure

```
Thesis/
├── app/
│   └── app.py                        # Streamlit XAI tool (2 tabs)
│
├── data/
│   ├── HDD/                          # Cleveland Heart Disease Dataset (CSV + scaler)
│   └── HAM10000/                     # HAM10000 metadata + image folders (benign/malignant)
│
├── saved_models/
│   ├── rf_model.pkl                  # Trained Random Forest
│   ├── dnn_model.keras               # Trained DNN
│   ├── cnn_tl_skin_cancer.keras      # Trained CNN (MobileNetV2)
│   └── vit_model_hf.h5               # Trained ViT weights (HuggingFace)
│
├── training_scripts/
│   ├── Random Forest/
│   │   └── rf_model.py               # RF training & evaluation
│   ├── DNN/
│   │   └── dnn.py                    # DNN training & evaluation
│   ├── CNN/
│   │   ├── cnn_imagenet.py           # CNN training & evaluation
│   │   └── cnn_imagenet_gracam.py    # Grad-CAM generation
│   ├── ViT/
│   │   ├── vit_model_hf.py           # ViT training & evaluation
│   │   └── vit_attention_rollout_hf.py.py  # Attention Rollout generation
│   ├── compare_shap.py               # SHAP comparison plots (RF vs DNN)
│   ├── compare_lime.py               # LIME reports (RF vs DNN)
│   ├── shap_waterfall.py             # SHAP waterfall plots per patient
│   └── correlation_heatmap.py        # EDA: feature correlation heatmap
│
├── outputs/
│   ├── eda/                          # Exploratory Data Analysis plots
│   ├── evaluation/
│   │   ├── training_curves/          # Loss & Accuracy curves (DNN, CNN, ViT)
│   │   ├── confusion_matrices/       # Confusion matrices (all 4 models)
│   │   └── roc_curves/               # ROC curves (RF, DNN + comparison)
│   └── xai/
│       ├── tabular/
│       │   ├── shap/                 # SHAP summary, bar, waterfall plots + .npy cache
│       │   └── lime/                 # LIME reports (RF & DNN, per patient)
│       └── vision/
│           ├── grad_cam/             # Grad-CAM heatmaps (TP, TN, FN)
│           └── attention_rollout/    # ViT Attention Rollout maps (TP, TN, FP, FN)
│
├── diagnose.py                       # Dermatology diagnosis pipeline
├── cnn_worker.py                     # Subprocess worker for CNN & Grad-CAM
├── mappings.py                       # Greek labels for Cleveland dataset features
├── requirements.txt
└── README.md
```

---

### Installation & Setup

**1. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> **Note:** The HAM10000 images are not included in this repository due to size constraints.
> Download them from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place them in:
> - `data/HAM10000/skin_cancer_data/benign/`
> - `data/HAM10000/skin_cancer_data/malignant/`

---

### Running the Scripts

**Streamlit tool (main application)**
```bash
streamlit run app/app.py
```

**Dermatology diagnosis (CLI)**
```bash
python3 diagnose.py --image path/to/image.jpg --output_dir diagnosis/
```

**Training scripts** (run from project root `Thesis/`)
```bash
# Train Random Forest
python3 training_scripts/Random\ Forest/rf_model.py

# Train DNN
python3 training_scripts/DNN/dnn.py

# Train CNN
python3 training_scripts/CNN/cnn_imagenet.py

# Train ViT
python3 training_scripts/ViT/vit_model_hf.py

# Generate SHAP comparison plots (RF vs DNN)
python3 training_scripts/compare_shap.py

# Generate LIME reports (RF vs DNN)
python3 training_scripts/compare_lime.py

# Generate SHAP waterfall plots
python3 training_scripts/shap_waterfall.py

# Generate Grad-CAM heatmaps (CNN)
python3 training_scripts/CNN/cnn_imagenet_gracam.py

# Generate Attention Rollout maps (ViT)
python3 "training_scripts/ViT/vit_attention_rollout_hf.py.py"

# Generate EDA correlation heatmap
python3 training_scripts/correlation_heatmap.py
```

---

### Dependencies

| Library | Version |
|---|---|
| TensorFlow | 2.21.0 |
| scikit-learn | 1.9.0 |
| transformers | 4.57.6 |
| streamlit | 1.61.1 |
| shap | 0.52.0 |
| lime | 0.2.0.1 |
| pandas | 3.0.3 |
| numpy | 2.5.1 |
| matplotlib | 3.11.0 |
| seaborn | 0.13.2 |
| joblib | 1.5.3 |
| opencv-python | 5.0.0.93 |
| Pillow | 12.2.0 |

---
