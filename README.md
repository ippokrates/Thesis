# Ανάπτυξη Εργαλείου XAI για Ερμηνεία Αποφάσεων σε Συστήματα Υγείας

Η εργασία εστιαζει σε δυο πεδια:
1. Καρδιολογια: Προβλεψη πιθανοτητας καρδιοπαθειας απο κλινικα δεδομενα με χρηση Random Forest και Deep Neural Network (DNN), καθως και επεξηγηση αποφασεων με SHAP και LIME.
2. Δερματολογια: Ταξινομηση δερματικων αλλοιωσεων (καλοηθεις / κακοηθεις) απο εικονες με χρηση Convolutional Neural Network (CNN / MobileNetV2) και Vision Transformer (ViT-B/16), καθως και οπτικη ερμηνεια με Grad-CAM και Attention Rollout.

Ολα τα αποτελεσματα ενσωματωνονται σε μια διαδραστικη εφαρμογη (Streamlit) καθως και σε αυτονομο εργαλειο γραμμης εντολων (CLI).

---

## Δομη

```text
Thesis/
│
├── app/
│   └── app.py                        # Εφαρμογη Streamlit
│
├── data/
│   ├── HDD/                          # Cleveland Heart Disease Dataset
│   │   ├── heart.csv                 # Αρχικο dataset
│   │   ├── data_preprocessing.py     # Script καθαρισμου και προεπεξεργασιας
│   │   ├── scaler.pkl                # Αποθηκευμενο StandardScaler object
│   │   ├── X_train_ready.csv         # Προεπεξεργασμενα δεδομενα εκπαιδευσης
│   │   ├── X_test_ready.csv          # Προεπεξεργασμενα δεδομενα ελεγχου
│   │   ├── y_train_ready.csv
│   │   └── y_test_ready.csv
│   │
│   └── HAM10000/                     # HAM10000 Dataset
│       ├── HAM10000_metadata.csv     # Μεταδεδομενα ασθενων και διαγνωσεων
│       ├── image_sort.py             # Script ταξινομησης εικονων σε κλασεις
│       └── skin_cancer_data/         # Φακελοι εικονων (benign / malignant)
│
├── saved_models/                     # Αποθηκευμενα εκπαιδευμενα μοντελα
│   ├── rf_model.pkl                  # Random Forest μοντελο
│   ├── dnn_model.keras               # Deep Neural Network (.keras)
│   ├── cnn_tl_skin_cancer.keras      # CNN MobileNetV2 (.keras)
│   └── vit_model_hf.h5               # Βάρη Vision Transformer
│
├── diagnosis/                        # Φακελος αποθηκευσης διαγνωσεων
│
├── training_scripts/                 # Scripts εκπαιδευσης και αναλυσης
│   ├── Random Forest/
│   │   └── rf_model.py               # Εκπαιδευση Random Forest
│   ├── DNN/
│   │   └── dnn.py                    # Εκπαιδευση DNN
│   ├── CNN/
│   │   ├── cnn_imagenet.py           # Εκπαιδευση CNN
│   │   └── cnn_imagenet_gradcam.py   # Παραγωγη Grad-CAM
│   ├── ViT/
│   │   ├── vit_model_hf.py           # Εκπαιδευση Vision Transformer
│   │   └── vit_attention_rollout_hf.py.py # Παραγωγη Attention Rollout
│   ├── compare_shap.py               # Συγκριτικη αναλυση SHAP (RF vs DNN)
│   ├── compare_lime.py               # Συγκριτικη αναλυση LIME (RF vs DNN)
│   ├── shap_analysis.py              # Beeswarm και summary διαγραμματα SHAP
│   └── shap_waterfall.py             # Waterfall διαγραμματα ανα ασθενη
│
├── outputs/                          # Παραγομενα αποτελεσματα και διαγραμματα
│   ├── eda/                          # Διαγραμματα διερευνητικης αναλυσης
│   ├── evaluation/
│   │   ├── training_curves/          # Καμπυλες loss/accuracy
│   │   ├── confusion_matrices/       # Πινακες συγχυσης ολων των μοντελων
│   │   └── roc_curves/               # Καμπυλες ROC
│   └── xai/
│       ├── tabular/                  # SHAP waterfall/beeswarm και LIME plots
│       └── vision/                   # Grad-CAM και Attention Rollout plots
│
├── diagnose.py                       # CLI εργαλειο συνθετης διαγνωσης
├── cnn_worker.py                     # Subprocess worker για το CNN Grad-CAM
├── mappings.py                       # Λεξικα κλινικων ορων στα ελληνικα
├── requirements.txt                  # Λιστα Python dependencies
└── README.md                       
```

---

## Εγκατασταση

1. Δημιουργια εικονικου περιβαλλοντος (venv):
```bash
python3 -m venv .venv
```

2. Ενεργοποιηση του περιβαλλοντος:
- Σε Linux / WSL2 / macOS:
```bash
source .venv/bin/activate
```
- Σε Windows (CMD):
```cmd
.venv\Scripts\activate.bat
```
- Σε Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

3. Εγκατασταση των dependencies:
```bash
pip install -r requirements.txt
```
---

## Λήψη Δεδομένων HAM10000 (για επανεκπαίδευση)

Λόγω μεγέθους οι εικόνες δεν περιλαμβάνονται στο github. Για να επανεκπαιδεύσετε τα μοντέλα:

1. Κατεβάστε τις εικόνες από το Kaggle:  
   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Τοποθετήστε τις εικόνες στον φάκελο:  
   `data/HAM10000/all_ham_images/`
3. Εκτελέστε το script ταξινόμησης:  
   ```bash
   python "data/HAM10000/image_sort.py"


---

## Οδηγιες Χρησης Εφαρμογης

Η διαδραστικη εφαρμογη Streamlit εκκινειται με την εντολη:

```bash
streamlit run app/app.py
```

---

## Αυτονομη Διαγνωση απο Γραμμη Εντολων

Μπορειτε να εκτελεσετε το διαγνωστικο script δερματολογιας απευθειας απο το τερματικο χωρις Streamlit:

```bash
python diagnose.py --image διαδρομη/προς/εικονα.jpg
```

Προαιρετικες παραμετροι:
- `--alpha 0.5`: Ρυθμιση διαφανειας Grad-CAM overlay.

Η συνθετη εικονα αποτελεσματος αποθηκευεται αυτοματα με timestamp στον φακελο `diagnosis/`.

---

## Εκπαιδευση Μοντελων και Αναπαραγωγη Αποτελεσματων

Ολα τα scripts εκτελουνται απο τον κεντρικο φακελο του εργου (`Thesis/`):

1. Προεπεξεργασια Δεδομενων Heart Disease Dataset:
```bash
python data/HDD/data_preprocessing.py
```

2. Εκπαιδευση Μοντελων:
```bash
# Random Forest
python "training_scripts/Random Forest/rf_model.py"

# Deep Neural Network (DNN)
python training_scripts/DNN/dnn.py

# CNN (MobileNetV2)
python training_scripts/CNN/cnn_imagenet.py

# Vision Transformer (ViT-B/16)
python training_scripts/ViT/vit_model_hf.py
```
Τα εκπαιδευμενα μοντελα αποθηκευονται αυτοματα στον φακελο `saved_models/`.

3. Παραγωγη Αναλυσεων XAI:
```bash
# Συγκριση SHAP (RF vs DNN)
python training_scripts/compare_shap.py

# Συγκριση LIME (RF vs DNN)
python training_scripts/compare_lime.py

# SHAP Waterfall ανα ασθενη
python training_scripts/shap_waterfall.py

# SHAP Beeswarm & Summary
python training_scripts/shap_analysis.py

# Grad-CAM για επιλεγμενα δειγματα
python training_scripts/CNN/cnn_imagenet_gradcam.py

# Attention Rollout για επιλεγμενα δειγματα
python training_scripts/ViT/vit_attention_rollout_hf.py
```

4. Διερευνητικη Αναλυση Δεδομενων (EDA):
```bash
python outputs/eda/class_distribution.py
python outputs/eda/7class_distribution.py
python outputs/eda/correlation_heatmap.py
```

---

## Άδεια Χρήσης

* **Πηγαίος Κώδικας:** Όλος ο πρωτότυπος κώδικας του παρόντος αποθετηρίου (εφαρμογή Streamlit, CLI εργαλείο, scripts επεξεργασίας, εκπαίδευσης και παραγωγής εξηγήσεων XAI) διατίθεται υπό την άδεια [MIT License](LICENSE).
* **Σύνολα Δεδομένων (Datasets):** Τα δεδομένα που χρησιμοποιήθηκαν δεν καλύπτονται από την άδεια του κώδικα και υπόκεινται αποκλειστικά στους όρους και τις άδειες των αρχικών δημιουργών τους:
  * **HAM10000 Dataset:** Διατίθεται υπό την άδεια [Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) από τους Tschandl et al. (επιτρέπεται μόνο για ακαδημαϊκή / μη-εμπορική έρευνα).
  * **Cleveland Heart Disease Dataset:** Προέρχεται από το αποθετήριο [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) (Detrano et al.).
* **Προεκπαιδευμένα Βάρη:** Οι βασικές αρχιτεκτονικές (MobileNetV2 μέσω Keras, ViT μέσω Hugging Face) διέπονται από τις αντίστοιχες άδειες χρήσης των παρόχων τους.

> **Ιατρική Αποποίηση Ευθύνης (Disclaimer):** Το λογισμικό και τα μοντέλα αναπτύχθηκαν στα πλαίσια ακαδημαϊκής πτυχιακής εργασίας για ερευνητικούς και εκπαιδευτικούς σκοπούς. Δεν αποτελούν εγκεκριμένο ιατροτεχνολογικό προϊόν και δεν προορίζονται για πραγματική κλινική διάγνωση ή λήψη ιατρικών αποφάσεων σε ασθενείς.