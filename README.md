# Ανάπτυξη Εργαλείου XAI για Ερμηνεία Αποφάσεων σε Συστήματα Υγείας

Η εργασία εστιάζει σε δύο πεδία:
1. Καρδιολογία: Πρόβλεψη πιθανότητας καρδιοπάθειας από κλινικά δεδομένα με χρήση Random Forest και Deep Neural Network (DNN), καθώς και επεξήγηση αποφάσεων με SHAP και LIME.
2. Δερματολογία: Ταξινόμηση δερματικών αλλοιώσεων (καλοήθεις / κακοήθεις) από εικόνες με χρήση Convolutional Neural Network (CNN / MobileNetV2) και Vision Transformer (ViT-B/16), καθώς και οπτική ερμηνεία με Grad-CAM και Attention Rollout.

Όλα τα αποτελέσματα ενσωματώνονται σε μια διαδραστική εφαρμογή (Streamlit) καθώς και σε αυτόνομο εργαλείο γραμμής εντολών (CLI).

---

## Δομή

```text
Thesis/
│
├── app/
│   ├── app.py                        # Streamlit interface
│   ├── derm_tab.py                   # Tab Δερματολογίας
│   ├── cardio_tab.py                 # Tab Καρδιολογίας
│   └── shared.py                     # Κοινός κώδικας
│
├── data/
│   ├── HDD/                          # Cleveland Heart Disease Dataset
│   │   ├── heart.csv                 # Αρχικό dataset
│   │   ├── data_preprocessing.py     # Script καθαρισμού και προεπεξεργασίας
│   │   ├── scaler.pkl                # Αποθηκευμένο StandardScaler object
│   │   ├── X_train_ready.csv         # Προεπεξεργασμένα δεδομένα εκπαίδευσης
│   │   ├── X_test_ready.csv          # Προεπεξεργασμένα δεδομένα ελέγχου
│   │   ├── y_train_ready.csv
│   │   └── y_test_ready.csv
│   │
│   └── HAM10000/                     # HAM10000 Dataset
│       ├── HAM10000_metadata.csv     # Μεταδεδομένα ασθενών και διαγνώσεων
│       ├── image_sort.py             # Script ταξινόμησης εικόνων σε κλάσεις
│       └── skin_cancer_data/         # Φάκελοι εικόνων (benign / malignant)
│
├── saved_models/                     # Αποθηκευμένα εκπαιδευμένα μοντέλα
│   ├── rf_model.pkl                  # Random Forest μοντέλο
│   ├── dnn_model.keras               # Deep Neural Network
│   ├── cnn_tl_skin_cancer.keras      # CNN 
│   └── vit_model_hf.h5               # Vision Transformer
│
├── diagnosis/                        # Φάκελος αποθήκευσης διαγνώσεων
│
├── training_scripts/                 # Scripts εκπαίδευσης και ανάλυσης
│   ├── Random Forest/
│   │   └── rf_model.py               # Εκπαίδευση Random Forest
│   ├── DNN/
│   │   └── dnn.py                    # Εκπαίδευση DNN
│   ├── CNN/
│   │   ├── cnn_imagenet.py           # Εκπαίδευση CNN
│   │   └── cnn_imagenet_gradcam.py   # Παραγωγή Grad-CAM
│   ├── ViT/
│   │   ├── vit_model_hf.py           # Εκπαίδευση Vision Transformer
│   │   └── vit_attention_rollout_hf.py # Παραγωγή Attention Rollout
│   ├── compare_shap.py               # Συγκριτική ανάλυση SHAP (RF vs DNN)
│   ├── compare_lime.py               # Συγκριτική ανάλυση LIME (RF vs DNN)
│   ├── shap_analysis.py              # Beeswarm και summary διαγράμματα SHAP
│   └── shap_waterfall.py             # Waterfall διαγράμματα ανά ασθενή
│
├── outputs/                          # Αποτελέσματα και διαγράμματα
│   ├── eda/                          # Διαγράμματα διερευνητικής ανάλυσης
│   ├── evaluation/
│   │   ├── training_curves/          # Καμπύλες loss/accuracy
│   │   ├── confusion_matrices/       # Πίνακες σύγχυσης
│   │   └── roc_curves/               # Καμπύλες ROC/PR
│   └── xai/
│       ├── tabular/                  # SHAP waterfall/beeswarm και LIME plots
│       └── vision/                   # Grad-CAM και Attention Rollout plots
│
├── diagnose.py                       # CLI εργαλείο διάγνωσης
├── cnn_worker.py                     # Subprocess worker για το CNN - Grad-CAM
├── mappings.py                       # Όροι και ετικέτες χαρακτηριστικών (ελληνικά/αγγλικά) 
├── requirements.txt                  # Python dependencies
└── README.md                       
```

---

## Εγκατάσταση

1. Δημιουργία εικονικού περιβάλλοντος (venv):
```bash
python3 -m venv .venv
```

2. Ενεργοποίηση του περιβάλλοντος:
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

3. Εγκατάσταση των dependencies:
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

## Οδηγίες Χρήσης Εφαρμογής

Η διαδραστική εφαρμογή Streamlit εκκινείται με την εντολή:

```bash
streamlit run app/app.py
```

---

## Αυτόνομη Διάγνωση από Γραμμή Εντολών

Μπορείτε να εκτελέσετε το διαγνωστικό script δερματολογίας απευθείας από το τερματικό χωρίς Streamlit:

```bash
python diagnose.py --image διαδρομη/προς/εικονα.jpg
```

Προαιρετικές παράμετροι:
- `--alpha 0.5`: Ρύθμιση διαφάνειας Grad-CAM overlay.

Η σύνθετη εικόνα αποτελέσματος αποθηκεύεται αυτόματα με timestamp στον φάκελο `diagnosis/`.

---

## Εκπαίδευση Μοντέλων και Αναπαραγωγή Αποτελεσμάτων

Όλα τα scripts εκτελούνται από τον κεντρικό φάκελο του έργου (`Thesis/`):

1. Προεπεξεργασία Δεδομένων Heart Disease Dataset:
```bash
python data/HDD/data_preprocessing.py
```

2. Εκπαίδευση Μοντέλων:
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
Τα εκπαιδευμένα μοντέλα αποθηκεύονται αυτόματα στον φάκελο `saved_models/`.

3. Παραγωγή Αναλύσεων XAI:
```bash
# Σύγκριση SHAP (RF vs DNN)
python training_scripts/compare_shap.py

# Σύγκριση LIME (RF vs DNN)
python training_scripts/compare_lime.py

# SHAP Waterfall ανά ασθενή
python training_scripts/shap_waterfall.py

# SHAP Beeswarm & Summary
python training_scripts/shap_analysis.py

# Grad-CAM για επιλεγμένα δείγματα
python training_scripts/CNN/cnn_imagenet_gradcam.py

# Attention Rollout για επιλεγμένα δείγματα
python training_scripts/ViT/vit_attention_rollout_hf.py
```

4. Διερευνητική Ανάλυση Δεδομένων (EDA):
```bash
python outputs/eda/class_distribution.py
python outputs/eda/7class_distribution.py
python outputs/eda/correlation_heatmap.py
```

---

## Άδεια Χρήσης

* **Πηγαίος Κώδικας:** Όλος ο πρωτότυπος κώδικας του παρόντος αποθετηρίου (εφαρμογή Streamlit, CLI εργαλείο, scripts επεξεργασίας, εκπαίδευσης και παραγωγής εξηγήσεων XAI) διατίθεται υπό την άδεια [MIT License](LICENSE).
* **Σύνολα Δεδομένων (Datasets):** Τα δεδομένα που χρησιμοποιήθηκαν δεν καλύπτονται από την άδεια του κώδικα και υπόκεινται αποκλειστικά στους όρους και τις άδειες των αρχικών δημιουργών τους:
   * **HAM10000 Dataset:** Αντλήθηκε από το [Kaggle (Skin Cancer MNIST: HAM10000)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (αρχική δημοσίευση: Tschandl et al., Nature Scientific Data 2018) και διατίθεται υπό την άδεια [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) αποκλειστικά για μη-εμπορική / ερευνητική χρήση.
  * **Cleveland Heart Disease Dataset:** Προέρχεται από το αποθετήριο [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) (Detrano et al.).
* **Προεκπαιδευμένα Βάρη:** Οι βασικές αρχιτεκτονικές (MobileNetV2 μέσω Keras, ViT μέσω Hugging Face) διέπονται από τις αντίστοιχες άδειες χρήσης των παρόχων τους.

> **Ιατρική Αποποίηση Ευθύνης (Disclaimer):** Το λογισμικό και τα μοντέλα αναπτύχθηκαν στα πλαίσια ακαδημαϊκής πτυχιακής εργασίας για ερευνητικούς και εκπαιδευτικούς σκοπούς. Δεν αποτελούν εγκεκριμένο ιατροτεχνολογικό προϊόν και δεν προορίζονται για πραγματική κλινική διάγνωση ή λήψη ιατρικών αποφάσεων σε ασθενείς.
