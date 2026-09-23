# Επεξηγησιμη Τεχνητη Νοημοσυνη στην Ιατρικη Διαγνωση
### Explainable AI (XAI) in Medical Diagnosis

Πτυχιακη εργασια με θεμα την αναπτυξη συστηματος υποστηριξης κλινικων αποφασεων με χρηση μοντελων μηχανικης και βαθιας μαθησης, σε συνδυασμο με μεθοδους Επεξηγησιμης Τεχνητης Νοημοσυνης (XAI).

Το συστημα εστιαζει σε δυο διαγνωστικα πεδια:
1. Καρδιολογια: Προβλεψη πιθανοτητας καρδιοπαθειας απο κλινικα δεδομενα πινακα με χρηση Random Forest και Deep Neural Network (DNN), καθως και επεξηγηση αποφασεων με SHAP και LIME.
2. Δερματολογια: Ταξινομηση δερματικων αλλοιωσεων (καλοηθεις / κακοηθεις) απο εικονες με χρηση Convolutional Neural Network (CNN / MobileNetV2) και Vision Transformer (ViT-B/16), καθως και οπτικη ερμηνεια με Grad-CAM και Attention Rollout.

Ολα τα αποτελεσματα ενσωματωνονται σε μια διαδραστικη εφαρμογη κλινικης υποστηριξης (Streamlit) καθως και σε αυτονομο εργαλειο γραμμης εντολων (CLI).

---

## Δομη Εργου

```text
Thesis/
│
├── app/
│   └── app.py                        # Κεντρικη διαδραστικη εφαρμογη Streamlit
│
├── data/
│   ├── HDD/                          # Δεδομενα Cleveland Heart Disease
│   │   ├── heart.csv                 # Αρχικο συνολο δεδομενων
│   │   ├── data_preprocessing.py     # Script καθαρισμου και προεπεξεργασιας
│   │   ├── scaler.pkl                # Αποθηκευμενο StandardScaler object
│   │   ├── X_train_ready.csv         # Προεπεξεργασμενα δεδομενα εκπαιδευσης
│   │   ├── X_test_ready.csv          # Προεπεξεργασμενα δεδομενα ελεγχου
│   │   ├── y_train_ready.csv
│   │   └── y_test_ready.csv
│   │
│   └── HAM10000/                     # Δεδομενα δερματικων αλλοιωσεων
│       ├── HAM10000_metadata.csv     # Μεταδεδομενα ασθενων και διαγνωσεων
│       ├── image sort.py             # Script ταξινομησης εικονων σε κλασεις
│       └── skin_cancer_data/         # Φακελοι εικονων (benign / malignant)
│
├── saved_models/                     # Αποθηκευμενα εκπαιδευμενα μοντελα
│   ├── rf_model.pkl                  # Random Forest μοντελο
│   ├── dnn_model.keras               # Deep Neural Network (.keras)
│   ├── cnn_tl_skin_cancer.keras      # CNN MobileNetV2 (.keras)
│   └── vit_model_hf.h5               # Βαρη Vision Transformer (HuggingFace)
│
├── diagnosis/                        # Φακελος αποθηκευσης συνθετων διαγνωσεων
│
├── training_scripts/                 # Scripts εκπαιδευσης και αναλυσης
│   ├── Random Forest/
│   │   └── rf_model.py               # Εκπαιδευση και αξιολογηση Random Forest
│   ├── DNN/
│   │   └── dnn.py                    # Εκπαιδευση και αξιολογηση DNN
│   ├── CNN/
│   │   ├── cnn_imagenet.py           # Εκπαιδευση transfer learning CNN
│   │   └── cnn_imagenet_gradcam.py   # Παραγωγη Grad-CAM χαρτων
│   ├── ViT/
│   │   ├── vit_model_hf.py           # Fine-tuning του Vision Transformer
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
│   │   └── roc_curves/               # Καμπυλες ROC και συγκρισεις
│   └── xai/
│       ├── tabular/                  # SHAP waterfall/beeswarm και LIME plots
│       └── vision/                   # Grad-CAM και Attention Rollout plots
│
├── diagnose.py                       # CLI εργαλειο συνθετης διαγνωσης
├── cnn_worker.py                     # Subprocess worker για το CNN Grad-CAM
├── mappings.py                       # Λεξικα κλινικων ορων στα ελληνικα
├── requirements.txt                  # Λιστα εξαρτησεων Python
└── README.txt                       # Τεκμηριωση εργου
```

---

## Εγκατασταση

2. Δημιουργια εικονικου περιβαλλοντος (venv):
```bash
python3 -m venv .venv
```

3. Ενεργοποιηση του περιβαλλοντος:
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

4. Εγκατασταση των dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Οδηγιες Χρησης Εφαρμογης

Η εφαρμογη κλινικης αποφασης εκκινειται με την εντολη:

```bash
streamlit run app/app.py
```

Μολις ανοιξει η εφαρμογη στον browser (http://localhost:8501), ειναι διαθεσιμες δυο λειτουργιες:

1. Καρτελα Δερματολογιας:
   - Ανεβασμα εικονας δερματικης βλαβης (.jpg, .png).
   - Επιλογη επιπεδου διαφανειας (alpha) για το Grad-CAM overlay.
   - Πατημα του κουμπιου διαγνωσης.
   - Εμφανιση πιθανοτητων κακοηθειας και ενδειξης συμφωνιας (Consensus Badge).
   - Ταυτοχρονη προβολη: Αρχικη Εικονα | Χαρτης Grad-CAM (CNN) | Χαρτης Attention Rollout (ViT).
   - Δυνατοτητα αποθηκευσης της συνολικης διαγνωστικης καρτελας.

2. Καρτελα Καρδιολογιας:
   - Εισαγωγη των 13 κλινικων παραμετρων του ασθενους μεσω φορμας.
   - Πατημα του κουμπιου εκτιμησης κινδυνου.
   - Εμφανιση προβλεψης και πιθανοτητας απο το Random Forest και το DNN.
   - Παραγωγη επεξηγησεων SHAP Waterfall και LIME bar chart για τον συγκεκριμενο ασθενη.

---

## Αυτονομη Διαγνωση απο Γραμμη Εντολων

Μπορειτε να εκτελεσετε το διαγνωστικο pipeline δερματολογιας απευθειας απο το τερματικο χωρις Streamlit:

```bash
python diagnose.py --image διαδρομη/προς/εικονα.jpg
```

Προαιρετικες παραμετροι:
- `--alpha 0.5`: Ρυθμιση διαφανειας Grad-CAM overlay.
- `--output_dir diagnosis/`: Φακελος αποθηκευσης της τελικης εικονας.

Η συνθετη εικονα αποτελεσματος αποθηκευεται αυτοματα με timestamp στον φακελο `diagnosis/`.

---

## Εκπαιδευση Μοντελων και Αναπαραγωγη Αποτελεσματων

Ολα τα scripts εκτελουνται απο τον κεντρικο φακελο του εργου (`Thesis/`):

1. Προεπεξεργασια Δεδομενων Καρδιολογιας:
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
python "training_scripts/ViT/vit_attention_rollout_hf.py"
```

4. Διερευνητικη Αναλυση Δεδομενων (EDA):
```bash
python outputs/eda/class_distribution.py
python outputs/eda/7class_distribution.py
python outputs/eda/correlation_heatmap.py
```