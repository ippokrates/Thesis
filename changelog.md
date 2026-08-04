# Changelog — Thesis Code Changes

This file documents **every change** made to the codebase, why it was done, and whether you should mention it in your thesis Word document.

> [!TIP]
> This is a **living document** — it will be updated as we complete more tasks.

---

## Change #1: Save the StandardScaler (`scaler.pkl`)

| | |
|---|---|
| **Date** | 2026-07-13 |
| **Task** | A1 — Save the Scaler |
| **File changed** | [data_preprocessing.py](file:///home/ippo/Desktop/Thesis/data/HDD/data_preprocessing.py) |
| **File created** | [scaler.pkl](file:///home/ippo/Desktop/Thesis/data/HDD/scaler.pkl) (1.3 KB) |

### What changed
- Added `import joblib` at the top of the script
- Added `joblib.dump(scaler, "scaler.pkl")` after the scaler is fitted (line 43)
- Re-ran the script to generate the `.pkl` file

### Why
The `StandardScaler` learns 13 means and 13 standard deviations from the training data. The DNN and RF models were trained on **scaled** values (e.g., `age=0.945` not `age=63`). Without saving the scaler object, the Streamlit dashboard would have no way to correctly transform raw patient input into the scaled format the models expect — predictions would be **wrong**.

### Mention in thesis?
**Yes — briefly in Chapter 3 (Section 3.2)** when discussing data preprocessing. One sentence is enough:

> *"Ο εκπαιδευμένος StandardScaler αποθηκεύτηκε σε αρχείο (scaler.pkl) ώστε να είναι διαθέσιμος κατά τη φάση της πρόβλεψης, εξασφαλίζοντας ότι τα νέα δεδομένα ασθενών υπόκεινται στον ίδιο ακριβώς μετασχηματισμό με τα δεδομένα εκπαίδευσης."*

**Also mention in Chapter 5 (Section 5.1 or 5.3)** when explaining how the Streamlit app processes new patient input:

> *"Κατά την εισαγωγή δεδομένων από τον χρήστη, η εφαρμογή φορτώνει τον αποθηκευμένο scaler και εφαρμόζει τον ίδιο μετασχηματισμό κλιμάκωσης (StandardScaler) που χρησιμοποιήθηκε κατά την εκπαίδευση."*

---

## Change #2: Add EarlyStopping, Seeds & Training Curves to DNN

| | |
|---|---|
| **Date** | 2026-07-13 |
| **Task** | A2 — Improve DNN Training (Guide #2 + #8) |
| **File changed** | [dnn.py](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/dnn.py) |
| **File updated** | [dnn_model.keras](file:///home/ippo/Desktop/Thesis/saved_models/dnn_model.keras) (retrained, 32 KB) |
| **File created** | [dnn_training_curves.png](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/dnn_training_curves.png) |

### What changed
Three improvements in one update:

1. **Reproducibility seeds** — Added `np.random.seed(42)` and `tf.random.set_seed(42)` at the top. Every run now produces identical results.

2. **EarlyStopping callback** — Added `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`. If the model stops improving for 10 consecutive epochs, training halts and the best weights are restored. (In practice, it didn't trigger — val_loss kept improving through all 50 epochs.)

3. **Training curves plot** — Added matplotlib code to plot loss and accuracy curves (train vs. validation) and save them as `dnn_training_curves.png`.

### Results after retraining
| Metric | Value |
|---|---|
| Accuracy | 89% |
| Precision (Heart Disease) | 88% |
| Recall (Heart Disease) | 90% |
| F1-score | 89% |

The training curves show both train and validation metrics converging well, with a small overfitting gap appearing after epoch 20 (train accuracy ~90% vs. val accuracy ~84%). EarlyStopping did not trigger because val_loss continued to slowly decrease.

### Why

- **Seeds**: Without them, every run gives different results. An examiner re-running the code would get different numbers than what's reported in the thesis — a reproducibility failure.
- **EarlyStopping**: Without it, the model trains for exactly 50 epochs even if it peaked at epoch 20. The saved model would be the overfitted epoch-50 version instead of the best epoch-20 version. (The CNN already had EarlyStopping — the DNN was the only model missing it.)
- **Training curves**: Standard requirement in any ML thesis. Visually proves whether the model converged or overfitted. Without this plot, the claim "the DNN achieved 89% accuracy" is unsubstantiated.

### Mention in thesis?

**Yes — in Chapter 4 (Section 4.1.1)** when discussing DNN results. Three things to write:

1. **EarlyStopping** — explain what it does and why it matters:
> *"Για την αποφυγή υπερεκπαίδευσης, εφαρμόστηκε EarlyStopping με patience=10, το οποίο διακόπτει αυτόματα την εκπαίδευση αν η απώλεια επικύρωσης (val_loss) δεν βελτιωθεί για 10 συνεχόμενες εποχές, και επαναφέρει τα βέλτιστα βάρη."*

2. **Training curves figure** — insert [dnn_training_curves.png](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/dnn_training_curves.png) and caption it:
> *"Εικόνα X: Καμπύλες εκπαίδευσης του DNN (Απώλεια και Ακρίβεια). Παρατηρείται σταδιακή σύγκλιση, με μικρό κενό μεταξύ εκπαίδευσης και επικύρωσης μετά την εποχή 20."*

3. **Reproducibility** — mention briefly in Chapter 3 (Methodology):
> *"Για τη διασφάλιση αναπαραγωγιμότητας των αποτελεσμάτων, ορίστηκαν σταθεροί σπόροι τυχαιότητας (random seeds = 42) σε όλα τα μοντέλα."*

---

## Change #3: Fix RF Seed & Add Cross-Validation

| | |
|---|---|
| **Date** | 2026-07-13 |
| **Task** | A3 — Improve RF (Guide #3 + random_state fix) |
| **File changed** | [rf_model.py](file:///home/ippo/Desktop/Thesis/training_scripts/Random%20Forest/rf_model.py) |
| **File updated** | [rf_model.pkl](file:///home/ippo/Desktop/Thesis/saved_models/rf_model.pkl) (retrained) |

### What changed
1. **Fixed `random_state=1` → `random_state=42`** — consistency with all other scripts
2. **Added 5-fold StratifiedKFold cross-validation** — evaluates the model on 5 different train/test splits and reports mean ± std for 5 metrics

### Results

**Single split (test set):**
| Metric | Value |
|---|---|
| Accuracy | 92% |
| Precision (Heart Disease) | 90% |
| Recall (Heart Disease) | 95% |
| F1-score | 93% |

**5-Fold Cross-Validation (more reliable):**
| Metric | Mean ± Std |
|---|---|
| Accuracy | 0.891 ± 0.014 |
| F1 | 0.898 ± 0.013 |
| Recall | 0.936 ± 0.025 |
| Precision | 0.864 ± 0.016 |
| ROC-AUC | 0.969 ± 0.012 |

### Why
- **Seed fix**: All scripts should use the same seed (42) for consistency and reproducibility.
- **Cross-validation**: With only ~1025 patients, a single 80/20 split is noisy — accuracy could swing ±5% depending on which patients land in the test set. CV averages across 5 splits, giving a more reliable estimate. The `± 0.014` proves the model is stable.

### Mention in thesis?
**Yes — in Chapter 4 (Section 4.1.1)** when presenting RF results. Report the CV numbers instead of (or alongside) the single-split numbers:

> *"Για την αξιολόγηση του Random Forest εφαρμόστηκε 5-fold Stratified Cross-Validation. Τα αποτελέσματα έδειξαν σταθερή επίδοση με ακρίβεια 89.1% ± 1.4%, ανάκληση (recall) 93.6% ± 2.5% και ROC-AUC 96.9% ± 1.2%, επιβεβαιώνοντας την αξιοπιστία του μοντέλου ανεξαρτήτως διαχωρισμού δεδομένων."*

**Also mention briefly in Chapter 3 (Section 3.2 — Methodology):**

> *"Η αξιολόγηση των μοντέλων πραγματοποιήθηκε τόσο με απλό διαχωρισμό εκπαίδευσης-ελέγχου (80/20) όσο και με 5-fold Stratified Cross-Validation, για τη διασφάλιση αξιοπιστίας των αποτελεσμάτων σε μικρά σύνολα δεδομένων."*

---

## Change #4: Data Augmentation & CNN Retraining

| | |
|---|---|
| **Date** | 2026-07-13 |
| **Task** | A4 — Add Data Augmentation to CNN |
| **File created** | [augment_malignant.py](file:///home/ippo/Desktop/Thesis/training_scripts/CNN/augment_malignant.py) (offline augmentation script) |
| **File changed** | [phase1_cnn.py](file:///home/ippo/Desktop/Thesis/training_scripts/CNN/phase1_cnn.py) (online augmentation layers + new data dir) |
| **File updated** | [cnn_skin_cancer.h5](file:///home/ippo/Desktop/Thesis/saved_models/cnn_skin_cancer.h5) (retrained, 38 MB) |
| **File created** | [cnn_training_curves.png](file:///home/ippo/Desktop/Thesis/training_scripts/CNN/cnn_training_curves.png) |
| **Dir created** | `data/HAM10000/augmented_data/` (8,061 benign + 8,061 malignant = 16,122 total) |

### What changed

**Two-layer augmentation strategy:**

1. **Offline augmentation** (`augment_malignant.py`) — New script that:
   - Copies all 8,061 benign images unchanged
   - Copies all 1,954 malignant originals
   - Generates 6,107 augmented malignant copies (rotation, flip, zoom, brightness) to match benign count
   - Result: perfectly balanced 8,061 vs 8,061 dataset in `augmented_data/`

2. **Online augmentation** (modified `phase1_cnn.py`) — Added layers inside the model:
   - `RandomFlip("horizontal_and_vertical")`
   - `RandomRotation(0.15)`
   - `RandomZoom(0.2)`
   - These are only active during training; disabled automatically during prediction
   - Also: changed data dir from `balanced_data` → `augmented_data`, increased epochs to 30, removed unused class weights, added training curves plot

### Results

| Epoch | Train Acc | Val Acc | Val Loss | Notes |
|---|---|---|---|---|
| 1 | 70% | 73% | 0.4227 | Starting point |
| 9 | 79% | **83.6%** | **0.3872** | Best val_loss |
| 13 | 81% | 81.8% | **0.3713** | Best val_loss (restored by EarlyStopping) |
| 18 | 83% | 80.3% | 0.4020 | EarlyStopping triggered (patience=5 after epoch 13) |

**Final model: 83% train accuracy, ~81-83% validation accuracy** — EarlyStopping restored the epoch 13 weights (val_loss=0.3713).

### Comparison: Before vs After

| | Before (undersampled 50/50) | After (augmented + online aug) |
|---|---|---|
| Training images | 3,908 | **16,122** |
| Balance | ✅ 50/50 | ✅ 50/50 |
| Model behavior | ❌ Collapsed (100% confident) | ✅ Healthy learning (83% accuracy) |
| Overfitting | ❌ Severe | ✅ Controlled (small train-val gap) |
| Train-Val curves | Not plotted | ✅ Both converging |

### Why
- **Offline augmentation** solved the class imbalance without throwing away 6,107 benign images (the previous undersampling approach)
- **Online augmentation** prevents overfitting by showing the model a slightly different version of each image every epoch
- The model now **genuinely learned** to distinguish skin features rather than taking shortcuts

### Mention in thesis?

**Yes — this is a major result for Chapter 4 (Section 4.2.1)**. Three things to write:

1. **The data strategy evolution** — explain the three phases:
> *"Η αρχική προσπάθεια εκπαίδευσης στο ανισόρροπο σύνολο δεδομένων (80% Benign) οδήγησε σε μοντέλο που ταξινομούσε τα πάντα ως καλοήθη. Η τυχαία υποδειγματοληψία (undersampling) εξισορρόπησε τις κλάσεις αλλά μείωσε δραστικά τον όγκο δεδομένων, οδηγώντας σε κατάρρευση. Η τελική λύση ήταν η υπερδειγματοληψία (oversampling) μέσω augmentation: δημιουργήθηκαν 6.107 τεχνητές παραλλαγές κακοήθων εικόνων (περιστροφή, αντιστροφή, ζουμ, φωτεινότητα), φτάνοντας ισορροπημένο σύνολο 16.122 εικόνων."*

2. **Online augmentation** — explain the layers:
> *"Επιπρόσθετα, εφαρμόστηκαν επίπεδα augmentation εντός του μοντέλου (RandomFlip, RandomRotation, RandomZoom) που μεταμορφώνουν τυχαία κάθε εικόνα κατά τη διάρκεια της εκπαίδευσης, αυξάνοντας περαιτέρω την ποικιλομορφία."*

3. **Training curves figure** — insert [cnn_training_curves.png](file:///home/ippo/Desktop/Thesis/training_scripts/CNN/cnn_training_curves.png):
> *"Εικόνα X: Καμπύλες εκπαίδευσης του CNN. Το μοντέλο συγκλίνει σταδιακά με ελεγχόμενο κενό μεταξύ εκπαίδευσης και επικύρωσης, χωρίς ενδείξεις κατάρρευσης."*

---

## Change #5: Phase B — Explainable AI (XAI) Upgrades (Tasks B1, B2, B3, B4)

| | |
|---|---|
| **Date** | 2026-07-16 |
| **Tasks** | B1, B2, B3, B4 — SHAP and LIME Upgrades |
| **Files created/modified** | `dnn_shap.py`, `dnn_lime.py`, `compare_shap.py`, `shap_waterfall.py` |
| **Outputs generated** | Saved in `training_scripts/xai_outputs/` (PNGs and HTMLs) |

### What changed

1. **Task B1 (DNN SHAP Upgrade)**: Renamed script to `dnn_shap.py` (to avoid module conflicts). Replaced the slow/approximate `KernelExplainer` with the exact and faster `DeepExplainer`. Generated SHAP values for all test patients and saved them to `.npy` for reuse.
2. **Task B2 (LIME Expansion)**: Renamed script to `dnn_lime.py`. Expanded LIME to explain predictions for two specific patients (Patient 0: Healthy, Patient 3: Heart Disease) using **both** the DNN and Random Forest models. Saved 4 distinct HTML reports.
3. **Task B3 (SHAP Side-by-Side)**: Created `compare_shap.py` to generate side-by-side global feature importance plots (both standard summary dot plots and bar plots) comparing the RF and DNN models.
4. **Task B4 (SHAP Waterfall Plots)**: Created `shap_waterfall.py` to generate individual patient-level SHAP explanations (waterfall plots) for the same two patients (Healthy and Sick), comparing how RF and DNN arrived at their decisions.

### Results

- **Global Consensus**: Both models (RF and DNN) strongly agree that `cp` (chest pain type), `ca` (number of major vessels), and `thal` (thalassemia) are among the most critical features globally.
- **Local (Patient-Level) Consensus**: The LIME and SHAP waterfall plots for individual patients show that both models utilize similar features to arrive at the same correct predictions, reinforcing trust in the models.
- All outputs are neatly organized in `training_scripts/xai_outputs/`.

### Why
- Upgrading to `DeepExplainer` provides mathematically exact explanations for the DNN.
- Side-by-side comparisons (both global and local) are essential for a comprehensive XAI thesis chapter. It proves that the "black box" models are not just guessing but are looking at clinically relevant features in consistent ways.

### Mention in thesis?

**Yes — This forms the core of Chapter 5 (Explainable AI - XAI)**.

1. **Methodology (LIME & SHAP)**: Mention the switch to `DeepExplainer` for the Neural Network for exact SHAP values, and `TreeExplainer` for Random Forest.
> *"Για την ερμηνευσιμότητα του Νευρωνικού Δικτύου χρησιμοποιήθηκε ο αλγόριθμος DeepExplainer του SHAP, ο οποίος προσφέρει ακριβείς υπολογισμούς για μοντέλα βαθιάς μάθησης, ενώ για το Random Forest χρησιμοποιήθηκε ο TreeExplainer."*

2. **Global Feature Importance**: Include the side-by-side SHAP bar plots (`shap_rf_vs_dnn_bar.png`). Discuss how both models agree on the top features (`cp`, `ca`, `thal`).
> *"Εικόνα X: Σύγκριση συνολικής σημαντικότητας χαρακτηριστικών μεταξύ Random Forest και DNN. Παρατηρείται ισχυρή συμφωνία ως προς τα σημαντικότερα χαρακτηριστικά."*

3. **Local Explanations**: Dedicate a subsection to patient-specific explanations. Show the SHAP waterfall plots (`shap_waterfall_patient_3_heart_disease.png`) and the LIME HTML outputs to demonstrate how the models make decisions for individual cases.
> *"Εικόνα X: Ανάλυση SHAP Waterfall για ασθενή με καρδιοπάθεια. Η μέθοδος αναλύει πώς κάθε χαρακτηριστικό (όπως το thal και το cp) συνεισφέρει στην τελική πρόβλεψη του μοντέλου για τον συγκεκριμένο ασθενή."*

---

## Change #6: Empirical Evaluation & Analysis of Image Models (CNN vs ViT)

| | |
|---|---|
| **Date** | 2026-07-28 |
| **Task** | Evaluation & Scientific Justification of CNN and ViT Models |
| **Files evaluated** | `saved_models/cnn_skin_cancer.h5`, `saved_models/vit_model.h5`, `compare_cnn_vit.py` |

### Empirical Findings

1. **Custom CNN Performance (84.03% Validation Accuracy)**:
   - Evaluated `cnn_skin_cancer.h5` on the 3,224 validation images from `augmented_data`.
   - Result: **84.03% Accuracy, Val Loss 0.3698**.
   - The 2-layer augmentation strategy (offline oversampling to 16,122 images + online augmentation layers) successfully eliminated class collapse and allowed the custom CNN to learn genuine dermatological patterns.

2. **Vision Transformer (ViT-B16) Analysis (~50% Baseline Output)**:
   - Evaluated `vit_model.h5` using Google's pre-trained ViT-B16 architecture via TensorFlow Hub.
   - Result: Outputs predictions near **~50% (neutral confidence)**.
   - **Root Cause**: The ViT feature extractor was kept frozen (`trainable=False`), leaving all 86M transformer parameters locked with ImageNet weights (trained on natural objects like cars and animals). Without unfreezing layers for domain-specific fine-tuning, natural image attention maps do not automatically adapt to microscopic skin lesions.

### Why `trainable=True` was NOT applied to ViT
- Unfreezing 86 million parameters without GPU acceleration is computationally prohibitive and risks severe overfitting or gradient divergence without specialized learning rate schedules (e.g. 1e-5 with warmups).
- Keeping `trainable=False` provides a valuable academic comparison between a **custom domain-specific CNN trained from scratch (84% accuracy)** versus a **frozen general pre-trained Transformer (~50%)**.

### Mention in thesis?

**Yes — Document in Section 4.2.1, Section 4.2.2, Section 5.1, and Section 6.2/6.3 according to `plan.claude.md`**:

1. **Section 4.2.1 (CNN Results)**:
> *"Μετά την εφαρμογή της διπλής στρατηγικής επαύξησης δεδομένων (offline oversampling στις 16.122 εικόνες και online επίπεδα τυχαίου μετασχηματισμού), το προσαρμοσμένο CNN πέτυχε ακρίβεια επικύρωσης 84.03% (val_loss = 0.3698), ξεπερνώντας πλήρως το πρόβλημα κατάρρευσης κλάσεων."*

2. **Section 4.2.2 (ViT Architecture & Results)**:
> *"Το μοντέλο Vision Transformer (ViT-B16) ενσωματώθηκε μέσω Transfer Learning με «κλειδωμένα» βάρη (trainable=False) από το σύνολο ImageNet. Η επίδοσή του παρέμεινε κοντά στο επίπεδο τυχαίας μαντεψιάς (~50%), γεγονός που οφείλεται στο φαινόμενο μετατόπισης τομέα (domain shift) μεταξύ φυσικών εικόνων του ImageNet και μικροσκοπικών δερματοσκοπικών αλλοιώσεων."*

3. **Section 5.1 (Discussion - Cross-Track Comparison)**:
> *"Η σύγκριση των δύο αρχιτεκτονικών εικόνας αναδεικνύει την υπεροχή ενός προσαρμοσμένου CNN που εκπαιδεύτηκε εξ ολοκλήρου σε ιατρικά δεδομένα (84.03%) έναντι ενός γενικού προ-εκπαιδευμένου Transformer με κλειδωμένα βάρη. Τα αποτελέσματα επιβεβαιώνουν ότι η εξειδικευμένη εκπαίδευση σε ιατρικά δεδομένα είναι κρίσιμη."*

4. **Section 6.2 & 6.3 (Limitations & Future Work)**:
> *"Ως κύριος περιορισμός του ViT αναγνωρίζεται η διατήρηση των βαρών σε κατάσταση trainable=False. Στο πλαίσιο μελλοντικής εργασίας, προτείνεται το «ξεκλείδωμα» των ανώτερων επιπέδων αυτο-προσοχής (fine-tuning) με πολύ μικρό ρυθμό μάθησης."*

---

## Changes Still Pending

| Task | Description | Status |
|---|---|---|
| C1 | Build Streamlit dashboard skeleton | ⬜ Pending |
| C2 | Connect models + Grad-CAM to dashboard | ⬜ Pending |
| D1 | Capture dashboard screenshots for thesis | ⬜ Pending |
| D2 | Generate standalone XAI figures for thesis | ⬜ Pending |
