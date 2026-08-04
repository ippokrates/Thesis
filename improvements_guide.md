# Improvements Guide — RF, DNN, SHAP & LIME

Each section below explains **what we have now**, **what changes**, **why**, and gives a **ready-to-use code snippet**.

---

## 🔴 1. Save the Scaler for the Streamlit App

### What we did
In [data_preprocessing.py](file:///home/ippo/Desktop/Thesis/data/HDD/data_preprocessing.py) we fit a `StandardScaler` on the training data and then saved the scaled CSVs — but we **never saved the scaler object itself**.

### Why it matters
When the Streamlit app receives new patient data (raw values like `age=52, chol=212`), it needs to apply **the exact same scaling** that the models were trained on. Without the saved scaler, you'd have to either re-run preprocessing or hardcode 13 means and 13 standard deviations — fragile and error-prone.

### What changes

In [data_preprocessing.py](file:///home/ippo/Desktop/Thesis/data/HDD/data_preprocessing.py), add one line after the scaler is fitted:

```python
import joblib

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# NEW — save the scaler so the Streamlit app can reuse it
joblib.dump(scaler, "scaler.pkl")
```

Then in `app.py` you'll load it with:
```python
scaler = joblib.load("data/HDD/scaler.pkl")
new_patient_scaled = scaler.transform(new_patient_df)
prediction = model.predict(new_patient_scaled)
```

---

## 🔴 2. Add EarlyStopping + Reproducibility Seeds to DNN

### What we did
In [dnn.py](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/dnn.py) we train for **exactly 50 epochs** every time, with no check on whether the model has stopped improving. We also don't set random seeds, so every run produces slightly different results.

### Why it matters
- **Without EarlyStopping**, the model might overfit after epoch 20 but keep training for 30 more epochs — the final saved model is *worse* than the best intermediate one.
- **Without seeds**, you can't reproduce your thesis results. An examiner running your code will get different numbers.

### What changes

```python
import numpy as np
import tensorflow as tf

# NEW — reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ... model definition stays the same ...

# NEW — stop early if val_loss doesn't improve for 10 epochs,
#        and automatically restore the best weights
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True  # keeps the best model, not the last
)

history = dnn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stop]   # NEW — pass the callback
)
```

Now if the model peaks at epoch 18 and val_loss starts rising, training stops at epoch 28 and the weights from epoch 18 are restored.

---

## 🟡 3. Cross-Validation for Random Forest

### What we did
In [rf_model.py](file:///home/ippo/Desktop/Thesis/training_scripts/Random%20Forest/rf_model.py) we did a **single 80/20 split** and reported one set of metrics.

### Why it matters
With only ~1025 patients, a single split is **noisy** — your reported accuracy could swing ±5% depending on which 205 patients happen to land in the test set. Cross-validation runs **5 different splits** and averages the results, giving you a much more reliable estimate. This is especially important in a thesis — it shows methodological rigour.

### What changes

Add this **after** the existing training code (keep the single-split results too):

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 5-fold cross-validation on the FULL dataset (before splitting)
X = pd.read_csv("data/HDD/X_train_ready.csv")  # or use the full X before split
y = pd.read_csv("data/HDD/y_train_ready.csv").values.ravel()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We care about multiple metrics in a medical context
for metric in ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']:
    scores = cross_val_score(rf_model, X, y, cv=cv, scoring=metric)
    print(f"{metric:>10}: {scores.mean():.3f} ± {scores.std():.3f}")
```

Example output:
```
  accuracy: 0.832 ± 0.024
        f1: 0.841 ± 0.021
    recall: 0.867 ± 0.035      ← "do we catch sick patients?"
 precision: 0.818 ± 0.029
   roc_auc: 0.901 ± 0.018
```

The `± 0.024` tells you (and the examiner) how stable your model really is.

---

## 🟡 4. Switch DNN SHAP to `DeepExplainer`

### What we did
In [shap.py](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/shap.py) we used `KernelExplainer` with a k-means summary of 50 background samples.

### Why it matters
- `KernelExplainer` is **model-agnostic** — it treats the DNN as a black box, perturbs inputs, and observes outputs. This is **very slow** and gives **approximate** SHAP values.
- `DeepExplainer` is designed specifically for neural networks — it backpropagates through the network to compute **exact** SHAP values in a fraction of the time.
- Using k-means with only 50 clusters on 820 samples can also **distort** the feature distributions.

### What changes

```python
# BEFORE (slow, approximate)
background = shap.kmeans(X_train, 50)
explainer = shap.KernelExplainer(dnn_predict, background)
X_test_sample = X_test.iloc[:100]
shap_values = explainer.shap_values(X_test_sample)

# AFTER (fast, exact for neural networks)
background = X_train.values[:100]  # use real samples, not k-means centroids
explainer = shap.DeepExplainer(dnn_model, background)
shap_values = explainer.shap_values(X_test.values)  # explain ALL 205 test patients, not just 100
```

> [!NOTE]
> If `DeepExplainer` throws compatibility errors with your TensorFlow version, fall back to `GradientExplainer` — it uses a similar approach but is more compatible:
> ```python
> explainer = shap.GradientExplainer(dnn_model, background)
> ```

---

## 🟡 5. LIME on Multiple Patients + on Random Forest

### What we did
In [lime.py](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/lime.py) we explained **one patient** (`patient_index = 0`) with **5 features**, and only for the DNN.

### Why it matters
- LIME gives **local** explanations — showing only one patient doesn't demonstrate that the explanations actually differ between patients. A healthy patient should have different top-features than a sick one.
- Only explaining the DNN means we miss the chance to ask: *"Does the RF agree with the DNN about **why** this patient is sick?"* — one of the most valuable questions in a comparative XAI thesis.

### What changes

**A) Explain multiple patients with different outcomes:**

```python
# Find one patient predicted healthy and one predicted sick
y_pred = (dnn_model.predict(X_test.values, verbose=0).flatten() > 0.5).astype(int)

healthy_idx = np.where(y_pred == 0)[0][0]   # first predicted-healthy patient
sick_idx    = np.where(y_pred == 1)[0][0]    # first predicted-sick patient

for idx, label in [(healthy_idx, "Healthy"), (sick_idx, "Heart Disease")]:
    print(f"\n--- LIME explanation for Patient {idx} (Predicted: {label}) ---")
    exp = explainer.explain_instance(
        data_row=X_test.iloc[idx].values,
        predict_fn=predict_fn,
        num_features=10  # show 10 features instead of 5
    )
    for rule in exp.as_list():
        print(f"  {rule[0]}: {rule[1]:+.4f}")
    exp.save_to_file(f"lime_patient_{idx}_{label.lower().replace(' ', '_')}.html")
```

**B) Add LIME to Random Forest too:**

```python
import joblib

rf_model = joblib.load("saved_models/rf_model.pkl")

# RF already outputs probabilities with predict_proba
def rf_predict_fn(data):
    return rf_model.predict_proba(data)

rf_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Healthy', 'Heart Disease'],
    mode='classification'
)

# Explain the SAME patients as the DNN for direct comparison
for idx, label in [(healthy_idx, "Healthy"), (sick_idx, "Heart Disease")]:
    exp = rf_explainer.explain_instance(
        data_row=X_test.iloc[idx].values,
        predict_fn=rf_predict_fn,
        num_features=10
    )
    print(f"\n--- RF LIME for Patient {idx} ({label}) ---")
    for rule in exp.as_list():
        print(f"  {rule[0]}: {rule[1]:+.4f}")
```

Now you can write in the thesis: *"For Patient X, both RF and DNN agreed that `thal` and `ca` were the top risk factors, but the DNN additionally relied heavily on `oldpeak`..."*

---

## 🟡 6. Side-by-Side SHAP Comparison (RF vs DNN)

### What we did
SHAP runs separately for each model in different scripts. There's no combined visualization.

### Why it matters
The whole point of the thesis is comparing models **and** their explanations. A side-by-side plot directly answers: *"Do both models learn the same clinical risk factors?"* This is arguably the strongest figure in the thesis.

### What changes

```python
import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

X_test = pd.read_csv("data/HDD/X_test_ready.csv")

# --- RF SHAP ---
rf_model = joblib.load("saved_models/rf_model.pkl")
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap = rf_explainer.shap_values(X_test)
# Handle shape: get class-1 values
if isinstance(rf_shap, list):
    rf_shap = rf_shap[1]
elif len(rf_shap.shape) == 3:
    rf_shap = rf_shap[:, :, 1]

# --- DNN SHAP ---
X_train = pd.read_csv("data/HDD/X_train_ready.csv")
dnn_model = load_model("saved_models/dnn_model.keras")
dnn_explainer = shap.DeepExplainer(dnn_model, X_train.values[:100])
dnn_shap = dnn_explainer.shap_values(X_test.values)
if isinstance(dnn_shap, list):
    dnn_shap = dnn_shap[0]

# --- Side-by-side plot ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

plt.sca(axes[0])
shap.summary_plot(rf_shap, X_test, show=False, plot_size=None)
axes[0].set_title("Random Forest — SHAP", fontsize=14)

plt.sca(axes[1])
shap.summary_plot(dnn_shap, X_test, show=False, plot_size=None)
axes[1].set_title("DNN — SHAP", fontsize=14)

plt.tight_layout()
plt.savefig("shap_rf_vs_dnn.png", dpi=150, bbox_inches='tight')
plt.show()
```

This produces a single figure with both models' feature importance side-by-side — perfect for Chapter 4 of the thesis.

---

## 🟢 7. Add SHAP Waterfall/Force Plots (Individual Patient)

### What we did
Both SHAP scripts only produce a **summary plot** (global feature importance across all patients).

### Why it matters
The summary plot answers *"which features matter **overall**?"* but not *"why was **this specific patient** diagnosed as sick?"*. A waterfall plot breaks down a single prediction step-by-step — the examiner sees the base probability nudged up by `thal`, up again by `ca`, down slightly by `age`, etc. It's the most intuitive XAI visualization for a medical audience.

### What changes

```python
# Pick one interesting patient (e.g., the first one predicted as sick)
patient_idx = sick_idx  # reuse from the LIME section

# Create an Explanation object for the waterfall plot
shap_explanation = shap.Explanation(
    values=rf_shap[patient_idx],
    base_values=rf_explainer.expected_value[1],  # baseline probability
    data=X_test.iloc[patient_idx].values,
    feature_names=X_test.columns.tolist()
)

# Waterfall — shows how each feature pushes prediction up or down
shap.plots.waterfall(shap_explanation, show=False)
plt.title(f"Patient {patient_idx} — Why Heart Disease?")
plt.tight_layout()
plt.savefig(f"shap_waterfall_patient_{patient_idx}.png", dpi=150, bbox_inches='tight')
plt.show()
```

The resulting figure reads like: *"Starting from a 46% base probability → `thal=3` pushes to 62% → `ca=2` pushes to 78% → ... → final prediction: 84% Heart Disease."*

---

## 🟢 8. Plot DNN Training Curves

### What we did
In [dnn.py](file:///home/ippo/Desktop/Thesis/training_scripts/DNN/dnn.py) we store `history = dnn_model.fit(...)` but **never plot it**.

### Why it matters
Training curves (loss & accuracy over epochs) visually prove whether the model **converged** properly or **overfitted**. If val_loss starts going up while train_loss keeps dropping, the model memorized the training data. This is a standard figure in any ML thesis — its absence would be noticed by an examiner.

### What changes

Add after `dnn_model.fit(...)`:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Accuracy curve
axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig("dnn_training_curves.png", dpi=150)
plt.show()
```

A healthy curve shows both lines converging together. A gap = overfitting. Both flat = underfitting.

---

## Quick Reference Table

| # | Change | File(s) | Lines of code |
|---|---|---|---|
| 1 | Save scaler | `data_preprocessing.py` | +2 |
| 2 | EarlyStopping + seeds | `dnn.py` | +6 |
| 3 | Cross-validation | `rf_model.py` | +8 |
| 4 | DeepExplainer | `DNN/shap.py` | ~3 changed |
| 5 | Multi-patient + RF LIME | `DNN/lime.py` + new | +25 |
| 6 | Side-by-side SHAP | New script | ~30 |
| 7 | Waterfall plots | `rf_shap.py` or new | +10 |
| 8 | Training curves | `dnn.py` | +15 |
