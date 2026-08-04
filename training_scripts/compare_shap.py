import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import shap
import matplotlib.pyplot as plt
import os

OUT_DIR = "training_scripts/xai_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

X_train = pd.read_csv("data/HDD/X_train_ready.csv")
X_test = pd.read_csv("data/HDD/X_test_ready.csv")

feature_mapping = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
    'restecg': 'Resting ECG Results',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression (Oldpeak)',
    'slope': 'Slope of ST Segment',
    'ca': 'Number of Major Vessels (0-3)',
    'thal': 'Thalassemia'
}
readable_feature_names = [feature_mapping.get(col, col) for col in X_test.columns]

# --- RF SHAP ---
print("Computing RF SHAP values...")
rf_model = joblib.load("saved_models/rf_model.pkl")
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap = rf_explainer.shap_values(X_test)

# handle shape: get class-1 (Heart Disease) values
if isinstance(rf_shap, list):
    rf_shap = rf_shap[1]
elif len(rf_shap.shape) == 3:
    rf_shap = rf_shap[:, :, 1]

# save for reuse in B4
np.save(f"{OUT_DIR}/rf_shap_values.npy", rf_shap)

# --- DNN SHAP ---
print("Computing DNN SHAP values...")
dnn_model = load_model("saved_models/dnn_model.keras")
background = X_train.values[:100]

try:
    dnn_explainer = shap.DeepExplainer(dnn_model, background)
    print("  Using DeepExplainer")
except Exception:
    dnn_explainer = shap.GradientExplainer(dnn_model, background)
    print("  Using GradientExplainer (fallback)")

dnn_shap = dnn_explainer.shap_values(X_test.values)
if isinstance(dnn_shap, list):
    dnn_shap = dnn_shap[0]

# DeepExplainer can return shape (n, 13, 1). Squeeze it to (n, 13) for correct plotting.
if len(dnn_shap.shape) > 2:
    dnn_shap = dnn_shap.squeeze()

# save for reuse in B4
np.save(f"{OUT_DIR}/dnn_shap_values.npy", dnn_shap)

# --- Individual summary plots ---
print("Generating SHAP summary plots...")

plt.figure(figsize=(10, 7))
shap.summary_plot(rf_shap, X_test, feature_names=readable_feature_names, show=False)
plt.title("Random Forest — SHAP Feature Importance", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("rf_shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: rf_shap_summary.png")

plt.figure(figsize=(10, 7))
shap.summary_plot(dnn_shap, X_test, feature_names=readable_feature_names, show=False)
plt.title("DNN — SHAP Feature Importance", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("dnn_shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: dnn_shap_summary.png")

# --- Individual bar plots ---
plt.figure(figsize=(10, 7))
shap.summary_plot(rf_shap, X_test, feature_names=readable_feature_names, plot_type="bar", show=False)
plt.title("Random Forest — Mean |SHAP|", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("rf_shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: rf_shap_bar.png")

plt.figure(figsize=(10, 7))
shap.summary_plot(dnn_shap, X_test, feature_names=readable_feature_names, plot_type="bar", show=False)
plt.title("DNN — Mean |SHAP|", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("dnn_shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: dnn_shap_bar.png")

# --- Combine into side-by-side images using PIL ---
from PIL import Image

for name, left_f, right_f in [
    ("shap_rf_vs_dnn.png", "rf_shap_summary.png", "dnn_shap_summary.png"),
    ("shap_rf_vs_dnn_bar.png", "rf_shap_bar.png", "dnn_shap_bar.png"),
]:
    left = Image.open(left_f)
    right = Image.open(right_f)
    # resize to same height
    h = min(left.height, right.height)
    left = left.resize((int(left.width * h / left.height), h))
    right = right.resize((int(right.width * h / right.height), h))
    combined = Image.new('RGB', (left.width + right.width, h), (255, 255, 255))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    combined.save(f"{OUT_DIR}/{name}", dpi=(150, 150))
    print(f"Saved combined: {OUT_DIR}/{name}")

# cleanup temp files
import os
for f in ["rf_shap_summary.png", "dnn_shap_summary.png", "rf_shap_bar.png", "dnn_shap_bar.png"]:
    os.remove(f)

print("\nDone! Both comparison plots saved.")
