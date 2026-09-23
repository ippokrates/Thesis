import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import shap
import matplotlib.pyplot as plt
import os
from PIL import Image

OUT_DIR = "outputs/xai/tabular/shap/"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

X_test = pd.read_csv("data/HDD/X_test_ready.csv")
y_test = pd.read_csv("data/HDD/y_test_ready.csv").values.ravel()

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

rf_model = joblib.load("saved_models/rf_model.pkl")
dnn_model = load_model("saved_models/dnn_model.keras")

dnn_preds = (dnn_model.predict(X_test.values, verbose=0).flatten() > 0.5).astype(int)
healthy_idx = np.where(dnn_preds == 0)[0][0]
sick_idx = np.where(dnn_preds == 1)[0][0]

print(f"Healthy patient: index {healthy_idx} (actual: {y_test[healthy_idx]})")
print(f"Sick patient:    index {sick_idx} (actual: {y_test[sick_idx]})")

# Try to load saved SHAP values
try:
    rf_shap = np.load(f"{OUT_DIR}/rf_shap_values.npy")
    dnn_shap = np.load(f"{OUT_DIR}/dnn_shap_values.npy")
    print("Loaded pre-computed SHAP values from B3")
except FileNotFoundError:
    print("Computing SHAP values from scratch...")
    # RF
    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap = rf_explainer.shap_values(X_test)
    if isinstance(rf_shap, list):
        rf_shap = rf_shap[1]
    elif len(rf_shap.shape) == 3:
        rf_shap = rf_shap[:, :, 1]

    # DNN
    X_train = pd.read_csv("data/HDD/X_train_ready.csv")
    background = X_train.values[:100]
    try:
        dnn_explainer = shap.DeepExplainer(dnn_model, background)
    except Exception:
        dnn_explainer = shap.GradientExplainer(dnn_model, background)
    dnn_shap = dnn_explainer.shap_values(X_test.values)
    if isinstance(dnn_shap, list):
        dnn_shap = dnn_shap[0]

# squeeze extra dimensions (DeepExplainer may return shape (n, 13, 1))
if len(dnn_shap.shape) > 2:
    dnn_shap = dnn_shap.squeeze()

rf_explainer = shap.TreeExplainer(rf_model)
rf_base = rf_explainer.expected_value
if isinstance(rf_base, (list, np.ndarray)):
    rf_base = rf_base[1]  # class 1 (Heart Disease)

# DNN base value = mean prediction on background
X_train = pd.read_csv("data/HDD/X_train_ready.csv")
dnn_base = dnn_model.predict(X_train.values[:100], verbose=0).flatten().mean()

feature_names = readable_feature_names

scaler = joblib.load("data/HDD/scaler.pkl")
X_test_unscaled = scaler.inverse_transform(X_test)

# Waterfall plots
for idx, label in [(healthy_idx, "Healthy"), (sick_idx, "Heart_Disease")]:
    # RF waterfall
    rf_explanation = shap.Explanation(
        values=rf_shap[idx],
        base_values=rf_base,
        data=np.round(X_test_unscaled[idx], 2),
        feature_names=feature_names
    )
    plt.figure()
    shap.plots.waterfall(rf_explanation, show=False)
    plt.title(f"Random Forest - Patient {idx} ({label})", fontsize=14, fontweight='bold', pad=20)
    rf_tmp = f"{OUT_DIR}/tmp_rf_{idx}.png"
    plt.savefig(rf_tmp, dpi=150, bbox_inches='tight')
    plt.close()

    # DNN waterfall
    dnn_explanation = shap.Explanation(
        values=dnn_shap[idx],
        base_values=dnn_base,
        data=np.round(X_test_unscaled[idx], 2),
        feature_names=feature_names
    )
    plt.figure()
    shap.plots.waterfall(dnn_explanation, show=False)
    plt.title(f"DNN - Patient {idx} ({label})", fontsize=14, fontweight='bold', pad=20)
    dnn_tmp = f"{OUT_DIR}/tmp_dnn_{idx}.png"
    plt.savefig(dnn_tmp, dpi=150, bbox_inches='tight')
    plt.close()

    # Combine images side by side
    img1 = Image.open(rf_tmp)
    img2 = Image.open(dnn_tmp)
    
    padding = 40
    new_width = img1.width + img2.width + padding
    new_height = max(img1.height, img2.height)
    
    combined = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    
    # Calculate vertical offsets to center them if heights differ
    y_offset1 = (new_height - img1.height) // 2
    y_offset2 = (new_height - img2.height) // 2
    
    combined.paste(img1, (0, y_offset1))
    combined.paste(img2, (img1.width + padding, y_offset2))

    filename = f"{OUT_DIR}/shap_waterfall_patient_{idx}_{label.lower()}.png"
    combined.save(filename)
    
    os.remove(rf_tmp)
    os.remove(dnn_tmp)
    
    print(f"Saved: {filename}")
    plt.close()

print("\nDone! Waterfall plots saved for both patients.")
