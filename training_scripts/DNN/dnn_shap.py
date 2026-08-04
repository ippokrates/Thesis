import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
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
dnn_model = load_model("saved_models/dnn_model.keras")

# DeepExplainer — designed for neural networks, faster and more accurate than KernelExplainer
# uses real training samples as background instead of k-means centroids
background = X_train.values[:100]

try:
    explainer = shap.DeepExplainer(dnn_model, background)
    print("Using DeepExplainer (exact SHAP for neural networks)")
except Exception as e:
    print(f"DeepExplainer failed ({e}), falling back to GradientExplainer...")
    explainer = shap.GradientExplainer(dnn_model, background)
    print("Using GradientExplainer (gradient-based SHAP for neural networks)")

# explain ALL test patients, not just 100
shap_values = explainer.shap_values(X_test.values)

# handle shape — DeepExplainer may return a list or a 3D array
if isinstance(shap_values, list):
    shap_values = shap_values[0]
if len(shap_values.shape) > 2:
    shap_values = shap_values.squeeze()

# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=readable_feature_names, show=False)
plt.title("DNN — SHAP Feature Importance (DeepExplainer)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dnn_shap_summary.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/dnn_shap_summary.png")
plt.close()

# save shap_values for reuse in B3/B4
np.save(f"{OUT_DIR}/dnn_shap_values.npy", shap_values)
print(f"Saved: {OUT_DIR}/dnn_shap_values.npy (for reuse in comparison scripts)")