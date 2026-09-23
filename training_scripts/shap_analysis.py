import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

X_train = pd.read_csv("data/HDD/X_train_ready.csv")
X_test = pd.read_csv("data/HDD/X_test_ready.csv")
feature_names = X_train.columns.tolist()

rf_model = joblib.load("saved_models/rf_model.pkl")
dnn_model = load_model("saved_models/dnn_model.keras")

print("Computing SHAP values for RF")
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_test)

# Handle both possible shap formats (list / 3D array)
if isinstance(rf_shap_values, list):
    rf_shap_values = rf_shap_values[1]          # class 1 = heart disease
elif rf_shap_values.ndim == 3:
    rf_shap_values = rf_shap_values[:, :, 1]

print("Computing SHAP values for DNN")
background = shap.sample(X_train, 50, random_state=42)   # summarize background for speed
dnn_explainer = shap.KernelExplainer(dnn_model.predict, background)
dnn_shap_values = dnn_explainer.shap_values(X_test, nsamples=100)

if isinstance(dnn_shap_values, list):
    dnn_shap_values = dnn_shap_values[0]
if dnn_shap_values.ndim == 3:
    dnn_shap_values = dnn_shap_values[:, :, 0]

# Save raw SHAP values for reuse
#np.save("shap_values_rf.npy", rf_shap_values)
#np.save("shap_values_dnn.npy", dnn_shap_values)

# Beeswarm plots
plt.figure()
shap.summary_plot(rf_shap_values, X_test, feature_names=feature_names, show=False)
plt.title("Random Forest - SHAP Feature Importance")
plt.tight_layout()
plt.savefig("outputs/xai/tabular/shap/shap_beeswarm_rf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_beeswarm_rf.png")

plt.figure()
shap.summary_plot(dnn_shap_values, X_test, feature_names=feature_names, show=False)
plt.title("DNN - SHAP Feature Importance")
plt.tight_layout()
plt.savefig("outputs/xai/tabular/shap/shap_beeswarm_dnn.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_beeswarm_dnn.png")
