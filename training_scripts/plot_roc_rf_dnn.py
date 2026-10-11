import pandas as pd, numpy as np, matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc

X_test = pd.read_csv("data/HDD/X_test_ready.csv")
y_test = pd.read_csv("data/HDD/y_test_ready.csv").values.ravel()

rf_prob = joblib.load("saved_models/rf_model.pkl").predict_proba(X_test)[:, 1]
dnn_prob = load_model("saved_models/dnn_model.keras").predict(X_test, verbose=0).flatten()

fig, ax = plt.subplots(figsize=(6, 5))
for y, name in [(rf_prob, "RF"), (dnn_prob, "DNN")]:
    fpr, tpr, _ = roc_curve(y_test, y)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})")
ax.plot([0, 1], [0, 1], "--", label="Chance")
ax.set(xlabel="False Positive Rate (1 - Specificity)",
       ylabel="True Positive Rate (Sensitivity / Recall)",
       title="ROC Curves - RF vs DNN")
ax.legend(); fig.tight_layout()
fig.savefig("outputs/evaluation/roc_curves/roc_curves_rf_vs_dnn.png", dpi=150)
print("saved roc_curves_rf_vs_dnn.png")

