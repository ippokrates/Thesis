import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import lime
import lime.lime_tabular
import os

OUT_DIR = "outputs/xai/tabular/lime/"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

X_train = pd.read_csv("data/HDD/X_train_ready.csv")
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
readable_feature_names = [feature_mapping.get(col, col) for col in X_train.columns]

dnn_model = load_model("saved_models/dnn_model.keras")
rf_model = joblib.load("saved_models/rf_model.pkl")

# LIME requires [P(class0), P(class1)] for each sample
def dnn_predict_fn(data):
    preds = dnn_model.predict(data, verbose=0).flatten()
    return np.vstack((1 - preds, preds)).T

def rf_predict_fn(data):
    return rf_model.predict_proba(data)

# Create explainers 
dnn_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=readable_feature_names,
    class_names=['Healthy', 'Heart Disease'],
    mode='classification'
)

rf_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=readable_feature_names,
    class_names=['Healthy', 'Heart Disease'],
    mode='classification'
)

# Find one healthy and one sick patient
dnn_preds = (dnn_model.predict(X_test.values, verbose=0).flatten() > 0.5).astype(int)
healthy_idx = np.where(dnn_preds == 0)[0][0]
sick_idx = np.where(dnn_preds == 1)[0][0]

print(f"Selected patients: Healthy = index {healthy_idx}, Sick = index {sick_idx}")
print(f"Actual labels:     Healthy patient = {y_test[healthy_idx]}, Sick patient = {y_test[sick_idx]}")
print()

# Explain both patients with both models
for idx, label in [(healthy_idx, "Healthy"), (sick_idx, "Heart_Disease")]:
    print(f"\n{'='*60}")
    print(f"Patient {idx} - Predicted: {label}")
    print(f"{'='*60}")

    # DNN explanation
    dnn_exp = dnn_explainer.explain_instance(
        data_row=X_test.iloc[idx].values,
        predict_fn=dnn_predict_fn,
        num_features=10
    )
    print(f"\n--- DNN LIME (top 10 features) ---")
    for rule in dnn_exp.as_list():
        print(f"  {rule[0]}: {rule[1]:+.4f}")
    dnn_exp.save_to_file(f"{OUT_DIR}/lime_dnn_patient_{idx}_{label.lower()}.html")

    # RF explanation
    rf_exp = rf_explainer.explain_instance(
        data_row=X_test.iloc[idx].values,
        predict_fn=rf_predict_fn,
        num_features=10
    )
    print(f"\n--- RF LIME (top 10 features) ---")
    for rule in rf_exp.as_list():
        print(f"  {rule[0]}: {rule[1]:+.4f}")
    rf_exp.save_to_file(f"{OUT_DIR}/lime_rf_patient_{idx}_{label.lower()}.html")

print(f"\n\nSaved 4 HTML reports:")
print(f"  - {OUT_DIR}/lime_dnn_patient_{healthy_idx}_healthy.html")
print(f"  - {OUT_DIR}/lime_dnn_patient_{sick_idx}_heart_disease.html")
print(f"  - {OUT_DIR}/lime_rf_patient_{healthy_idx}_healthy.html")
print(f"  - {OUT_DIR}/lime_rf_patient_{sick_idx}_heart_disease.html")
