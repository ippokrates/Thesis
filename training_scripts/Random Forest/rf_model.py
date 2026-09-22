'''
impport pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

X_train = pd.read_csv("data/HDD/X_train_ready.csv")
X_test = pd.read_csv("data/HDD/X_test_ready.csv")

# .values.ravel() converts the pandas column into the exact 1D array format sklearn expects
y_train = pd.read_csv("data/HDD/y_train_ready.csv").values.ravel() 
y_test = pd.read_csv("data/HDD/y_test_ready.csv").values.ravel()

# n_estimators=100, 100 decision trees.
# random_state=42 ensures you get the exact same results every time you run the code.
rf_model = RandomForestClassifier(n_estimators=100,
    max_depth=5,             # to prevent memorization
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42)

# train the model
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(classification_report(y_test, y_pred, target_names=["Healthy (0)", "Heart Disease (1)"]))
print(confusion_matrix(y_test, y_pred))
joblib.dump(rf_model, "saved_models/rf_model.pkl")
print("RF model saved as 'rf_model.pkl'")

# 5-fold cross-validation for more reliable evaluation
# uses the FULL training set (not the single 80/20 split)
print("--- 5-Fold Stratified Cross-Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for metric in ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']:
    scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring=metric)
    print(f"{metric:>10}: {scores.mean():.3f} ± {scores.std():.3f}")
'''
############################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, balanced_accuracy_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib

X_train = pd.read_csv("data/HDD/X_train_ready.csv")
X_test = pd.read_csv("data/HDD/X_test_ready.csv")

y_train = pd.read_csv("data/HDD/y_train_ready.csv").values.ravel() 
y_test = pd.read_csv("data/HDD/y_test_ready.csv").values.ravel()

rf_model = RandomForestClassifier(n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_pred_probs = rf_model.predict_proba(X_test)[:, 1]  # χρειάζεται για AUC

print(classification_report(y_test, y_pred, target_names=["Healthy (0)", "Heart Disease (1)"]))
print(confusion_matrix(y_test, y_pred))

# AUC and Balanced Accuracy
auc_score = roc_auc_score(y_test, y_pred_probs)
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"RF AUC: {auc_score:.4f}")
print(f"RF Balanced Accuracy: {bal_acc:.4f}")

# Confusion matrix as image
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy (0)", "Heart Disease (1)"])
disp.plot()
plt.title("Confusion Matrix - RF")
plt.savefig("outputs/evaluation/confusion_matrices/confusion_matrix_rf.png", dpi=150)
print("Confusion matrix saved as 'confusion_matrix_rf.png'")

joblib.dump(rf_model, "saved_models/rf_model.pkl")
print("RF model saved as 'rf_model.pkl'")

# 5-fold cross-validation for more reliable evaluation
print("\n--- 5-Fold Stratified Cross-Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for metric in ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']:
    scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring=metric)
    print(f"{metric:>10}: {scores.mean():.3f} ± {scores.std():.3f}")  #  indented σωστά
