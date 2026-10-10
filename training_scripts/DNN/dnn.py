import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, balanced_accuracy_score, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)


X_train = pd.read_csv("data/HDD/X_train_ready.csv")
X_test = pd.read_csv("data/HDD/X_test_ready.csv")
y_train = pd.read_csv("data/HDD/y_train_ready.csv").values.ravel()
y_test = pd.read_csv("data/HDD/y_test_ready.csv").values.ravel()

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

dnn_model = Sequential()

# input layer: 13 medical features
dnn_model.add(Input(shape=(13,)))
# We use 16 artificial neurons and the 'relu' activation function.
dnn_model.add(Dense(16, activation='relu'))

# turn off 20% of neurons to prevent overfitting
dnn_model.add(Dropout(0.2))

# Second Layer 8 neurons
dnn_model.add(Dense(8, activation='relu'))

# Output Layer
# sigmoid function transforms any real-valued input into a value between 0 and 1 (healthy/sick)
dnn_model.add(Dense(1, activation='sigmoid'))
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# early stopping if model doesnt improve after 10 epochs and keeps the best version
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the Model
# epochs=50 network looks at the dataset 50 times
# batch_size=16: updates after looking at 16 patients at a time.
history = dnn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stop])

# predictions on test data
y_pred_probs = dnn_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).ravel() # convert bool to int, above 0.5 means heart disease (1)

print("DNN Classification Report")
print(classification_report(y_test, y_pred, target_names=["Healthy (0)", "Heart Disease (1)"]))

print("DNN Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# AUC and Balanced Accuracy
auc_score = roc_auc_score(y_test, y_pred_probs)
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"DNN AUC: {auc_score:.4f}")
print(f"DNN Balanced Accuracy: {bal_acc:.4f}")

# Confusion matrix as image
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy (0)", "Heart \n Disease (1)"])
disp.plot()
plt.title("Confusion Matrix - DNN")
plt.savefig("outputs/evaluation/confusion_matrices/confusion_matrix_dnn.png", dpi=150)
print("Confusion matrix saved as 'confusion_matrix_dnn.png'")




dnn_model.save("saved_models/dnn_model.keras")
print("DNN model saved as 'dnn_model.keras'")

# Training curves
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
plt.savefig("outputs/evaluation/training_curves/dnn_training_curves.png", dpi=150)
print("Training curves saved as 'dnn_training_curves.png'")
