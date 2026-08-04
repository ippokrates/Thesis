import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)


X_train = pd.read_csv("data/HDD/X_train_ready.csv")
X_test = pd.read_csv("data/HDD/X_test_ready.csv")
y_train = pd.read_csv("data/HDD/y_train_ready.csv").values.ravel()
y_test = pd.read_csv("data/HDD/y_test_ready.csv").values.ravel()

dnn_model = Sequential()

# input layer/first layer
# input_dim=13 because there are 13 medical features
# We use 16 artificial neurons and the 'relu' activation function.
dnn_model.add(Dense(16, input_dim=13, activation='relu'))

# turn off 20% of neurons to prevent "memorization" (overfitting)
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

# train the Model
# epochs=50 network looks at the dataset 50 times
# batch_size=16: updates after looking at 16 patients at a time.
history = dnn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.2, callbacks=[early_stop])

# predictions on test data
y_pred_probs = dnn_model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).ravel() # convert bool to int, above 0.5 means heart disease (1)

print("DNN Classification Report")
print(classification_report(y_test, y_pred, target_names=["Healthy (0)", "Heart Disease (1)"]))

print("DNN Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

dnn_model.save("saved_models/dnn_model.keras")
print("DNN model saved as 'dnn_model.keras'")

# training curves
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
print("Training curves saved as 'dnn_training_curves.png'")