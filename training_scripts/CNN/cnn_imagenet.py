import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

data_dir = 'data/HAM10000/skin_cancer_data'

# Stratified train/val split
# image_dataset_from_directory doesnt do stratified splitting with imbalanced
# datasets (like HAM10000), certain seeds produce validation sets with almost no benign samples
# We fix this by collecting all
# file paths and labels, then using sklearn's stratified split.

class_names = sorted(os.listdir(data_dir))  # ['benign', 'malignant']
class_names = [c for c in class_names if os.path.isdir(os.path.join(data_dir, c))]
print(f"Class order: {class_names}")   # class 0 = benign, class 1 = malignant

all_paths = []
all_labels = []
for label_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for fname in sorted(os.listdir(class_dir)):
        fpath = os.path.join(class_dir, fname)
        if os.path.isfile(fpath):
            all_paths.append(fpath)
            all_labels.append(label_idx)

all_paths = np.array(all_paths)
all_labels = np.array(all_labels)

print(f"Total images: {len(all_paths)}")
for i, name in enumerate(class_names):
    print(f"  {name} (class {i}): {np.sum(all_labels == i)}")

# Stratified 80/20 split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

print(f"\nTrain: {len(train_paths)} | Val: {len(val_paths)}")
for i, name in enumerate(class_names):
    print(f"  Train {name}: {np.sum(train_labels == i)} | Val {name}: {np.sum(val_labels == i)}")


# Build tf.data.Dataset from file paths
def load_and_preprocess(path, label):
    """Read image file, decode, resize, and rescale to [0,1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0   # rescale to [0,1]
    return img, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(train_paths), seed=42)
train_dataset = train_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Transfer learning backbone
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # frozen backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs=base_model.input, outputs=outputs)

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# class weight computed dynamically from actual training labels
# n_benign_train = int(np.sum(train_labels == 0))
# n_malignant_train = int(np.sum(train_labels == 1))
# class_weight = {0: 1.0, 1: n_benign_train / n_malignant_train}
# print(f"\nClass weights: {{benign: {class_weight[0]:.2f}, malignant: {class_weight[1]:.2f}}}")

history = cnn_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[early_stop],
    #class_weight=class_weight
)

cnn_model.save('saved_models/cnn_tl_skin_cancer.keras')
print("CNN model saved as 'cnn_tl_skin_cancer.keras'")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('CNN Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('CNN Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/evaluation/training_curves/cnn_training_curves.png', dpi=150)
print("Training curves saved as 'cnn_training_curves.png'")
plt.close()

# Classification Report
y_true = []
y_pred_prob = []

for images, labels in val_dataset:
    preds = cnn_model.predict(images, verbose=0)
    y_pred_prob.extend(preds.flatten())
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = (np.array(y_pred_prob) > 0.5).astype(int)

print(f"\nEvaluation class order: {class_names}")
print(f"Val set distribution: benign={np.sum(y_true == 0)}, malignant={np.sum(y_true == 1)}")

print("\n Classification Report (CNN / MobileNetV2) ")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax_cm)
ax_cm.set_title('Confusion Matrix -- CNN (MobileNetV2)')
fig_cm.tight_layout()
fig_cm.savefig('outputs/evaluation/confusion_matrices/confusion_matrix_cnn_tl.png', dpi=150)
print("Confusion matrix saved as 'confusion_matrix_cnn_tl.png'")
plt.close(fig_cm)
