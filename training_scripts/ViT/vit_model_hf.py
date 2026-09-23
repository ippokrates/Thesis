import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from transformers import TFViTModel
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Config
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 10
DATA_DIR   = 'data/HAM10000/skin_cancer_data'
WEIGHTS_PATH = 'saved_models/vit_model_hf.h5'   # weights-only .h5 file

# Custom Keras layer
class ViTBackboneLayer(tf.keras.layers.Layer):
    """
    Wraps HuggingFace TFViTModel as a Keras layer.

    Input:  (batch, H, W, C)  - channels-last (TF/Keras convention)
    Output: (batch, 197, 768) - 1 CLS token + 196 patch tokens, each 768-dim

    Internally transposes to channels-first (batch, C, H, W) because
    HuggingFace ViT follows the PyTorch channels-first convention.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading HuggingFace ViT-B16 backbone (google/vit-base-patch16-224)...")
        self.vit = TFViTModel.from_pretrained("google/vit-base-patch16-224", use_safetensors=False)
        self.vit.trainable = False   # freeze backbone, only train the Dense head

    def call(self, inputs, training=False):
        # Convert from Keras channels-last (B, H, W, C) to HuggingFace channels-first (B, C, H, W)
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])
        # backbone always runs in inference mode (trainable=False)
        outputs = self.vit(x, training=False)
        return outputs.last_hidden_state   # (batch, 197, 768)

    def get_config(self):
        return super().get_config()


# Build model
def build_model():
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name='input_image')
    backbone_out = ViTBackboneLayer(name='vit_backbone')(inputs)

    # Extract the CLS token (index 0)
    # After 12 self-attention layers, the CLS token has global
    # information from all 196 patch tokens
    cls_token = tf.keras.layers.Lambda(
        lambda x: x[:, 0, :], name='cls_token'
    )(backbone_out)                          # (batch, 768)

    # Classification head
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_head')(cls_token)
    x = tf.keras.layers.Dropout(0.3, name='dropout')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vit_hf_classifier')


# Stratified train/val split
print(f"Loading data from: {DATA_DIR}")
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Class order: {class_names}")   # ['benign', 'malignant']

all_paths = []
all_labels = []
for label_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(DATA_DIR, class_name)
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

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

print(f"\nTrain: {len(train_paths)} | Val: {len(val_paths)}")
for i, name in enumerate(class_names):
    print(f"  Train {name}: {np.sum(train_labels == i)} | Val {name}: {np.sum(val_labels == i)}")


# Build tf.data.Dataset
def load_and_preprocess(path, label):
    """Read image file, decode, resize, and rescale to [0,1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, list(IMG_SIZE))
    img = img / 255.0   # rescale to [0,1]
    return img, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(train_paths), seed=42)
train_dataset = train_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Compile and train model
model = build_model()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print("\nStarting training (backbone frozen - only Dense head is trained)...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Save weights
# We save weights only (not the full model) because ViTBackboneLayer is a
# custom layer. The architecture is rebuilt using the
# same build_model() function then weights are loaded
os.makedirs('saved_models', exist_ok=True)
model.save_weights(WEIGHTS_PATH)
print(f"\nWeights saved to: {WEIGHTS_PATH}")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('ViT Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('ViT Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/evaluation/training_curves/vit_training_curves.png', dpi=150)
print("Training curves saved as 'vit_training_curves.png'")

# Classification Report
y_true = []
y_pred_prob = []

for images, labels in val_dataset:
    preds = model.predict(images, verbose=0)
    y_pred_prob.extend(preds.flatten())
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = (np.array(y_pred_prob) > 0.5).astype(int)

print(f"\nEvaluation class order: {class_names}")
print(f"Val set distribution: benign={np.sum(y_true == 0)}, malignant={np.sum(y_true == 1)}")

print("\n Classification Report (ViT-B/16 / HuggingFace) ")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax_cm)
ax_cm.set_title('Confusion Matrix - ViT-B/16 (HuggingFace)')
fig_cm.tight_layout()
fig_cm.savefig('outputs/evaluation/confusion_matrices/confusion_matrix_vit_hf.png', dpi=150)
print("Confusion matrix saved as 'confusion_matrix_vit_hf.png'")
plt.close(fig_cm)
