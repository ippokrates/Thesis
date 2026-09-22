import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from transformers import TFViTModel
import matplotlib.pyplot as plt
import cv2

# ── Configuration ─────────────────────────────────────────────────────────────
WEIGHTS_PATH = 'saved_models/vit_model_hf.h5'
IMAGE_PATH   = 'data/HAM10000/skin_cancer_data/malignant/ISIC_0032400.jpg'
OUTPUT_PATH  = 'vit_attention_rollout_hf_malignant_ISIC_0032400.png'
IMG_SIZE     = (224, 224)

class ViTBackboneLayer(tf.keras.layers.Layer):
    """
    Input:  (batch, H, W, C) channels-last
    Output: (batch, 197, 768) — CLS token + 196 patch tokens
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading HuggingFace ViT-B16 backbone...")
        self.vit = TFViTModel.from_pretrained("google/vit-base-patch16-224", use_safetensors=False)
        self.vit.trainable = False

    def call(self, inputs, training=False):
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])
        return self.vit(x, training=False).last_hidden_state 

    def get_config(self):
        return super().get_config()


def build_model():
    inputs        = tf.keras.Input(shape=IMG_SIZE + (3,), name='input_image')
    backbone_out  = ViTBackboneLayer(name='vit_backbone')(inputs)
    cls_token     = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name='cls_token')(backbone_out)
    x             = tf.keras.layers.Dense(128, activation='relu', name='dense_head')(cls_token)
    x             = tf.keras.layers.Dropout(0.3, name='dropout')(x)
    outputs       = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vit_hf_classifier')


print("Rebuilding model architecture...")
model = build_model()

if not os.path.exists(WEIGHTS_PATH):
    print(f"\nERROR: Weights not found at '{WEIGHTS_PATH}'")
    exit()

model.load_weights(WEIGHTS_PATH)
print(f"Weights loaded from: {WEIGHTS_PATH}")

if not os.path.exists(IMAGE_PATH):
    print(f"\nERROR: Image not found at '{IMAGE_PATH}'")
    exit()

img_raw    = tf.io.read_file(IMAGE_PATH)
img_decoded = tf.image.decode_jpeg(img_raw, channels=3)
img_resized = tf.image.resize(img_decoded, list(IMG_SIZE))
img_scaled = img_resized.numpy() / 255.0
img_batch  = np.expand_dims(img_scaled, axis=0)

# Get prediction from the model
pred_value = model.predict(img_batch, verbose=0)[0][0]
label      = "Malignant" if pred_value > 0.5 else "Benign"
print(f"Prediction: {pred_value*100:.1f}% Malignant → {label}")

# Attention Rollout
print("Computing Attention Rollout...")

backbone_layer = model.get_layer('vit_backbone')
internal_vit   = backbone_layer.vit   # TFViTModel instance

# HuggingFace ViT expects channels-first: (batch, C, H, W)
img_chw = tf.transpose(img_batch, perm=[0, 3, 1, 2])   # (1, 3, 224, 224)

# Forward pass getting attention matrices from all 12 layers
vit_outputs = internal_vit(img_chw, output_attentions=True, training=False)

# vit_outputs.attentions: tuple of 12 tensors, each (1, 12, 197, 197)
attentions = vit_outputs.attentions

# Rollout algorithm (Abnar & Zuidema, 2020)
# For each transformer layer L:
#   1. Average attention over all 12 heads → (197, 197)
#   2. Account for residual connections: A_L = 0.5*A_L + 0.5*I
#      (information flows both through attention AND directly via skip connection)
#   3. Re-normalise rows so they sum to 1
#   4. Propagate: rollout = A_L @ rollout   (layer-by-layer matrix product)
#
# After 12 layers, rollout[0, :] = how much the CLS token at the output
# can be attributed to each of the 197 input tokens.
# We take indices 1: to skip the CLS token itself → 196 patch importances.
# Reshape 196 → 14×14 to match the image grid.

rollout = np.eye(197)   # start with identity (each token attends only to itself)

for layer_attn in attentions:
    # layer_attn: (1, 12, 197, 197) → numpy → average over 12 heads
    attn_np  = layer_attn.numpy()[0]           # (12, 197, 197)
    attn_avg = np.mean(attn_np, axis=0)        # (197, 197)

    # Residual connection: blend with identity, then row-normalise
    attn_avg = attn_avg + np.eye(197)
    attn_avg = attn_avg / attn_avg.sum(axis=-1, keepdims=True)

    # Propagate through layer
    rollout = attn_avg @ rollout               # (197, 197)

# CLS token (row 0) attention to all patch tokens (indices 1:)
cls_attention = rollout[0, 1:]                 # (196,)
cls_attention = cls_attention.reshape(14, 14)  # (14, 14)

# Normalize to [0, 1]
a_min, a_max = cls_attention.min(), cls_attention.max()
cls_attention = (cls_attention - a_min) / (a_max - a_min + 1e-8)

# Build overlay
original_bgr    = cv2.imread(IMAGE_PATH)
original_bgr    = cv2.resize(original_bgr, IMG_SIZE)

# Upsample 14×14 attention map to 224×224
heatmap_resized = cv2.resize(cls_attention, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
heatmap_uint8   = np.uint8(255 * heatmap_resized)
heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# Blend: 40% heatmap, 60% original
overlay = heatmap_color * 0.4 + original_bgr * 0.6
overlay = np.uint8(overlay)

# Plot
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
# axes[0].set_title(f"Original Image\n{pred_value*100:.1f}% Malignant ({label})")
axes[0].set_title(f"Αρχική Εικόνα\nΠρόβλεψη Κινδύνου: {pred_value*100:.1f}%")
axes[0].axis('off')

# axes[1].imshow(cls_attention, cmap='jet', interpolation='bilinear')
# axes[1].set_title("Attention Rollout\n(14×14 CLS→patch attention map)")
# axes[1].axis('off')

axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1].set_title("Attention Rollout\n(upsampled to 224×224)")
axes[1].axis('off')

# plt.suptitle("ViT (HuggingFace) — Attention Rollout", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_PATH}")
# plt.show()
print(f"Prediction (this script): {pred_value:.4f}")
