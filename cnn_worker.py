"""
cnn_worker.py — CNN inference worker for diagnose.py
=====================================================
Called as a subprocess by diagnose.py so that the CNN .keras model
is loaded WITHOUT TF_USE_LEGACY_KERAS=1 (which is required by the ViT
but breaks native tf.keras model deserialization).

Usage (called internally by diagnose.py — not meant to be run directly):
    python cnn_worker.py --image PATH --model PATH --alpha 0.4 --npy_out PATH

Output:
    Writes Grad-CAM overlay array (uint8 BGR, CNN_SIZE) to --npy_out as .npy
    Prints a single JSON line to stdout: {"cnn_prob": <float>}
"""

import os
# NO TF_USE_LEGACY_KERAS — this is the whole point of the subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import argparse
import numpy as np
import cv2
import tensorflow as tf

CNN_SIZE = (128, 128)  # (H, W) — must match training


def load_image_for_model(image_path: str) -> np.ndarray:
    """Same pipeline as training: tf.io → decode → resize → /255."""
    img_raw     = tf.io.read_file(image_path)
    img_decoded = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img_resized = tf.image.resize(img_decoded, list(CNN_SIZE))
    img_scaled  = img_resized.numpy() / 255.0
    return np.expand_dims(img_scaled.astype(np.float32), axis=0)  # (1, 128, 128, 3)


def load_image_bgr(image_path: str) -> np.ndarray:
    """Load with OpenCV for overlay construction."""
    img = cv2.imread(image_path)
    if img is None:
        import matplotlib.image as mpimg
        img_rgb = mpimg.imread(image_path)
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (CNN_SIZE[1], CNN_SIZE[0]))  # cv2 takes (W, H)


def compute_gradcam(img_batch: np.ndarray, model, last_conv_layer_name: str = "out_relu") -> np.ndarray:
    """Grad-CAM heatmap — float [0,1] at conv-layer spatial resolution."""
    grad_model = tf.keras.Model(
        inputs  = model.input,
        outputs = [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_batch)
        class_channel = preds[:, 0]

    grads        = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output  = conv_output[0]
    heatmap      = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    hmax         = tf.math.reduce_max(heatmap)
    if hmax > 0:
        heatmap = heatmap / hmax
    return heatmap.numpy()


def gradcam_overlay(heatmap: np.ndarray, original_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """Blend heatmap with original image."""
    h, w            = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay         = heatmap_color * alpha + original_bgr * (1 - alpha)
    return np.uint8(overlay)


def main():
    parser = argparse.ArgumentParser(description="CNN worker — internal use by diagnose.py")
    parser.add_argument("--image",   required=True,          help="Path to input image")
    parser.add_argument("--model",   required=True,          help="Path to .keras model file")
    parser.add_argument("--alpha",   type=float, default=0.4, help="Grad-CAM overlay alpha")
    parser.add_argument("--npy_out", required=True,          help="Output path for Grad-CAM overlay (.npy)")
    args = parser.parse_args()

    # Load CNN model — works fine WITHOUT TF_USE_LEGACY_KERAS
    model = tf.keras.models.load_model(args.model)

    # Preprocess
    img_batch = load_image_for_model(args.image)

    # Prediction
    cnn_prob = float(model.predict(img_batch, verbose=0)[0][0])

    # Grad-CAM heatmap
    heatmap = compute_gradcam(img_batch, model, "out_relu")

    # Build BGR overlay
    orig_bgr    = load_image_bgr(args.image)
    overlay_bgr = gradcam_overlay(heatmap, orig_bgr, args.alpha)

    # Save overlay to temp .npy file (passed back to diagnose.py)
    np.save(args.npy_out, overlay_bgr)

    # Print result as JSON to stdout — diagnose.py reads this
    print(json.dumps({"cnn_prob": cnn_prob}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
