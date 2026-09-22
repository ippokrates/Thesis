"""
diagnose.py
Δέχεται μία εικόνα δέρματος οποιουδήποτε μεγέθους και:
  1. Κάνει πρόβλεψη με το CNN (MobileNetV2, 128×128) + Grad-CAM
  2. Κάνει πρόβλεψη με το ViT-B/16 (HuggingFace, 224×224) + Attention Rollout
  3. Εκτυπώνει τις πιθανότητες στο τερματικό
  4. Αποθηκεύει ένα composite PNG με 3 sub-plots:
       [Αρχική εικόνα] | [Grad-CAM CNN] | [Attention Rollout ViT]

Χρήση:
    python diagnose.py --image /path/to/lesion.jpg

"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # silence TF info/warning noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"




import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import TFViTModel


SCRIPT_DIR   = Path(__file__).parent.resolve()
CNN_MODEL    = SCRIPT_DIR / "saved_models" / "cnn_tl_skin_cancer.keras"
VIT_WEIGHTS  = SCRIPT_DIR / "saved_models" / "vit_model_hf.h5"
CNN_SIZE = (128, 128)   # (H, W)
VIT_SIZE = (224, 224)   # (H, W)
DIAGNOSIS_DIR = SCRIPT_DIR / "diagnosis"
LABELS = {0: "Benign (Καλοήθης)", 1: "Malignant (Κακοήθης)"}



# ViT custom Keras layer must be identical to vit_model_hf.py
class ViTBackboneLayer(tf.keras.layers.Layer):
    """
    Wraps HuggingFace TFViTModel as a Keras layer.
    Input:  (batch, H, W, C)  channels-last
    Output: (batch, 197, 768) CLS + 196 patch tokens
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vit = TFViTModel.from_pretrained(
            "google/vit-base-patch16-224", use_safetensors=False
        )
        self.vit.trainable = False

    def call(self, inputs, training=False):
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])
        return self.vit(x, training=False).last_hidden_state

    def get_config(self):
        return super().get_config()


def build_vit_model():
    # Rebuild the ViT architecture (same as vit_model_hf.py)
    inputs       = tf.keras.Input(shape=VIT_SIZE + (3,), name="input_image")
    backbone_out = ViTBackboneLayer(name="vit_backbone")(inputs)
    cls_token    = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name="cls_token")(backbone_out)
    x            = tf.keras.layers.Dense(128, activation="relu", name="dense_head")(cls_token)
    x            = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs      = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vit_hf_classifier")


def load_image_for_model(image_path: str, size: tuple) -> np.ndarray:
    img_raw     = tf.io.read_file(image_path)

    # decode_image handles JPEG/PNG/GIF/BMP automatically
    img_decoded = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    img_resized = tf.image.resize(img_decoded, list(size))
    img_scaled  = img_resized.numpy() / 255.0
    return np.expand_dims(img_scaled.astype(np.float32), axis=0)   # (1, H, W, 3)


def load_image_bgr(image_path: str, size: tuple) -> np.ndarray:
    # Load image with OpenCV (BGR) and resize, used for overlay construction
    img = cv2.imread(image_path)
    if img is None:
        # Fallback: load with matplotlib and convert
        import matplotlib.image as mpimg
        img_rgb = mpimg.imread(image_path)
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (size[1], size[0]))   # cv2 takes (W, H)



def compute_attention_rollout(img_batch: np.ndarray, vit_model) -> np.ndarray:
    """
    For each of the 12 transformer layers:
      1. Average attention over 12 heads  (197, 197)
      2. Add identity (residual connections): A = A + I
      3. Row-normalise
      4. Propagate: rollout = A @ rollout
    Returns a 2-D float array (14x14) normalised to [0, 1].
    """
    backbone_layer = vit_model.get_layer("vit_backbone")
    internal_vit   = backbone_layer.vit

    # HuggingFace ViT expects channels-first (batch, C, H, W)
    img_chw = tf.transpose(img_batch, perm=[0, 3, 1, 2])

    vit_outputs = internal_vit(img_chw, output_attentions=True, training=False)
    attentions  = vit_outputs.attentions   # tuple of 12 x (1, 12, 197, 197)

    n_tokens = 197
    rollout  = np.eye(n_tokens, dtype=np.float32)

    for layer_attn in attentions:
        attn_np  = layer_attn.numpy()[0]                              # (12, 197, 197)
        attn_avg = np.mean(attn_np, axis=0)                          # (197, 197)
        attn_avg = attn_avg + np.eye(n_tokens)                       # residual
        attn_avg = attn_avg / attn_avg.sum(axis=-1, keepdims=True)   # row-norm
        rollout  = attn_avg @ rollout

    cls_attention = rollout[0, 1:]                           # (196,)
    cls_attention = cls_attention.reshape(14, 14)            # (14, 14)

    a_min, a_max  = cls_attention.min(), cls_attention.max()
    cls_attention = (cls_attention - a_min) / (a_max - a_min + 1e-8)
    return cls_attention


def attention_overlay(attn_map: np.ndarray, original_bgr: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Upsample 14x14 attention map to image size and blend with original (BGR)."""
    h, w = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = heatmap_color * alpha + original_bgr * (1 - alpha)
    return np.uint8(overlay)


def build_composite(
    image_path: str,
    cnn_prob: float,
    vit_prob: float,
    gradcam_bgr: np.ndarray,
    rollout_bgr: np.ndarray,
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    Build a 1x3 composite figure:
       [Αρχική Εικόνα] | [Grad-CAM CNN] | [Attention Rollout ViT]
    """
    # Load original at VIT_SIZE for display (larger resolution looks better)
    orig_bgr = load_image_bgr(image_path, VIT_SIZE)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    def prob_label(prob: float) -> str:
        label = "Malignant (Κακοήθης)" if prob > 0.5 else "Benign (Καλοήθης)"
        return f"{label}\n{prob * 100:.1f}% πιθανότητα κακοήθειας"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")

    panel_cfg = [
        (orig_rgb,                                     "Αρχική Εικόνα",                                            "#e0e0e0"),
        (cv2.cvtColor(gradcam_bgr, cv2.COLOR_BGR2RGB), f"Grad-CAM  ·  CNN (MobileNetV2)\n{prob_label(cnn_prob)}", "#ff6b6b"),
        (cv2.cvtColor(rollout_bgr, cv2.COLOR_BGR2RGB), f"Attention Rollout  ·  ViT-B/16\n{prob_label(vit_prob)}", "#4ecdc4"),
    ]

    for ax, (img, title, color) in zip(axes, panel_cfg):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    # Super-title with ensemble / summary
    cnn_decision = "Κακοήθης" if cnn_prob > 0.5 else "Καλοήθης"
    vit_decision = "Κακοήθης" if vit_prob > 0.5 else "Καλοήθης"
    plt.suptitle(
        f"Αποτέλεσμα  ·  CNN: {cnn_decision} ({cnn_prob*100:.1f}%)   |   "
        f"ViT: {vit_decision} ({vit_prob*100:.1f}%)",
        fontsize=13,
        fontweight="bold",
        color="white",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Δερματολογικό Διαγνωστικό Εργαλείο - CNN (Grad-CAM) & ViT (Attention Rollout)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Παραδείγματα:
  python diagnose.py --image data/HAM10000/skin_cancer_data/malignant/ISIC_0027525.jpg
  python diagnose.py --image /path/to/lesion.jpg
        """,
    )
    parser.add_argument("--image",       required=True,          help="Διαδρομή εικόνας δέρματος (JPEG/PNG/BMP κ.λπ.)")
    parser.add_argument("--cnn_model",   default=str(CNN_MODEL),  help=f"CNN model (.keras) default: {CNN_MODEL}")
    parser.add_argument("--vit_weights", default=str(VIT_WEIGHTS),help=f"ViT weights (.h5)  default: {VIT_WEIGHTS}")
    parser.add_argument("--alpha",       type=float, default=0.4, help="Διαφάνεια XAI overlay (0-1, default: 0.4)")
    parser.add_argument("--dpi",         type=int,   default=150, help="DPI εξόδου (default: 150)")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image
    if not os.path.isfile(image_path):
        print(f"\n[ERROR] Η εικόνα δεν βρέθηκε: {image_path}")
        sys.exit(1)

    # Validate model files
    for fpath, label in [(args.cnn_model, "CNN model"), (args.vit_weights, "ViT weights")]:
        if not os.path.isfile(fpath):
            print(f"\n[ERROR] {label} δεν βρέθηκε: {fpath}")
            sys.exit(1)

    # Output directory
    output_dir = DIAGNOSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stem      = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = str(output_dir / f"{stem}_diagnosis_{timestamp}.png")
    print("\nΦόρτωση CNN (MobileNetV2) subprocess")

    cnn_worker = SCRIPT_DIR / "cnn_worker.py"
    if not cnn_worker.is_file():
        print(f"\n[ERROR] cnn_worker.py δεν βρέθηκε: {cnn_worker}")
        print("Βεβαιωθείτε ότι το cnn_worker.py βρίσκεται δίπλα στο diagnose.py.")
        sys.exit(1)

    # Temp file for the Grad-CAM overlay array
    npy_fd, npy_path = tempfile.mkstemp(suffix=".npy")
    os.close(npy_fd)

    # Build a clean environment, strip TF_USE_LEGACY_KERAS so the worker
    # loads the .keras model with native Keras 3, not tf_keras (legacy).
    # Subprocesses inherit the parent env by default, which would re-introduce the conflict we are trying to avoid.
    clean_env = {k: v for k, v in os.environ.items() if k != "TF_USE_LEGACY_KERAS"}

    try:
        proc = subprocess.run(
            [
                sys.executable, str(cnn_worker),
                "--image",   image_path,
                "--model",   args.cnn_model,
                "--alpha",   str(args.alpha),
                "--npy_out", npy_path,
            ],
            env=clean_env,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            print("\n[ERROR] CNN worker απέτυχε:")
            # Show from the beginning so the actual traceback is visible
            print(proc.stderr if proc.stderr else "(no stderr)")
            sys.exit(1)

        # Parse result from stdout (last non-empty line = JSON)
        json_line = [l for l in proc.stdout.strip().splitlines() if l.strip()][-1]
        cnn_data  = json.loads(json_line)
        cnn_prob  = cnn_data["cnn_prob"]

        # Load Grad-CAM overlay saved by worker
        gradcam_bgr = np.load(npy_path)

    finally:
        if os.path.exists(npy_path):
            os.remove(npy_path)

    cnn_label = "Malignant (Κακοήθης)" if cnn_prob > 0.5 else "Benign (Καλοήθης)"
    #print(f"       -> CNN: {cnn_label}  |  P(malignant) = {cnn_prob*100:.1f}%  |  P(benign) = {(1-cnn_prob)*100:.1f}%")



    print("Φόρτωση ViT-B/16")
    vit_model = build_vit_model()
    vit_model.load_weights(args.vit_weights)

    img_vit  = load_image_for_model(image_path, VIT_SIZE)
    vit_prob = float(vit_model.predict(img_vit, verbose=0)[0][0])

    attn_map     = compute_attention_rollout(img_vit, vit_model)
    orig_bgr_vit = load_image_bgr(image_path, VIT_SIZE)
    rollout_bgr  = attention_overlay(attn_map, orig_bgr_vit, alpha=args.alpha)

    vit_label = "Malignant (Κακοήθης)" if vit_prob > 0.5 else "Benign (Καλοήθης)"
    #print(f"       -> ViT: {vit_label}  |  P(malignant) = {vit_prob*100:.1f}%  |  P(benign) = {(1-vit_prob)*100:.1f}%")

    # Building final image
    print("Δημιουργία εικόνας")
    build_composite(
        image_path  = image_path,
        cnn_prob    = cnn_prob,
        vit_prob    = vit_prob,
        gradcam_bgr = gradcam_bgr,
        rollout_bgr = rollout_bgr,
        output_path = out_path,
        dpi         = args.dpi,
    )

    # Results Summary 
    print("\n" + "-" * 60)
    print("  ΑΠΟΤΕΛΕΣΜΑΤΑ ")
    print("-" * 60)
    print(f"  CNN  (MobileNetV2) : {cnn_label}")
    print(f"    P(malignant)  = {cnn_prob*100:6.1f}%")
    print(f"    P(benign)  = {(1-cnn_prob)*100:6.1f}%")
    print()
    print(f"  ViT  (ViT-B/16)    : {vit_label}")
    print(f"    P(malignant)  = {vit_prob*100:6.1f}%")
    print(f"    P(benign)  = {(1-vit_prob)*100:6.1f}%")
    print("-" * 60)

    # Ensemble (simple average)
    avg_prob  = (cnn_prob + vit_prob) / 2.0
    avg_label = "Malignant (Κακοήθης)" if avg_prob > 0.5 else "Benign (Καλοήθης)"
    print(f"\n  ENSEMBLE (average): {avg_label}")
    print(f"    P(malignant)  = {avg_prob*100:6.1f}%")
    print(f"\n  Saved to: {out_path}")
    print("-" * 60)



if __name__ == "__main__":
    main()
