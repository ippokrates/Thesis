import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_PATH   = '' # eg data/HAM10000/skin_cancer_data/malignant/ISIC_0030539.jpg
OUTPUT_PATH  = '' # eg /outputs/xai/vision/cnn_gradcam_ISIC_0030539.png

model = tf.keras.models.load_model('saved_models/cnn_tl_skin_cancer.keras')

# Print full architecture once to find the last conv layer name in MobileNetV2
model.summary()
last_conv_layer_name = 'out_relu'

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = 0  # binary sigmoid output has only one channel
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


if not os.path.exists(IMAGE_PATH):
    print("Path/Image not found")
    exit()


img = tf.io.read_file(IMAGE_PATH)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, [224, 224])
img_array_scaled = np.expand_dims(preprocess_input(img.numpy()), axis=0)  # [-1,1], must match training

prediction = model.predict(img_array_scaled)[0][0]
heatmap = make_gradcam_heatmap(img_array_scaled, model, last_conv_layer_name)

original_img = cv2.imread(IMAGE_PATH)
original_img = cv2.resize(original_img, (224, 224))

heatmap_resized = cv2.resize(heatmap, (224, 224))
heatmap_resized = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

superimposed_img = heatmap_color * 0.4 + original_img * 0.6
superimposed_img = np.uint8(superimposed_img)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title(f"Αρχική Εικόνα\nΠρόβλεψη Κινδύνου: {prediction*100:.1f}%")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM")
plt.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')

print("Saved gradcam")