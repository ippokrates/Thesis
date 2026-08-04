import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os

model = tf.keras.models.load_model('cnn_skin_cancer.keras')

# finding last Convolutional Layer 
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # fixes "has never been called" error
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = inputs
    last_conv_output = None
    
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x
            
    grad_model = tf.keras.Model(inputs=inputs, outputs=[last_conv_output, x])

    # calculating gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # heatmap creation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


IMAGE_PATH = 'data/HAM10000/all_ham_images/ISIC_0032258.jpg' 
if not os.path.exists(IMAGE_PATH):
    print("Path/Image not found")
    exit()

img = tf.keras.utils.load_img(IMAGE_PATH, target_size=(128, 128))
img_array = tf.keras.utils.img_to_array(img)
img_array_scaled = np.expand_dims(img_array, axis=0)

# Prediction and heatmap creation
prediction = model.predict(img_array_scaled)[0][0]
heatmap = make_gradcam_heatmap(img_array_scaled, model, last_conv_layer_name)

original_img = cv2.imread(IMAGE_PATH)
original_img = cv2.resize(original_img, (128, 128))

heatmap_resized = cv2.resize(heatmap, (128, 128))
heatmap_resized = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

superimposed_img = heatmap_color * 0.4 + original_img * 0.6
superimposed_img = np.uint8(superimposed_img)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title(f"Αρχικη Εικονα\nΠροβλεψη Κινδυνου: {prediction*100:.1f}%")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM")
plt.axis('off')
plt.tight_layout()
plt.show()