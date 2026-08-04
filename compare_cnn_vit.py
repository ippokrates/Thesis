import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import random
import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# --- 1. Paths ---
DATA_DIR = 'data/HAM10000/skin_cancer_data'
CNN_MODEL_PATH = 'saved_models/cnn_skin_cancer.h5' 
VIT_MODEL_PATH = 'saved_models/vit_model.h5'

# --- 2. Φόρτωση CNN ---
print("Φόρτωση CNN μοντέλου...")
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

# --- 3. Φόρτωση ViT (Χειρουργική Μέθοδος) ---
print("Ανακατασκευή ViT αρχιτεκτονικής από τη Google...")
VIT_URL = "https://tfhub.dev/sayakpaul/vit_b16_classification/1"
vit_layer = hub.KerasLayer(VIT_URL, trainable=False)

# Φτιάχνουμε την αρχιτεκτονική ακριβώς όπως την εκπαίδευσες
inputs = tf.keras.Input(shape=(224, 224, 3))
x = vit_layer(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

vit_model = tf.keras.Model(inputs, outputs)

print("Εισαγωγή των δικών σου εκπαιδευμένων βαρών...")
# ΕΔΩ ΕΙΝΑΙ Η ΜΑΓΕΙΑ: Το by_name=True ψάχνει μόνο τα δικά σου layers. 
# Το skip_mismatch=True αγνοεί τα χαλασμένα βάρη του KerasLayer και δε σκάει!
vit_model.load_weights(VIT_MODEL_PATH, by_name=True, skip_mismatch=True)

print("Τα μοντέλα φορτώθηκαν επιτυχώς! 🎉\n")

# --- 4. Επιλογή Τυχαίων Εικόνων ---
test_images = []
classes = ['benign', 'malignant']

for class_name in classes:
    folder_path = os.path.join(DATA_DIR, class_name)
    all_images = glob.glob(os.path.join(folder_path, '*.jpg'))
    selected_images = random.sample(all_images, 2)
    for img_path in selected_images:
        test_images.append({'path': img_path, 'true_label': class_name})

random.shuffle(test_images)

# --- 5. Πρόβλεψη και Οπτικοποίηση ---
plt.figure(figsize=(15, 10))

for i, item in enumerate(test_images):
    img_path = item['path']
    true_label = item['true_label']
    
    # 1. Προετοιμασία για το CNN (128x128)
    img_cnn = image.load_img(img_path, target_size=(128, 128))
    img_array_cnn = image.img_to_array(img_cnn) / 255.0
    cnn_batch = np.expand_dims(img_array_cnn, axis=0)
    
    # 2. Προετοιμασία για το ViT (224x224)
    img_vit = image.load_img(img_path, target_size=(224, 224))
    img_array_vit = image.img_to_array(img_vit) / 255.0
    vit_batch = np.expand_dims(img_array_vit, axis=0)
    
    # Προβλέψεις (το καθένα παίρνει το σωστό μέγεθος)
    cnn_pred_val = cnn_model.predict(cnn_batch, verbose=0)[0][0]
    vit_pred_val = vit_model.predict(vit_batch, verbose=0)[0][0]
    
    cnn_pred_class = 'malignant' if cnn_pred_val > 0.5 else 'benign'
    vit_pred_class = 'malignant' if vit_pred_val > 0.5 else 'benign'
    
    cnn_conf = cnn_pred_val if cnn_pred_class == 'malignant' else 1 - cnn_pred_val
    vit_conf = vit_pred_val if vit_pred_class == 'malignant' else 1 - vit_pred_val
    
    # Δημιουργία γραφικού (δείχνουμε την εικόνα του ViT που έχει καλύτερη ανάλυση 224)
    plt.subplot(2, 2, i + 1)
    plt.imshow(img_vit)
    plt.axis('off')
    
    title_color = 'green' if cnn_pred_class == true_label and vit_pred_class == true_label else 'black'
    title_text = (
        f"ΠΡΑΓΜΑΤΙΚΟ: {true_label.upper()}\n"
        f"CNN: {cnn_pred_class.upper()} ({cnn_conf*100:.1f}%)\n"
        f"ViT: {vit_pred_class.upper()} ({vit_conf*100:.1f}%)"
    )
    plt.title(title_text, color=title_color, fontsize=12, fontweight='bold')

plt.tight_layout()
print("Εμφάνιση αποτελεσμάτων...")
plt.show()