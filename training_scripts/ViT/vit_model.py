import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
#os.environ['LD_LIBRARY_PATH'] = '/home/ippo/Desktop/Thesis/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/ippo/Desktop/Thesis/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
# 1. Βασικές Ρυθμίσεις (Τα ViT συνήθως απαιτούν εικόνες ακριβώς 224x224)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Προσαρμοσμένα Paths
DATA_DIR = 'data/HAM10000/skin_cancer_data' 
SAVE_PATH = 'saved_models/vit_model.keras'

print("Φόρτωση δεδομένων από:", DATA_DIR)

# 2. Προετοιμασία Δεδομένων (Data Augmentation)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2 # 80% Training, 20% Validation
)

print("Δημιουργία Training Generator...")
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', 
    subset='training'
)

print("Δημιουργία Validation Generator...")
val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 3. Κατέβασμα του Google Vision Transformer (ViT-B16)
print("Λήψη του pre-trained ViT μοντέλου από το TensorFlow Hub...")
VIT_URL = "https://tfhub.dev/sayakpaul/vit_b16_classification/1"

vit_layer = hub.KerasLayer(
    VIT_URL, 
    trainable=False, # Το "κλειδώνουμε" 
)

# 4. Χτίσιμο του τελικού Μοντέλου (Χρησιμοποιώντας Functional API)
print("Χτίσιμο αρχιτεκτονικής...")

# Ορίζουμε ρητά το Input layer
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

# Περνάμε το Input μέσα από το ViT layer
x = vit_layer(inputs)

# Προσθέτουμε τα δικά μας layers (Classification Head)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x) # sigmoid για binary

# Δημιουργούμε το τελικό μοντέλο ενώνοντας inputs και outputs
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Εκπαίδευση
print("Έναρξη εκπαίδευσης...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 6. Αποθήκευση
model.save(SAVE_PATH)
print(f"🎉 Το ViT εκπαιδεύτηκε και αποθηκεύτηκε επιτυχώς στο: {SAVE_PATH}")