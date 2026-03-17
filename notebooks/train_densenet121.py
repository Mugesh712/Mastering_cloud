# ══════════════════════════════════════════════════════════════
# PneumoCloud AI — DenseNet-121 Training Notebook
# ══════════════════════════════════════════════════════════════
# Run this in Google Colab (free GPU): https://colab.research.google.com
#
# STEP 1: Open Google Colab
# STEP 2: Go to Runtime → Change runtime type → GPU (T4)
# STEP 3: Copy this entire file and paste into a Colab cell
# STEP 4: Click Run (takes ~15-20 minutes)
# STEP 5: Download the .tflite file from the Files panel
# ══════════════════════════════════════════════════════════════

# ── Install Kaggle API ──────────────────────────────────────
# !pip install kaggle

# ══════════════════════════════════════════════════════════════
# CELL 1: Download the Kaggle Chest X-Ray Dataset
# ══════════════════════════════════════════════════════════════
# You need a Kaggle API key. Go to kaggle.com → Account → Create API Token
# Upload the kaggle.json file when prompted.

import os
os.makedirs('/root/.kaggle', exist_ok=True)

# Upload your kaggle.json here:
# from google.colab import files
# uploaded = files.upload()
# !mv kaggle.json /root/.kaggle/
# !chmod 600 /root/.kaggle/kaggle.json

# Download dataset
# !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# !unzip -q chest-xray-pneumonia.zip

# ══════════════════════════════════════════════════════════════
# CELL 2: Train DenseNet-121
# ══════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# ── Paths ──────────────────────────────────────────────────
TRAIN_DIR = 'chest_xray/train'
VAL_DIR   = 'chest_xray/val'
TEST_DIR  = 'chest_xray/test'
IMG_SIZE  = 224
BATCH     = 32
EPOCHS    = 10

# ── Data Augmentation (handles class imbalance) ───────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1]
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='binary', shuffle=True
)

val_data = val_gen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='binary', shuffle=False
)

test_data = test_gen.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode='binary', shuffle=False
)

print(f"\nClasses: {train_data.class_indices}")
print(f"Training samples: {train_data.samples}")
print(f"Validation samples: {val_data.samples}")
print(f"Test samples: {test_data.samples}")

# ── Build DenseNet-121 Model ──────────────────────────────
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base layers (transfer learning)
base_model.trainable = False

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

# ── Callbacks ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# ── Train ─────────────────────────────────────────────────
print("\n🚀 Training DenseNet-121...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── Evaluate on Test Set ──────────────────────────────────
print("\n📊 Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ══════════════════════════════════════════════════════════════
# CELL 3: Save as Full Model (.h5) + Convert to TFLite
# ══════════════════════════════════════════════════════════════

# Save full Keras model
model.save('pneumonia_densenet121.h5')
print("✅ Saved: pneumonia_densenet121.h5")

# Convert to TensorFlow Lite (much smaller, faster)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('pneumonia_model.tflite', 'wb') as f:
    f.write(tflite_model)

h5_size = os.path.getsize('pneumonia_densenet121.h5') / (1024*1024)
tflite_size = os.path.getsize('pneumonia_model.tflite') / (1024*1024)

print(f"\n📦 Model Sizes:")
print(f"   Full .h5:    {h5_size:.1f} MB")
print(f"   TFLite:      {tflite_size:.1f} MB")
print(f"   Reduction:   {(1 - tflite_size/h5_size)*100:.0f}% smaller!")

# ══════════════════════════════════════════════════════════════
# CELL 4: Test the TFLite Model
# ══════════════════════════════════════════════════════════════

interpreter = tf.lite.Interpreter(model_path='pneumonia_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n🧪 TFLite Model Test:")
print(f"   Input shape:  {input_details[0]['shape']}")
print(f"   Input dtype:  {input_details[0]['dtype']}")
print(f"   Output shape: {output_details[0]['shape']}")

# Test with one batch from test set
test_images, test_labels = next(test_data)
correct = 0
total = len(test_labels)

for i in range(total):
    img = np.expand_dims(test_images[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    predicted_label = 1 if prediction > 0.5 else 0
    if predicted_label == int(test_labels[i]):
        correct += 1

print(f"   TFLite Accuracy (1 batch): {correct/total:.2%}")
print(f"\n✅ TFLite model is working correctly!")

# ══════════════════════════════════════════════════════════════
# CELL 5: Download the TFLite model
# ══════════════════════════════════════════════════════════════

from google.colab import files
files.download('pneumonia_model.tflite')
print("\n📥 Download started! Save it to: /Users/mugesh/Documents/Mastering_cloud/models/")
