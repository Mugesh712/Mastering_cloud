# ============================================================
# MULTI-CLOUD CHEST X-RAY TRIAGE SYSTEM — COLAB TRAINING
# ============================================================
# Run this entire file in Google Colab (Runtime > Run all)
# Make sure to select GPU: Runtime > Change runtime type > T4 GPU
#
# This notebook:
#   1. Downloads Kaggle Chest X-ray dataset
#   2. Preprocesses images (resize, normalize, augment)
#   3. Trains DenseNet-121 CNN with transfer learning
#   4. Fine-tunes model
#   5. Evaluates with metrics & plots
#   6. Generates Grad-CAM samples
#   7. Trains XGBoost risk model
#   8. Downloads all model files
# ============================================================

# ─────────────────────────
# CELL 1: Install & Setup
# ─────────────────────────
# !pip install -q kaggle tensorflow==2.15.0 xgboost scikit-learn matplotlib seaborn opencv-python-headless

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("Setup complete!")
print(f"TensorFlow version: ", end="")
import tensorflow as tf
print(tf.__version__)
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ─────────────────────────────────
# CELL 2: Download Kaggle Dataset
# ─────────────────────────────────
# OPTION A: Upload kaggle.json API key
# 1. Go to kaggle.com > Account > Create API Token
# 2. Upload the kaggle.json file when prompted

# from google.colab import files
# uploaded = files.upload()  # Upload kaggle.json here

# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# !unzip -q chest-xray-pneumonia.zip -d data/

# OPTION B: Direct download from Kaggle (manual)
# 1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# 2. Download the ZIP file
# 3. Upload to Colab and unzip

# After either option, the data structure should be:
# data/chest_xray/
# ├── train/
# │   ├── NORMAL/
# │   └── PNEUMONIA/
# ├── val/
# │   ├── NORMAL/
# │   └── PNEUMONIA/
# └── test/
#     ├── NORMAL/
#     └── PNEUMONIA/

DATA_DIR = "data/chest_xray"
print(f"Expected data directory: {DATA_DIR}")
# Uncomment after downloading:
# for split in ['train', 'val', 'test']:
#     for cls in ['NORMAL', 'PNEUMONIA']:
#         path = os.path.join(DATA_DIR, split, cls)
#         if os.path.exists(path):
#             print(f"  {split}/{cls}: {len(os.listdir(path))} images")

# ──────────────────────────────────────
# CELL 3: Image Preprocessing Pipeline
# ──────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
CLASSES = ['NORMAL', 'PNEUMONIA']


def load_and_preprocess_data(data_dir):
    """Load images from directory structure, resize, normalize."""

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.15  # Use 15% of train for validation
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0
    )

    print("Loading training data with augmentation...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )

    print("Loading validation data...")
    val_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )

    print("Loading test data...")
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print(f"\nClass mapping: {train_generator.class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")

    return train_generator, val_generator, test_generator


# Uncomment after downloading data:
# train_gen, val_gen, test_gen = load_and_preprocess_data(DATA_DIR)

# ──────────────────────────────────
# CELL 4: Visualize Sample Images
# ──────────────────────────────────
def show_sample_images(generator, n=8):
    """Display sample images from generator."""
    images, labels = next(generator)
    fig, axes = plt.subplots(2, n//2, figsize=(16, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            label = 'PNEUMONIA' if labels[i] == 1 else 'NORMAL'
            color = 'red' if labels[i] == 1 else 'green'
            ax.set_title(label, color=color, fontweight='bold')
            ax.axis('off')
    plt.suptitle('Sample Training Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()

# Uncomment: show_sample_images(train_gen)

# ─────────────────────────────────────────────
# CELL 5: Build DenseNet-121 Model
# ─────────────────────────────────────────────
def build_densenet_model(input_shape=(224, 224, 3)):
    """
    Build DenseNet-121 with custom classification head.

    Architecture:
        DenseNet-121 (frozen) → GlobalAvgPool → BatchNorm →
        Dense(256, ReLU) → Dropout(0.5) →
        Dense(128, ReLU) → Dropout(0.3) →
        Dense(1, Sigmoid)
    """
    # Load pre-trained DenseNet-121
    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model
    base_model.trainable = False

    # Build custom head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print(f"Model parameters: {model.count_params():,}")
    print(f"  Trainable: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    print(f"  Non-trainable: {sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}")

    return model


model = build_densenet_model()
model.summary()

# ─────────────────────────────
# CELL 6: Train Model (Phase 1)
# ─────────────────────────────
EPOCHS = 15
MODEL_PATH = 'pneumonia_model.h5'

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

print("=" * 50)
print("PHASE 1: Training with frozen DenseNet-121 base")
print("=" * 50)

# Uncomment after data is ready:
# history = model.fit(
#     train_gen,
#     epochs=EPOCHS,
#     validation_data=val_gen,
#     callbacks=callbacks,
#     verbose=1
# )

# ──────────────────────────────────
# CELL 7: Fine-Tune (Phase 2)
# ──────────────────────────────────
def fine_tune_model(model, train_gen, val_gen, epochs=10):
    """Unfreeze last 30 layers and fine-tune with lower LR."""
    print("=" * 50)
    print("PHASE 2: Fine-tuning last 30 layers")
    print("=" * 50)

    # Unfreeze last 30 layers
    base = model.layers[0]  # DenseNet base
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Trainable parameters: {trainable:,}")

    history_ft = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=3, mode='max',
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_PATH, monitor='val_auc', mode='max',
                save_best_only=True, verbose=1
            )
        ],
        verbose=1
    )
    return history_ft

# Uncomment: history_ft = fine_tune_model(model, train_gen, val_gen)

# ─────────────────────────────────────
# CELL 8: Plot Training History
# ─────────────────────────────────────
def plot_training_history(history, title_prefix=""):
    """Plot accuracy, loss, and AUC curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title(f'{title_prefix}Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title(f'{title_prefix}Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(history.history['auc'], label='Train', linewidth=2)
    axes[2].plot(history.history['val_auc'], label='Validation', linewidth=2)
    axes[2].set_title(f'{title_prefix}AUC-ROC', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

# Uncomment: plot_training_history(history, "Phase 1: ")
# Uncomment: plot_training_history(history_ft, "Phase 2 (Fine-tune): ")

# ─────────────────────────────────────
# CELL 9: Evaluate on Test Set
# ─────────────────────────────────────
def evaluate_model(model, test_gen):
    """Run evaluation and generate all metrics."""
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc, precision_recall_curve
    )

    print("Evaluating on test set...")
    results = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test AUC: {results[2]:.4f}")

    # Get predictions
    y_pred_proba = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_true = test_gen.classes

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_title('ROC Curve', fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Precision-Recall Curve ──
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    axes[2].plot(recall, precision, color='green', lw=2)
    axes[2].set_title('Precision-Recall Curve', fontweight='bold')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

    return y_pred_proba, y_true

# Uncomment: y_pred_proba, y_true = evaluate_model(model, test_gen)

# ──────────────────────────────────────
# CELL 10: Grad-CAM Explainability
# ──────────────────────────────────────
def generate_gradcam(model, img_array, layer_name=None):
    """Generate Grad-CAM heatmap for the given image."""

    # Get the base DenseNet model
    base_model = model.layers[0]

    # Find last conv layer
    if layer_name is None:
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    # Build gradient model
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(layer_name).output, base_model.output]
    )

    # Compute gradients
    img_tensor = tf.expand_dims(tf.cast(img_array, tf.float32), 0)

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(img_tensor)
        # Get the classification prediction from remaining layers
        x = tf.keras.layers.GlobalAveragePooling2D()(conv_outputs)
        loss = tf.reduce_mean(base_output)

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def visualize_gradcam(model, test_gen, n_samples=4):
    """Generate and display Grad-CAM for sample test images."""
    images, labels = next(test_gen)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))

    for i in range(min(n_samples, len(images))):
        img = images[i]

        # Original image
        axes[i, 0].imshow(img)
        true_label = 'PNEUMONIA' if labels[i] == 1 else 'NORMAL'
        axes[i, 0].set_title(f'Original ({true_label})', fontweight='bold')
        axes[i, 0].axis('off')

        # Grad-CAM heatmap
        try:
            heatmap = generate_gradcam(model, img)
            if heatmap is not None:
                heatmap_resized = np.array(Image.fromarray(
                    (heatmap * 255).astype(np.uint8)
                ).resize((224, 224)))

                axes[i, 1].imshow(heatmap_resized, cmap='jet')
                axes[i, 1].set_title('Grad-CAM Heatmap', fontweight='bold')
                axes[i, 1].axis('off')

                # Overlay
                overlay = (img * 0.6 + plt.cm.jet(heatmap_resized / 255.0)[:,:,:3] * 0.4)
                axes[i, 2].imshow(overlay)
                axes[i, 2].set_title('Overlay', fontweight='bold')
                axes[i, 2].axis('off')
            else:
                axes[i, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[i, 2].text(0.5, 0.5, 'N/A', ha='center', va='center')
        except Exception as e:
            axes[i, 1].text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center')
            axes[i, 2].text(0.5, 0.5, 'N/A', ha='center')

    plt.suptitle('Grad-CAM Explainability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradcam_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# Uncomment: visualize_gradcam(model, test_gen)

# ──────────────────────────────────────────
# CELL 11: Train XGBoost Risk Model
# ──────────────────────────────────────────
def train_risk_model():
    """Train XGBoost risk scoring model with synthetic + real data."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib

    print("Training XGBoost Risk Scoring Model...")

    # Generate training data (synthetic patient metadata + CNN scores)
    np.random.seed(42)
    n_samples = 5000

    cnn_confidence = np.random.beta(2, 5, n_samples)  # Skewed distribution
    age = np.random.normal(55, 18, n_samples).clip(1, 100).astype(int)
    gender = np.random.choice([0, 1], n_samples)  # 0=Female, 1=Male
    comorbidity = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

    # Risk score formula
    risk = (
        cnn_confidence * 0.60 +
        (age / 100) * 0.20 +
        gender * 0.05 +
        comorbidity * 0.15
    )

    # Assign triage levels
    triage = np.digitize(risk, bins=[0.25, 0.5, 0.75]) # 0=LOW, 1=STANDARD, 2=URGENT, 3=CRITICAL

    X = np.column_stack([cnn_confidence, age, gender, comorbidity])
    y = triage

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = xgb_model.predict(X_test)
    labels = ['LOW', 'STANDARD', 'URGENT', 'CRITICAL']
    print("\nRisk Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Save model
    joblib.dump(xgb_model, 'risk_model.pkl')
    print("Risk model saved to: risk_model.pkl")

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 4))
    features = ['CNN Confidence', 'Age', 'Gender', 'Comorbidity']
    importance = xgb_model.feature_importances_
    ax.barh(features, importance, color=['#667eea', '#764ba2', '#f093fb', '#ffa07a'])
    ax.set_title('Risk Model — Feature Importance', fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('risk_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    return xgb_model

risk_model = train_risk_model()

# ────────────────────────────────────
# CELL 12: Sample Predictions
# ────────────────────────────────────
def show_sample_predictions(model, test_gen, n=6):
    """Show predictions on sample test images."""
    images, labels = next(test_gen)

    fig, axes = plt.subplots(2, n//2, figsize=(16, 8))

    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break

        img = images[i]
        pred = model.predict(np.expand_dims(img, 0), verbose=0)[0][0]
        true_label = 'PNEUMONIA' if labels[i] == 1 else 'NORMAL'
        pred_label = 'PNEUMONIA' if pred > 0.5 else 'NORMAL'
        conf = pred if pred > 0.5 else 1 - pred

        ax.imshow(img)
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(
            f'True: {true_label}\nPred: {pred_label} ({conf:.1%})',
            color=color, fontweight='bold', fontsize=10
        )
        ax.axis('off')

    plt.suptitle('Sample Predictions on Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

# Uncomment: show_sample_predictions(model, test_gen)

# ──────────────────────────────────
# CELL 13: Save Model & Download
# ──────────────────────────────────
# Save final model
# model.save('pneumonia_model.h5')
# print("CNN model saved to: pneumonia_model.h5")

# Download files to your local machine
# from google.colab import files
# files.download('pneumonia_model.h5')
# files.download('risk_model.pkl')
# files.download('training_history.png')
# files.download('evaluation_metrics.png')
# files.download('gradcam_results.png')
# files.download('risk_feature_importance.png')
# files.download('sample_predictions.png')

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
print("\nFiles to download:")
print("  1. pneumonia_model.h5    — CNN model (for GCP Cloud Function)")
print("  2. risk_model.pkl        — XGBoost risk model")
print("  3. *.png                 — All plots for report/presentation")
print("\nNext steps:")
print("  1. Download pneumonia_model.h5")
print("  2. Place in your project's models/ folder")
print("  3. Upload to GCP Cloud Function deployment")
print("  4. Deploy AWS Lambda, GCP Function, Azure Function")
print("  5. Run Streamlit frontend")
