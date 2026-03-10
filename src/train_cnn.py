"""
CNN Model Training — DenseNet-121 for Chest X-ray Classification
=================================================================
Multi-Cloud Intelligent Chest X-ray Triage System

This script:
1. Loads preprocessed X-ray images
2. Builds DenseNet-121 with custom classification head
3. Trains using transfer learning (ImageNet weights)
4. Saves the trained model as .h5 file

Usage (Google Colab):
    1. Upload preprocess.py and train_cnn.py to Colab
    2. Upload the Kaggle dataset to Colab
    3. Run:
       >>> from preprocess import load_and_preprocess
       >>> X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/raw')
       >>> from train_cnn import build_model, train_model
       >>> model = build_model()
       >>> history = train_model(model, X_train, y_train, X_val, y_val)
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ──────────────────────────────────────
# Configuration
# ──────────────────────────────────────
IMG_SIZE = 224
NUM_CLASSES = 2          # Normal, Pneumonia (binary)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'models/pneumonia_model.h5'


def build_model(num_classes=NUM_CLASSES):
    """
    Build DenseNet-121 with custom classification head.

    Architecture:
        DenseNet-121 (pretrained on ImageNet, frozen base)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dense(256, relu)
        → Dropout(0.5)
        → Dense(128, relu)
        → Dropout(0.3)
        → Dense(1, sigmoid)     [binary]
    """
    # Load pretrained DenseNet-121 (without top classification layers)
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base model layers (transfer learning)
    for layer in base_model.layers:
        layer.trainable = False

    # Build custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    if num_classes == 2:
        output = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        output = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=['accuracy']
    )

    print(f"\nModel built successfully!")
    print(f"  Total params: {model.count_params():,}")
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Loss function: {loss}")

    return model


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=EPOCHS, batch_size=BATCH_SIZE, save_path=MODEL_SAVE_PATH):
    """
    Train the model with callbacks for early stopping and learning rate reduction.

    Returns:
        history object with training metrics
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    callbacks = [
        # Stop training if val_loss doesn't improve for 5 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when val_loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n" + "=" * 50)
    print("TRAINING STARTED")
    print("=" * 50)
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  Train size : {len(X_train)}")
    print(f"  Val size   : {len(X_val)}")
    print("=" * 50 + "\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nModel saved to: {save_path}")
    return history


def fine_tune_model(model, X_train, y_train, X_val, y_val,
                    unfreeze_layers=30, epochs=10, save_path=MODEL_SAVE_PATH):
    """
    Fine-tune the model by unfreezing the last N layers of DenseNet-121.
    Call this AFTER initial training for better accuracy.
    """
    # Unfreeze last N layers
    for layer in model.layers[-unfreeze_layers:]:
        layer.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"\nFine-tuning: Unfroze last {unfreeze_layers} layers")
    print(f"  Trainable params: {trainable:,}")

    history = train_model(model, X_train, y_train, X_val, y_val,
                          epochs=epochs, save_path=save_path)
    return history


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved training history plot to models/training_history.png")


if __name__ == '__main__':
    from preprocess import load_and_preprocess

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/raw')

    # Build and train
    model = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Plot results
    plot_training_history(history)

    # Fine-tune (optional, for better accuracy)
    history_ft = fine_tune_model(model, X_train, y_train, X_val, y_val)
    plot_training_history(history_ft)
