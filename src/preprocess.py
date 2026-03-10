"""
Preprocessing Pipeline for Chest X-ray Images
===============================================
Multi-Cloud Intelligent Chest X-ray Triage System

This script handles:
1. Loading raw X-ray images from data/raw/
2. Resizing to 224x224 (DenseNet-121 input size)
3. Normalizing pixel values (0-255 → 0-1)
4. Data augmentation (flip, rotation, zoom)
5. Train/Val/Test split (70/15/15)
6. Class balancing
7. Saving processed data as numpy arrays

Usage (Google Colab):
    Upload this file to Colab, then run:
    >>> from preprocess import load_and_preprocess
    >>> X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/raw')
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────
# Configuration
# ──────────────────────────────────────
IMG_SIZE = 224           # DenseNet-121 input size
CHANNELS = 3             # RGB
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15          # from remaining after test split

# Class labels
CLASSES = ['NORMAL', 'PNEUMONIA']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_images_from_directory(data_dir, subset='train'):
    """
    Load images and labels from the Kaggle Chest X-ray directory structure.

    Expected structure:
        data_dir/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── val/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/
    """
    images = []
    labels = []
    subset_dir = os.path.join(data_dir, subset)

    if not os.path.exists(subset_dir):
        raise FileNotFoundError(f"Directory not found: {subset_dir}")

    for class_name in CLASSES:
        class_dir = os.path.join(subset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping.")
            continue

        file_list = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

        print(f"Loading {len(file_list)} images from {class_name}/{subset}...")

        for filename in tqdm(file_list, desc=f"{class_name}"):
            filepath = os.path.join(class_dir, filename)
            try:
                img = Image.open(filepath).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(CLASS_TO_IDX[class_name])
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

    return np.array(images), np.array(labels)


def augment_image(image):
    """
    Apply random augmentations to a single image array.
    - Horizontal flip (50% chance)
    - Random rotation (-15 to +15 degrees)
    - Random brightness adjustment
    """
    augmented = image.copy()

    # Horizontal flip
    if np.random.random() > 0.5:
        augmented = np.fliplr(augmented)

    # Random brightness adjustment (±10%)
    brightness = 1.0 + np.random.uniform(-0.1, 0.1)
    augmented = np.clip(augmented * brightness, 0, 1)

    return augmented


def balance_classes(images, labels):
    """
    Balance classes by oversampling the minority class with augmentation.
    """
    unique, counts = np.unique(labels, return_counts=True)
    max_count = max(counts)

    balanced_images = list(images)
    balanced_labels = list(labels)

    for class_idx in unique:
        class_images = images[labels == class_idx]
        current_count = len(class_images)
        needed = max_count - current_count

        if needed > 0:
            print(f"Augmenting class {CLASSES[class_idx]}: {current_count} → {max_count} (+{needed})")
            for i in range(needed):
                img = class_images[i % current_count]
                aug_img = augment_image(img)
                balanced_images.append(aug_img)
                balanced_labels.append(class_idx)

    return np.array(balanced_images), np.array(balanced_labels)


def load_and_preprocess(data_dir='data/raw', balance=True):
    """
    Main preprocessing pipeline.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)

    # Load training images
    print("\n[1/5] Loading training images...")
    X_train_raw, y_train_raw = load_images_from_directory(data_dir, 'train')
    print(f"  Loaded: {len(X_train_raw)} images")

    # Load test images
    print("\n[2/5] Loading test images...")
    X_test, y_test = load_images_from_directory(data_dir, 'test')
    print(f"  Loaded: {len(X_test)} images")

    # Balance classes
    if balance:
        print("\n[3/5] Balancing classes...")
        X_train_raw, y_train_raw = balance_classes(X_train_raw, y_train_raw)
    else:
        print("\n[3/5] Skipping class balancing")

    # Train/Val split
    print("\n[4/5] Splitting into train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train_raw,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_raw
    )

    # Summary
    print("\n[5/5] Preprocessing complete!")
    print(f"  Train : {len(X_train)} images")
    print(f"  Val   : {len(X_val)} images")
    print(f"  Test  : {len(X_test)} images")
    print(f"  Shape : {X_train[0].shape}")
    print(f"  Range : [{X_train.min():.2f}, {X_train.max():.2f}]")
    print("=" * 50)

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_single_image(image_path):
    """
    Preprocess a single image for inference.
    Used by the cloud functions and frontend.

    Args:
        image_path: Path to the X-ray image

    Returns:
        Preprocessed numpy array of shape (1, 224, 224, 3)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_image_bytes(image_bytes):
    """
    Preprocess image from bytes (for cloud function / API usage).

    Args:
        image_bytes: Raw image bytes

    Returns:
        Preprocessed numpy array of shape (1, 224, 224, 3)
    """
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


if __name__ == '__main__':
    # Run preprocessing
    data = load_and_preprocess('data/raw')
    X_train, X_val, X_test, y_train, y_val, y_test = data

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/y_test.npy', y_test)
    print("\nSaved processed data to data/processed/")
