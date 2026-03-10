"""
Grad-CAM Heatmap Generator
============================
Multi-Cloud Intelligent Chest X-ray Triage System

Grad-CAM (Gradient-weighted Class Activation Mapping) generates
a heatmap showing which regions of the X-ray image the CNN model
focused on when making its diagnosis.

This makes the AI model EXPLAINABLE — doctors can see WHY
the model made its prediction.

Usage:
    >>> from gradcam import generate_gradcam, overlay_gradcam
    >>> heatmap = generate_gradcam(model, image, 'conv5_block16_concat')
    >>> result = overlay_gradcam(original_image, heatmap)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
import base64


def generate_gradcam(model, img_array, last_conv_layer_name=None):
    """
    Generate Grad-CAM heatmap for a given image.

    Args:
        model: Trained Keras model
        img_array: Preprocessed image array of shape (1, 224, 224, 3)
        last_conv_layer_name: Name of the last convolutional layer.
            For DenseNet-121, use 'conv5_block16_concat'.
            If None, auto-detects.

    Returns:
        heatmap: numpy array of shape (224, 224) with values 0-1
    """
    # Auto-detect last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name:
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        raise ValueError("Could not find a convolutional layer in the model")

    # Create a model that outputs both the last conv layer and the prediction
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]    # For binary classification

    # Get gradients of the loss with respect to conv layer output
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize to original image size
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap).resize((224, 224))
    heatmap = np.array(heatmap_img, dtype=np.float32) / 255.0

    return heatmap


def overlay_gradcam(original_image, heatmap, alpha=0.4, colormap='jet'):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        original_image: Original image array (224, 224, 3) with values 0-1
        heatmap: Grad-CAM heatmap (224, 224) with values 0-1
        alpha: Transparency of heatmap overlay (0-1)
        colormap: Matplotlib colormap name

    Returns:
        overlaid_image: numpy array (224, 224, 3) with heatmap overlay
    """
    # Apply colormap to heatmap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]    # Drop alpha channel

    # Overlay on original image
    if original_image.ndim == 2:
        original_image = np.stack([original_image] * 3, axis=-1)

    overlaid = original_image * (1 - alpha) + heatmap_colored * alpha
    overlaid = np.clip(overlaid, 0, 1)

    return overlaid


def gradcam_to_base64(overlaid_image):
    """
    Convert Grad-CAM overlay image to base64 string.
    Used for sending via API / storing in database.

    Args:
        overlaid_image: numpy array (224, 224, 3) with values 0-1

    Returns:
        str: Base64 encoded PNG image
    """
    img_uint8 = np.uint8(overlaid_image * 255)
    pil_img = Image.fromarray(img_uint8)

    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def plot_gradcam(original_image, heatmap, prediction, confidence,
                 save_path=None):
    """
    Plot side-by-side: Original X-ray | Grad-CAM Heatmap | Overlay.

    Args:
        original_image: (224, 224, 3) array
        heatmap: (224, 224) array
        prediction: str ('NORMAL' or 'PNEUMONIA')
        confidence: float (0-1)
        save_path: Optional path to save the figure
    """
    overlay = overlay_gradcam(original_image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(original_image, cmap='gray' if original_image.ndim == 2 else None)
    axes[0].set_title('Original X-Ray', fontsize=14)
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(overlay)
    color = 'red' if prediction == 'PNEUMONIA' else 'green'
    axes[2].set_title(f'{prediction} ({confidence:.2%})', fontsize=14, color=color)
    axes[2].axis('off')

    plt.suptitle('Grad-CAM Explainability — Where the Model Looks',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Grad-CAM visualization to {save_path}")

    plt.show()


def generate_gradcam_for_image(model, image_path, last_conv_layer_name=None):
    """
    Complete Grad-CAM pipeline for a single image file.

    Args:
        model: Trained Keras model
        image_path: Path to the X-ray image
        last_conv_layer_name: Last conv layer name

    Returns:
        dict with original image, heatmap, overlay, prediction, confidence
    """
    # Load and preprocess
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Predict
    pred_prob = model.predict(img_input, verbose=0)[0][0]
    prediction = 'PNEUMONIA' if pred_prob > 0.5 else 'NORMAL'
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

    # Generate Grad-CAM
    heatmap = generate_gradcam(model, img_input, last_conv_layer_name)
    overlay = overlay_gradcam(img_array, heatmap)

    return {
        'original': img_array,
        'heatmap': heatmap,
        'overlay': overlay,
        'prediction': prediction,
        'confidence': float(confidence),
        'overlay_base64': gradcam_to_base64(overlay)
    }
