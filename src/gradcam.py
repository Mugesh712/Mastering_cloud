"""
gradcam.py — Grad-CAM Heatmap Generator
=========================================
PneumoCloud AI | Multi-Cloud Pneumonia Detection System

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights
the regions of the X-ray that drove the AI's decision.

HOW IT WORKS:
  1. Run a forward pass through DenseNet-121
  2. Take the gradients from the last convolutional layer
  3. Weight each feature map by its gradient magnitude
  4. Project the result back onto the original image as a colour heatmap

If the model is not loaded, returns None gracefully.
"""

import numpy as np
import io
import base64
from PIL import Image


def generate_gradcam(model, img_array: np.ndarray) -> str | None:
    """
    Generate a Grad-CAM heatmap and return it as a base64-encoded PNG string.

    Args:
        model:     Loaded Keras DenseNet-121 model (or None)
        img_array: Preprocessed image as numpy array, shape (224, 224, 3), values in [0, 1]

    Returns:
        base64 PNG string of the heatmap overlay, or None if model unavailable
    """
    if model is None:
        return None     # no model loaded — caller handles this gracefully

    try:
        import tensorflow as tf

        # ── Step 1: Create a sub-model that outputs the last conv layer AND the final prediction ──
        # We need both to compute gradients
        last_conv_layer = model.get_layer('conv5_block16_2_conv')   # DenseNet-121 last conv layer
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        # ── Step 2: Record gradients during the forward pass ──
        img_input = np.expand_dims(img_array, axis=0)   # shape: (1, 224, 224, 3)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            # We care about the class with index 0 (pneumonia probability)
            loss = predictions[:, 0]

        # ── Step 3: Compute gradients of the prediction w.r.t. the conv layer output ──
        grads = tape.gradient(loss, conv_outputs)

        # ── Step 4: Pool gradients over spatial dimensions (Global Average Pooling) ──
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # ── Step 5: Weight each feature map channel by its pooled gradient ──
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ── Step 6: Normalise heatmap to [0, 1] ──
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # ── Step 7: Resize heatmap from conv layer size back to 224x224 ──
        heatmap_img = Image.fromarray(np.uint8(heatmap * 255))
        heatmap_img = heatmap_img.resize((224, 224), Image.LANCZOS)
        heatmap_arr = np.array(heatmap_img)

        # ── Step 8: Apply colour map (red = high attention, blue = low) ──
        # Manual jet colourmap: blue → cyan → green → yellow → red
        coloured = np.zeros((224, 224, 3), dtype=np.uint8)
        coloured[:, :, 0] = np.clip(heatmap_arr * 2 - 128, 0, 255)          # red channel
        coloured[:, :, 1] = np.clip(255 - np.abs(heatmap_arr * 2 - 255), 0, 255)  # green
        coloured[:, :, 2] = np.clip(255 - heatmap_arr * 2, 0, 255)          # blue channel

        # ── Step 9: Overlay heatmap on original X-ray image ──
        original_img = Image.fromarray(np.uint8(img_array * 255))
        heatmap_pil  = Image.fromarray(coloured)
        blended      = Image.blend(original_img, heatmap_pil, alpha=0.4)

        # ── Step 10: Encode as base64 PNG so it can be sent over HTTP or displayed ──
        buffer = io.BytesIO()
        blended.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"[GradCAM] Could not generate heatmap: {e}")
        return None
