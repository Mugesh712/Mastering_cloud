"""
Model Evaluation & Metrics
============================
Multi-Cloud Intelligent Chest X-ray Triage System

This script evaluates the trained CNN model and generates:
1. Confusion Matrix
2. AUC-ROC Curve
3. Precision / Recall / F1-Score
4. Classification Report
5. Sample predictions visualization

Usage (Google Colab):
    >>> from evaluate import evaluate_model, plot_all_metrics
    >>> evaluate_model('models/pneumonia_model.h5', X_test, y_test)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    f1_score, accuracy_score
)
import tensorflow as tf
import os


def evaluate_model(model_path, X_test, y_test, class_names=None):
    """
    Complete model evaluation pipeline.

    Args:
        model_path: Path to saved .h5 model
        X_test: Test images (numpy array)
        y_test: Test labels (numpy array)
        class_names: List of class names

    Returns:
        dict: All evaluation metrics
    """
    if class_names is None:
        class_names = ['NORMAL', 'PNEUMONIA']

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    # Load model
    print("\n[1/5] Loading model...")
    model = tf.keras.models.load_model(model_path)

    # Predict
    print("[2/5] Running predictions on test set...")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Metrics
    print("[3/5] Computing metrics...")
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # AUC-ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {roc_auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'auc_roc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision,
        'recall_curve': recall,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'report': report
    }

    return metrics


def plot_confusion_matrix(cm, class_names=None, save_path='models/confusion_matrix.png'):
    """Plot and save confusion matrix heatmap."""
    if class_names is None:
        class_names = ['NORMAL', 'PNEUMONIA']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={'size': 16})
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path='models/roc_curve.png'):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved ROC curve to {save_path}")


def plot_precision_recall_curve(precision, recall, save_path='models/precision_recall.png'):
    """Plot and save Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved precision-recall curve to {save_path}")


def plot_sample_predictions(X_test, y_test, y_pred, y_pred_prob,
                            class_names=None, n_samples=8,
                            save_path='models/sample_predictions.png'):
    """Plot grid of sample predictions with confidence."""
    if class_names is None:
        class_names = ['NORMAL', 'PNEUMONIA']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    for idx, ax in zip(indices, axes.flatten()):
        ax.imshow(X_test[idx], cmap='gray')
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        conf = y_pred_prob[idx].flatten()[0]

        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({conf:.2f})',
                     color=color, fontsize=10)
        ax.axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved sample predictions to {save_path}")


def plot_all_metrics(metrics, X_test=None, y_test=None):
    """Generate all evaluation plots."""
    print("\n[4/5] Generating evaluation plots...")

    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc_roc'])
    plot_precision_recall_curve(metrics['precision_curve'], metrics['recall_curve'])

    if X_test is not None and y_test is not None:
        plot_sample_predictions(X_test, y_test, metrics['y_pred'], metrics['y_pred_prob'])

    print("\n[5/5] All evaluation plots saved to models/ directory")


if __name__ == '__main__':
    # Load test data
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')

    # Evaluate
    metrics = evaluate_model('models/pneumonia_model.h5', X_test, y_test)
    plot_all_metrics(metrics, X_test, y_test)
