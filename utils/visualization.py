import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_fscore_support)


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies,
                         model_name="Model"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_accuracies, label='Train Accuracy')
    axes[0].plot(val_accuracies,   label='Val Accuracy')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{model_name} — Accuracy'); axes[0].legend()

    axes[1].plot(train_losses, label='Train Loss')
    axes[1].plot(val_losses,   label='Val Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{model_name} — Loss'); axes[1].legend()

    plt.tight_layout(); plt.show()


def plot_confusion_matrix(all_labels, all_preds, classes, model_name="Model"):
    cm   = confusion_matrix(all_labels, all_preds)
    size = max(10, len(classes) // 2)
    plt.figure(figsize=(size, size - 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'{model_name} — Confusion Matrix')
    plt.tight_layout(); plt.show()
    return cm


def print_classification_report(all_labels, all_preds, classes, model_name="Model"):
    report = classification_report(all_labels, all_preds, target_names=classes)
    print(f"\n{model_name} — Classification Report:\n{report}")
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=1)
    print(f"Weighted  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    return p, r, f1
