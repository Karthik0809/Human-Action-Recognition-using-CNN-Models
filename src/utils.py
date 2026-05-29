import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name='Model'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Training and Validation Loss')
    ax1.legend()

    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} — Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_metrics.png', dpi=150)
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, classes, model_name='Model'):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=150)
    plt.show()
