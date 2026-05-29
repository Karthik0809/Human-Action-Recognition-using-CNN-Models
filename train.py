"""
Main training script for Human Action Recognition.

Usage:
    python train.py --model vgg --data_dir JPEGImages
    python train.py --model resnet --data_dir JPEGImages
    python train.py --model densenet --data_dir JPEGImages
    python train.py --model googlenet --data_dir JPEGImages
    python train.py --model pretrained_googlenet --data_dir JPEGImages
    python train.py --model pretrained_densenet --data_dir JPEGImages
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from config import CLASSES, NUM_CLASSES, MODEL_CONFIGS
from src.dataset import get_dataloaders
from src.models import make_vgg16, make_resnet50, make_densenet, make_pretrained_densenet
from src.models.googlenet import GoogLeNet, make_pretrained_googlenet
from src.train import train_model, evaluate_model
from src.utils import plot_metrics, plot_confusion_matrix


def build_model(name, cfg):
    if name == 'vgg':
        return make_vgg16(NUM_CLASSES)
    elif name == 'resnet':
        return make_resnet50(NUM_CLASSES)
    elif name == 'densenet':
        return make_densenet(NUM_CLASSES, cfg.get('dropout_prob', 0.5))
    elif name == 'googlenet':
        return GoogLeNet(num_classes=NUM_CLASSES, aux_logits=True)
    elif name == 'pretrained_googlenet':
        return make_pretrained_googlenet(NUM_CLASSES)
    elif name == 'pretrained_densenet':
        return make_pretrained_densenet(NUM_CLASSES)
    else:
        raise ValueError(f'Unknown model: {name}')


def main():
    parser = argparse.ArgumentParser(description='Train HAR CNN models on Stanford 40 Actions')
    parser.add_argument('--model', type=str, default='vgg',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model architecture to train')
    parser.add_argument('--data_dir', type=str, default='JPEGImages',
                        help='Root directory of JPEGImages dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_loader, val_loader, test_loader, _ = get_dataloaders(args.data_dir, args.batch_size)

    cfg = MODEL_CONFIGS[args.model]
    model = build_model(args.model, cfg)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0))

    print(f'\nModel: {args.model.upper()}  |  lr={cfg["lr"]}  |  epochs={cfg["num_epochs"]}\n')
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        CLASSES, device, num_epochs=cfg['num_epochs'],
    )

    avg_loss, accuracy, all_preds, all_labels, report, precision, recall, f1 = evaluate_model(
        model, test_loader, CLASSES, device, criterion,
    )

    print(f'\nTest Loss: {avg_loss:.4f}  |  Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}')
    print('\nClassification Report:\n', report)

    plot_metrics(train_losses, val_losses, train_accs, val_accs, model_name=args.model)
    plot_confusion_matrix(all_labels, all_preds, CLASSES, model_name=args.model)

    save_path = f'{args.model}_har.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Saved model weights to {save_path}')


if __name__ == '__main__':
    main()
