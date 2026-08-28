"""
train.py — Two-stage fine-tuning for Human Action Recognition on Stanford 40.

Key improvements over the original single-stage approach:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ STAGE 1 – Head warmup (backbone frozen)                                 │
  │   Freeze all backbone weights. Train only the classification head with  │
  │   a high LR (1e-3) for a few epochs. This gives the random head a good │
  │   starting point without destroying pretrained backbone features.       │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ STAGE 2 – Full fine-tuning (all layers unfrozen)                        │
  │   Unfreeze everything. Train with a low LR + AdamW + CosineAnnealingLR │
  │   + label smoothing + Mixup. Cosine decay reaches near-zero at the end  │
  │   which consistently outperforms fixed or plateau-based schedules.      │
  └─────────────────────────────────────────────────────────────────────────┘

  Other improvements:
    • Mixup augmentation — interpolates pairs of images/labels in each batch
    • AdamW optimizer    — decoupled weight decay (better generalisation)
    • AMP (mixed precision) — ~2x faster on GPU, no accuracy loss
    • Gradient clipping  — prevents exploding gradients
    • Early stopping     — patience=10 on val accuracy
    • Best-model checkpointing with val_acc stored

Usage:
    python train.py                             # all 6 models
    python train.py --model resnet50
    python train.py --model densenet --resume   # resume stage-2 from checkpoint
    python train.py --batch_size 16
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as tv_models

import config
from utils.dataset import get_data_loaders
from utils.visualization import plot_training_curves, plot_confusion_matrix, print_classification_report
from models.vgg16 import make_vgg16
from models.resnet import make_resnet50
from models.densenet import make_densenet
from models.googlenet import make_googlenet


# ── Utilities ──────────────────────────────────────────────────────────────────

def _count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model, model_name: str):
    """Freeze backbone weights so only the head trains in Stage 1."""
    if model_name == "vgg16":
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_name == "resnet50":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    elif model_name in ("densenet", "pretrained_densenet"):
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_name in ("googlenet", "pretrained_googlenet"):
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ── Mixup ──────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha: float, device):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, classes, device,
                    scaler, use_amp, mixup_alpha=0.0):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for images, label_names in loader:
        inputs = images.to(device, non_blocking=True)
        labels = torch.tensor([classes.index(l) for l in label_names], device=device)

        inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha, device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        run_loss += loss.item() * inputs.size(0)
        _, pred   = torch.max(outputs, 1)
        total    += labels.size(0)
        # accuracy against the dominant label (y_a) — standard practice with Mixup
        correct  += (pred == y_a).sum().item()

    return run_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, classes, device, use_amp):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, label_names in loader:
        inputs = images.to(device, non_blocking=True)
        labels = torch.tensor([classes.index(l) for l in label_names], device=device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
        val_loss += loss.item() * inputs.size(0)
        _, pred   = torch.max(outputs, 1)
        total    += labels.size(0)
        correct  += (pred == labels).sum().item()
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return val_loss / len(loader.dataset), correct / total, all_preds, all_labels


def run_stage(tag, model, train_loader, val_loader, criterion, optimizer,
              classes, device, num_epochs, scheduler=None,
              patience=None, checkpoint_path=None,
              best_val_acc=0.0, mixup_alpha=0.0):
    """
    One training stage.  Returns (model, train_losses, val_losses,
    train_accs, val_accs, best_val_acc).
    """
    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_losses, val_losses         = [], []
    train_accuracies, val_accuracies = [], []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            classes, device, scaler, use_amp, mixup_alpha
        )
        v_loss, v_acc, _, _ = evaluate(
            model, val_loader, criterion, classes, device, use_amp
        )

        train_losses.append(t_loss); train_accuracies.append(t_acc)
        val_losses.append(v_loss);   val_accuracies.append(v_acc)

        lr = optimizer.param_groups[0]['lr']
        print(f"  [{tag}] Ep {epoch+1:3d}/{num_epochs}  lr={lr:.2e}  "
              f"Train {t_acc:.4f}  Val {v_acc:.4f}")

        if scheduler is not None:
            scheduler.step()

        if checkpoint_path and v_acc > best_val_acc:
            best_val_acc      = v_acc
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "epoch":            epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc":          best_val_acc,
                "classes":          classes,
            }, checkpoint_path)
            print(f"    ** Best saved  val_acc={best_val_acc:.4f}")
        else:
            epochs_no_improve += 1

        if patience and epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}.")
            break

    return model, train_losses, val_losses, train_accuracies, val_accuracies, best_val_acc


# ── Model builder ──────────────────────────────────────────────────────────────

def _build_model(model_name: str, num_classes: int):
    if model_name == "vgg16":
        return make_vgg16(num_classes)
    elif model_name == "resnet50":
        return make_resnet50(num_classes)
    elif model_name == "densenet":
        return make_densenet(num_classes=num_classes)
    elif model_name == "googlenet":
        return make_googlenet(num_classes=num_classes)
    elif model_name == "pretrained_googlenet":
        try:
            m = tv_models.googlenet(
                weights=tv_models.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=False)
        except AttributeError:
            m = tv_models.googlenet(pretrained=True, aux_logits=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif model_name == "pretrained_densenet":
        try:
            m = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
        except AttributeError:
            m = tv_models.densenet121(pretrained=True)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Evaluation helper ──────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, classes, device, criterion):
    use_amp = device.type == "cuda"
    loss, acc, preds, lbls = evaluate(model, test_loader, criterion, classes, device, use_amp)
    print(f"Test Loss: {loss:.4f}  Test Accuracy: {acc:.4f}")
    return loss, acc, preds, lbls


# ── Full two-stage run ─────────────────────────────────────────────────────────

def run(model_name, train_loader, val_loader, test_loader, classes, device, resume=False):
    num_classes = len(classes)
    cfg         = config.MODEL_CONFIGS[model_name]
    ckpt_path   = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_best.pth")

    print(f"\n{'='*65}")
    print(f"  {model_name.upper()}")
    print(f"{'='*65}")

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    except TypeError:
        criterion = nn.CrossEntropyLoss()

    model = _build_model(model_name, num_classes)
    model.to(device)

    total_p, _ = _count_params(model)
    print(f"  Parameters: {total_p:,} total")

    best_val_acc = 0.0

    # Resume: load existing checkpoint and skip to stage-2 (or continue from saved acc)
    if resume and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            best_val_acc = ckpt.get("val_acc", 0.0)
            print(f"  Resumed  val_acc={best_val_acc:.4f} — skipping to Stage 2")
        except Exception as e:
            print(f"  Could not load checkpoint ({e}) — starting fresh")
            best_val_acc = 0.0
    else:
        # ── STAGE 1: Head warmup ───────────────────────────────────────────────
        epochs_head = cfg.get("epochs_head", 5)
        lr_head     = cfg.get("lr_head", 1e-3)

        print(f"\n  [Stage 1] Freeze backbone  |  epochs={epochs_head}  lr={lr_head:.0e}")
        freeze_backbone(model, model_name)

        _, trainable_s1 = _count_params(model)
        print(f"  Trainable params (head only): {trainable_s1:,}")

        opt_head = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_head, weight_decay=1e-4,
        )

        model, *_, best_val_acc = run_stage(
            tag="S1", model=model,
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=opt_head,
            classes=classes, device=device,
            num_epochs=epochs_head,
            checkpoint_path=ckpt_path,
            best_val_acc=best_val_acc,
            mixup_alpha=0.0,   # no Mixup in warmup — head needs clean labels
        )

    # ── STAGE 2: Full fine-tuning ──────────────────────────────────────────────
    epochs_full   = cfg.get("num_epochs", 45)
    lr_backbone   = cfg.get("lr_backbone", 5e-5)
    weight_decay  = cfg.get("weight_decay", 1e-4)
    mixup_alpha   = cfg.get("mixup_alpha", 0.3)

    print(f"\n  [Stage 2] Unfreeze all  |  epochs={epochs_full}  lr={lr_backbone:.0e}  "
          f"mixup={mixup_alpha}")
    unfreeze_all(model)

    _, trainable_s2 = _count_params(model)
    print(f"  Trainable params (all): {trainable_s2:,}")

    opt_full  = optim.AdamW(model.parameters(), lr=lr_backbone, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(opt_full, T_max=epochs_full, eta_min=1e-7)

    model, tl, vl, ta, va, best_val_acc = run_stage(
        tag="S2", model=model,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=opt_full,
        classes=classes, device=device,
        num_epochs=epochs_full,
        scheduler=scheduler,
        patience=config.EARLY_STOP_PATIENCE,
        checkpoint_path=ckpt_path,
        best_val_acc=best_val_acc,
        mixup_alpha=mixup_alpha,
    )

    # Load best weights
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n  Loaded best checkpoint  val_acc={ckpt['val_acc']:.4f}")

    plot_training_curves(tl, vl, ta, va, model_name=model_name)
    _, _, preds, lbls = evaluate_model(model, test_loader, classes, device, criterion)
    plot_confusion_matrix(lbls, preds, classes, model_name=model_name)
    print_classification_report(lbls, preds, classes, model_name=model_name)
    return model


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train human action recognition models")
    parser.add_argument("--model", default="all",
                        choices=["all", "vgg16", "resnet50", "densenet", "googlenet",
                                 "pretrained_googlenet", "pretrained_densenet"])
    parser.add_argument("--data_dir",   default=config.DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume stage-2 from existing checkpoint (skip stage-1)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  AMP: {'enabled' if device.type == 'cuda' else 'disabled (CPU)'}")

    train_loader, val_loader, test_loader, classes = get_data_loaders(
        args.data_dir, batch_size=args.batch_size,
        train_ratio=config.TRAIN_RATIO, val_ratio=config.VAL_RATIO, seed=config.SEED
    )

    names = (["vgg16", "resnet50", "densenet", "googlenet",
               "pretrained_googlenet", "pretrained_densenet"]
             if args.model == "all" else [args.model])

    for name in names:
        run(name, train_loader, val_loader, test_loader, classes, device, resume=args.resume)


if __name__ == "__main__":
    main()
