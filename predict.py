"""
predict.py — Single-image inference with Test-Time Augmentation and ensemble support.

Features:
  • Loads any saved checkpoint from checkpoints/
  • Test-Time Augmentation (TTA) — averages predictions over 4 augmented views (+1-2% accuracy)
  • Ensemble — average predictions across multiple models
  • Confidence threshold — labels prediction "Unknown" if max prob < threshold

Usage:
    # Single model, single image
    python predict.py --image photo.jpg --model pretrained_densenet

    # With TTA enabled
    python predict.py --image photo.jpg --model resnet50 --tta

    # Ensemble of multiple models
    python predict.py --image photo.jpg --ensemble resnet50 pretrained_densenet vgg16

    # Stricter unknown threshold, show top-5
    python predict.py --image photo.jpg --model vgg16 --threshold 0.6 --top_k 5
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image
from torchvision import transforms

import config
from models.vgg16 import make_vgg16
from models.resnet import make_resnet50
from models.densenet import make_densenet
from models.googlenet import make_googlenet

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

ALL_MODELS = ["vgg16", "resnet50", "densenet", "googlenet",
              "pretrained_googlenet", "pretrained_densenet"]


# ── Transforms ────────────────────────────────────────────────────────────────

def _val_tfm():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
    ])

def _tta_tfms():
    """Four deterministic augmented views — averaged for better accuracy."""
    norm = [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    return [
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]           + norm),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.RandomHorizontalFlip(p=1.0)]                       + norm),
        transforms.Compose([transforms.Resize(280), transforms.CenterCrop(224)]           + norm),
        transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224)]           + norm),
    ]


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(model_name: str, device: torch.device):
    path = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(path):
        print(f"[ERROR] No checkpoint: '{path}'.  Run: python train.py --model {model_name}")
        sys.exit(1)
    ckpt        = torch.load(path, map_location=device)
    classes     = ckpt["classes"]
    num_classes = len(classes)

    if model_name == "vgg16":
        model = make_vgg16(num_classes)
    elif model_name == "resnet50":
        model = make_resnet50(num_classes)
    elif model_name == "densenet":
        model = make_densenet(num_classes=num_classes)
    elif model_name == "googlenet":
        model = make_googlenet(num_classes=num_classes)
    elif model_name == "pretrained_googlenet":
        model = tv_models.googlenet(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "pretrained_densenet":
        model = tv_models.densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded '{model_name}'  (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    return model, classes


# ── Inference helpers ─────────────────────────────────────────────────────────

def _get_probs(model, img: Image.Image, device: torch.device, use_tta: bool) -> torch.Tensor:
    """Return averaged softmax probabilities (num_classes,) for one PIL image."""
    tfms = _tta_tfms() if use_tta else [_val_tfm()]
    probs_sum = None
    with torch.no_grad():
        for tfm in tfms:
            t = tfm(img).unsqueeze(0).to(device)
            p = F.softmax(model(t), dim=1).squeeze(0)
            probs_sum = p if probs_sum is None else probs_sum + p
    return probs_sum / len(tfms)


# ── Public API ────────────────────────────────────────────────────────────────

def predict(image_path: str, model_name: str = None, ensemble: list = None,
            threshold: float = None, top_k: int = 3, use_tta: bool = False):
    """
    Predict the action class for a single image.

    Args:
        image_path:  Path to the image file.
        model_name:  Single model name (ignored if ensemble is set).
        ensemble:    List of model names to ensemble.
        threshold:   Confidence below which result is "Unknown".
        top_k:       How many predictions to display.
        use_tta:     Enable Test-Time Augmentation.

    Returns:
        dict: predicted_class, confidence, unknown, top_k_predictions
    """
    if threshold is None:
        threshold = config.CONFIDENCE_THRESHOLD

    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: '{image_path}'")
        sys.exit(1)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img     = Image.open(image_path).convert("RGB")
    models  = ensemble if ensemble else [model_name]

    probs_sum = None
    classes   = None
    for name in models:
        model, classes = load_checkpoint(name, device)
        p = _get_probs(model, img, device, use_tta)
        probs_sum = p if probs_sum is None else probs_sum + p

    probs = probs_sum / len(models)
    top_probs, top_idxs = torch.topk(probs, k=min(top_k, len(classes)))

    best_conf  = top_probs[0].item()
    best_class = classes[top_idxs[0].item()]
    is_unknown = best_conf < threshold

    # ── Output ──────────────────────────────────────────────────────────────
    label = "Ensemble" if (ensemble and len(ensemble) > 1) else model_name
    tta_tag = " + TTA" if use_tta else ""
    print(f"\nImage : {image_path}")
    print(f"Model : {label}{tta_tag}")
    print("-" * 50)
    if is_unknown:
        print(f"Result : UNKNOWN  (max confidence {best_conf*100:.1f}% < threshold {threshold*100:.0f}%)")
        print( "         Image does not clearly match any known action class.")
    else:
        print(f"Result : {best_class}  ({best_conf*100:.1f}% confidence)")

    print(f"\nTop-{top_k} predictions:")
    for rank, (prob, idx) in enumerate(zip(top_probs, top_idxs), 1):
        bar = "█" * int(prob.item() * 30)
        print(f"  {rank}. {classes[idx.item()]:<35} {prob.item()*100:5.1f}%  {bar}")
    print()

    return {
        "predicted_class":    best_class if not is_unknown else "Unknown",
        "confidence":         best_conf,
        "unknown":            is_unknown,
        "top_k_predictions":  [{"class": classes[i.item()], "confidence": p.item()}
                                for p, i in zip(top_probs, top_idxs)],
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict action class for a single image")
    parser.add_argument("--image",     required=True)
    parser.add_argument("--model",     choices=ALL_MODELS, default=None,
                        help="Single model checkpoint to use")
    parser.add_argument("--ensemble",  nargs="+", choices=ALL_MODELS, default=None,
                        help="Ensemble of model checkpoints (space-separated)")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Confidence threshold (default {config.CONFIDENCE_THRESHOLD})")
    parser.add_argument("--top_k",     type=int,   default=3)
    parser.add_argument("--tta",       action="store_true",
                        help="Enable Test-Time Augmentation (4 augmented views averaged)")
    args = parser.parse_args()

    if args.ensemble is None and args.model is None:
        parser.error("Provide --model or --ensemble.")

    predict(args.image, model_name=args.model, ensemble=args.ensemble,
            threshold=args.threshold, top_k=args.top_k, use_tta=args.tta)


if __name__ == "__main__":
    main()
