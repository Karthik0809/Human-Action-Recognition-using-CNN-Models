import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(BASE_DIR, "data", "JPEGImages")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# ── Data split ────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15   # test = remaining 0.10

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE          = 32
SEED                = 42
EARLY_STOP_PATIENCE = 10   # increased — cosine schedule needs more room

# ── Inference ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.50   # below this → "Unknown" in predict.py

# ── Per-model hyperparameters ─────────────────────────────────────────────────
#
# Two-stage fine-tuning strategy (all pretrained-backbone models):
#   Stage 1  — freeze backbone, train head only with high LR (fast convergence).
#   Stage 2  — unfreeze all, fine-tune with low LR + cosine annealing.
#
# This typically adds 3-5 percentage points over single-stage training.
#
MODEL_CONFIGS = {
    "vgg16": {
        # Stage 1: head warmup
        "lr_head":         1e-3,
        "epochs_head":     5,
        # Stage 2: full fine-tune
        "lr_backbone":     5e-5,
        "weight_decay":    1e-4,
        "num_epochs":      45,   # cosine T_max
        "mixup_alpha":     0.3,  # Mixup interpolation strength (0 = off)
    },
    "resnet50": {
        "lr_head":         1e-3,
        "epochs_head":     5,
        "lr_backbone":     5e-5,
        "weight_decay":    1e-4,
        "num_epochs":      45,
        "mixup_alpha":     0.4,
    },
    "densenet": {
        "lr_head":         1e-3,
        "epochs_head":     5,
        "lr_backbone":     3e-5,
        "weight_decay":    1e-4,
        "num_epochs":      45,
        "mixup_alpha":     0.4,
    },
    "googlenet": {
        "lr_head":         1e-3,
        "epochs_head":     5,
        "lr_backbone":     5e-5,
        "weight_decay":    1e-4,
        "num_epochs":      45,
        "mixup_alpha":     0.3,
    },
    "pretrained_googlenet": {
        "lr_head":         1e-3,
        "epochs_head":     5,
        "lr_backbone":     5e-5,
        "weight_decay":    1e-4,
        "num_epochs":      45,
        "mixup_alpha":     0.3,
    },
    "pretrained_densenet": {
        "lr_head":         1e-3,
        "epochs_head":     5,
        "lr_backbone":     3e-5,
        "weight_decay":    1e-4,
        "num_epochs":      45,
        "mixup_alpha":     0.4,
    },
}
