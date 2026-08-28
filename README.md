# Human Action Recognition
### Stanford 40 Actions — Multi-Model Comparison

Classifies 40 human actions (Applauding, Climbing, Cooking, Running …) using four deep learning architectures —
one built from scratch, three using ImageNet-pretrained weights fine-tuned on Stanford 40.

---

## Models and Results

Ranked by best validation accuracy. Full per-epoch logs are in `training_*.log`.

| # | Model | Type | Params | Val Accuracy |
|---|-------|------|--------|--------------|
| 1 | DenseNet-121 (torchvision) | ImageNet pretrained + fine-tune | ~8 M | **87.54%** |
| 2 | ResNet-50 | Pretrained backbone + new head | ~25 M | 86.91% |
| 3 | DenseNet (custom) | From scratch | ~7 M | 86.77% |
| 4 | VGG-16 | Pretrained backbone + new head | ~138 M | 81.60% |

> **Note:** GoogLeNet implementations (`models/googlenet.py`) are present in the repo but are **not part of the
> benchmark** — both the custom and torchvision variants fail to initialise under the installed torchvision
> version with `ValueError: The parameter 'aux_logits' expected value True but got False instead`.
> See `training_googlenet.log` and `training_pretrained_googlenet.log`. Fixing this is open work.

---

## Dataset

**Stanford 40 Actions** — 9 532 images across 40 action categories.  
Official site: http://vision.stanford.edu/Datasets/40actions.html

The download script fetches and extracts it automatically.

---

## Project Structure

```
Human Action Recognition/
│
├── models/
│   ├── vgg16.py           — VGG-16 from scratch + pretrained loader
│   ├── resnet.py          — ResNet-50 (Bottleneck) + pretrained loader
│   ├── densenet.py        — DenseNet from scratch
│   └── googlenet.py       — GoogLeNet (Inception modules) — see note above, not benchmarked
│
├── utils/
│   ├── dataset.py         — Dataset class, augmented transforms, data loaders
│   ├── visualization.py   — Training curves, confusion matrix, classification report
│   └── gradcam.py         — Grad-CAM heatmap generator + overlay visualizer
│
├── data/                  — Dataset downloaded here (auto-created)
├── checkpoints/           — Best model weights saved here during training
│
├── config.py              — All paths and hyperparameters in one place
├── download_dataset.py    — Downloads + extracts the Stanford 40 dataset
├── train.py               — Full training pipeline (CLI + importable functions)
├── predict.py             — Single-image inference with TTA and ensemble
├── main.ipynb             — Jupyter notebook — step-by-step walkthrough
└── requirements.txt       — Python package dependencies
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset (~1 GB)

```bash
python download_dataset.py
```

This downloads `Stanford40_JPEGImages.tgz`, extracts it to `data/JPEGImages/`, and removes the archive.

---

## Training

```bash
# Train all 4 benchmarked models sequentially
python train.py

# Train a single model
python train.py --model vgg16
python train.py --model resnet50
python train.py --model densenet
python train.py --model pretrained_densenet

# Resume an interrupted training run
python train.py --model resnet50 --resume

# Custom batch size
python train.py --model pretrained_densenet --batch_size 16
```

Checkpoints are saved to `checkpoints/<model>_best.pth` at every epoch that improves validation accuracy.

---

## Prediction

```bash
# Single model
python predict.py --image photo.jpg --model pretrained_densenet

# With Test-Time Augmentation (+1-2% accuracy, 4x slower)
python predict.py --image photo.jpg --model resnet50 --tta

# Ensemble of multiple models (most accurate)
python predict.py --image photo.jpg --ensemble pretrained_densenet resnet50 densenet --tta

# Show top-5 and use a stricter unknown threshold
python predict.py --image photo.jpg --model vgg16 --top_k 5 --threshold 0.6
```

If the model's confidence is below `--threshold` (default 0.50), the result is reported as **UNKNOWN**.

---

## Notebook

Open `main.ipynb` in Jupyter or VS Code.  
Sections:
1. Install + download  
2. Data loaders  
3–6. Train and evaluate each model  
7. Grad-CAM visualisation  
8. Predict on new images (single / TTA / ensemble)

---

## Training Improvements

Compared to the original Google Colab notebook:

| Feature | Original | This project |
|---------|----------|-------------|
| Augmentation | Resize + ToTensor only | RandomResizedCrop, Flip, Rotation, ColorJitter, **RandomErasing** |
| Val/test transforms | Same as train | Separate deterministic (Resize 256 → CenterCrop 224) |
| LR schedule | Fixed | **ReduceLROnPlateau** (halves on plateau) |
| Loss | CrossEntropyLoss | **Label smoothing** (0.1) |
| Gradient stability | None | **Gradient clipping** (max_norm=5) |
| Overfitting guard | None | **Early stopping** (patience=10) |
| Best model | Last epoch | **Checkpointed** at best val accuracy |
| GPU speed | Standard | **Mixed Precision (AMP)** — ~2x faster |
| Inference | Plain forward | **TTA** + **Ensemble** options |
| Explainability | None | **Grad-CAM** visualizations |

---

## Hyperparameters

See `config.py` to adjust learning rates, epochs, batch size, early stopping patience, and the confidence threshold for unknown prediction.

---

## Requirements

- Python 3.8+
- PyTorch ≥ 1.10  
- torchvision ≥ 0.11  
- Pillow, NumPy, Matplotlib, Seaborn, scikit-learn  

GPU with CUDA is recommended but not required (CPU training is slow for the larger models).
