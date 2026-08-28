import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class CustomImageDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_train_transform():
    """
    Strong augmentation pipeline for fine-tuning pretrained models.

    Key additions vs. the old pipeline:
      • AutoAugment (ImageNet policy) — policy-search-derived transforms that
        consistently add 2-3% accuracy on classification benchmarks.
      • RandomResizedCrop with wider scale range (0.5-1.0) — more occlusion
        and zoom variation.
      • ColorJitter kept for domain-specific variation.
      • RandomErasing kept at p=0.3 to simulate partial occlusion.
    """
    aug_list = [
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    ]

    # AutoAugment: available since torchvision 0.10
    try:
        aug_list.append(
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
        )
    except AttributeError:
        # Fallback for older torchvision: just add rotation
        aug_list.append(transforms.RandomRotation(20))

    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
    ]

    return transforms.Compose(aug_list)


def get_val_transform():
    """Deterministic transform for validation and test — no randomness."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_inference_transform():
    """Single-image inference transform (same as val)."""
    return get_val_transform()


def get_tta_transforms():
    """
    Five deterministic transforms for Test-Time Augmentation.
    Predictions averaged over all five — typically adds 1-2% accuracy.
    """
    norm = [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    return [
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]  + norm),
        transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224)]  + norm),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.RandomHorizontalFlip(p=1.0)]              + norm),
        transforms.Compose([transforms.Resize(280), transforms.CenterCrop(224)]  + norm),
        transforms.Compose([transforms.Resize(232), transforms.CenterCrop(224)]  + norm),
    ]


def get_data_loaders(data_dir, batch_size=32, train_ratio=0.75, val_ratio=0.15, seed=42):
    """
    Build train / val / test DataLoaders.
    Train split gets augmented transforms; val/test get deterministic transforms.

    Returns: train_loader, val_loader, test_loader, classes (sorted list)
    """
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    filepaths, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            fp = os.path.join(cls_dir, fname)
            if os.path.isfile(fp):
                filepaths.append(fp)
                labels.append(cls)

    total      = len(filepaths)
    train_size = int(train_ratio * total)
    val_size   = int(val_ratio   * total)

    random.seed(seed)
    idx = list(range(total))
    random.shuffle(idx)

    train_idx = idx[:train_size]
    val_idx   = idx[train_size:train_size + val_size]
    test_idx  = idx[train_size + val_size:]

    def _make_ds(indices, tfm):
        return CustomImageDataset(
            [filepaths[i] for i in indices],
            [labels[i]    for i in indices],
            transform=tfm,
        )

    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(_make_ds(train_idx, get_train_transform()),
                              batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(_make_ds(val_idx,   get_val_transform()),
                              batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(_make_ds(test_idx,  get_val_transform()),
                              batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader, classes
