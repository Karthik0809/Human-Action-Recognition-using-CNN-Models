import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image

from config import CLASSES, DATA_DIR, IMG_SIZE, MEAN, STD, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO


class HARDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_data(data_dir=DATA_DIR, classes=CLASSES):
    filepaths, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            filepaths.append(os.path.join(cls_dir, fname))
            labels.append(cls)
    return filepaths, labels


def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    filepaths, labels = load_data(data_dir)
    dataset = HARDataset(filepaths, labels, transform=build_transform())

    total = len(filepaths)
    train_size = int(TRAIN_RATIO * total)
    val_size = int(VAL_RATIO * total)

    indices = list(range(total))
    random.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))

    print(f'Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}')
    return train_loader, val_loader, test_loader, dataset
