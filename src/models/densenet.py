import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class _Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([out, x], 1)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.avgpool(self.conv(F.relu(self.bn(x))))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, num_classes=40, dropout_prob=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = nn.Sequential(*[
                _Bottleneck(num_features + j * growth_rate, growth_rate)
                for j in range(num_layers)
            ])
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                self.features.add_module(f'transition{i + 1}', _Transition(num_features, num_features // 2))
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(F.relu(features, inplace=True), (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(self.dropout(out))


def make_densenet(num_classes=40, dropout_prob=0.5):
    """DenseNet-121 architecture trained from scratch."""
    return DenseNet(num_classes=num_classes, dropout_prob=dropout_prob)


def make_pretrained_densenet(num_classes):
    """DenseNet-121 with ImageNet pretrained weights, classifier replaced for num_classes."""
    model = tv_models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model
