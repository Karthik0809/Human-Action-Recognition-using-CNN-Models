"""
models/densenet.py — DenseNet-121 with ImageNet pretrained weights.

Using a pretrained backbone dramatically improves accuracy on small datasets
like Stanford 40 (~9k images, 40 classes) compared to training from scratch.
The classifier head is replaced with a new linear layer for the target classes.
"""
import torch.nn as nn
import torchvision.models as tv_models


def make_densenet(num_classes=40, dropout_prob=0.5):
    """
    DenseNet-121 with ImageNet pretrained backbone.
    The classifier head is replaced for the target number of classes.
    """
    try:
        # torchvision >= 0.13
        model = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
    except AttributeError:
        # torchvision < 0.13
        model = tv_models.densenet121(pretrained=True)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_prob),
        nn.Linear(in_features, num_classes),
    )
    return model
