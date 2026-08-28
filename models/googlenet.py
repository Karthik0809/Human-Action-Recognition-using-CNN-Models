"""
models/googlenet.py — GoogLeNet with ImageNet pretrained weights.

Using a pretrained backbone dramatically improves accuracy on small datasets
like Stanford 40 (~9k images, 40 classes) compared to training from scratch.
The fully-connected head is replaced with a new linear layer for the target classes.
"""
import torch.nn as nn
import torchvision.models as tv_models


def make_googlenet(num_classes=40):
    """
    GoogLeNet with ImageNet pretrained backbone (no auxiliary classifiers).
    The fc head is replaced for the target number of classes.
    """
    try:
        # torchvision >= 0.13
        model = tv_models.googlenet(
            weights=tv_models.GoogLeNet_Weights.IMAGENET1K_V1,
            aux_logits=False,
        )
    except AttributeError:
        # torchvision < 0.13
        model = tv_models.googlenet(pretrained=True, aux_logits=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Alias kept so any code that does `from models.googlenet import GoogLeNet` still works.
GoogLeNet = make_googlenet
