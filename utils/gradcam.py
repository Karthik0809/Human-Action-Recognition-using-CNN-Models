"""
utils/gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM).

Grad-CAM highlights the image regions the model focuses on when making a prediction.
Useful for understanding and comparing how different architectures attend to actions.

Usage (in notebook):
    from utils.gradcam import GradCAM, visualize_gradcam
    from models.resnet import make_resnet50

    model, classes = load_checkpoint('resnet50', device)
    cam = GradCAM(model, target_layer=model.layer4)
    visualize_gradcam(cam, 'path/to/image.jpg', classes, predicted_class_idx)
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class GradCAM:
    """
    Generic Grad-CAM implementation.  Works with any model as long as you
    specify the target convolutional layer.

    Recommended target layers per architecture:
        VGG-16               : model.features[-3]          (last conv block)
        ResNet-50            : model.layer4
        DenseNet-121         : model.features.denseblock4
        GoogLeNet            : model.inception5b
        Pretrained GoogLeNet : model.inception5b
        Pretrained DenseNet  : model.features.denseblock4
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self._activations = output.detach()

        def backward_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Compute the Grad-CAM heatmap for a single image tensor (1, C, H, W).

        Args:
            input_tensor: Preprocessed image tensor on the same device as the model.
            class_idx:    Class index to explain. If None, uses the top predicted class.

        Returns:
            cam: numpy array (H, W) with values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward on the selected class score
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Pool gradients over spatial dimensions → importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        """Call when done to avoid memory leaks in long-running notebooks."""
        for h in getattr(self, '_hook_handles', []):
            h.remove()


# ── Visualisation helper ──────────────────────────────────────────────────────

def _preprocess(image_path: str, device: torch.device):
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    img = Image.open(image_path).convert('RGB')
    return tfm(img).unsqueeze(0).to(device), img


def visualize_gradcam(cam_obj: GradCAM, image_path: str, classes: list,
                      class_idx: int = None, alpha: float = 0.5,
                      model_name: str = ""):
    """
    Overlay a Grad-CAM heatmap on the original image and display it.

    Args:
        cam_obj:    A GradCAM instance.
        image_path: Path to the input image.
        classes:    List of class names.
        class_idx:  Class to explain (None → top predicted class).
        alpha:      Heatmap overlay transparency (0=image only, 1=heatmap only).
        model_name: Used in the plot title.
    """
    device = next(cam_obj.model.parameters()).device
    tensor, orig_img = _preprocess(image_path, device)

    cam = cam_obj.generate(tensor, class_idx)

    if class_idx is None:
        with torch.no_grad():
            logits    = cam_obj.model(tensor)
            class_idx = logits.argmax(dim=1).item()
            conf      = torch.softmax(logits, dim=1)[0, class_idx].item()
    else:
        conf = None

    # Build overlay
    orig_np  = np.array(orig_img.resize((224, 224))) / 255.0
    heatmap  = plt.cm.jet(cam)[..., :3]               # (H, W, 3) RGB
    overlay  = (1 - alpha) * orig_np + alpha * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(orig_np);  axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(cam, cmap='jet'); axes[1].set_title('Grad-CAM'); axes[1].axis('off')
    axes[2].imshow(np.clip(overlay, 0, 1)); axes[2].axis('off')

    title = f"{model_name} → {classes[class_idx]}"
    if conf is not None:
        title += f"  ({conf*100:.1f}%)"
    axes[2].set_title(title)
    plt.tight_layout(); plt.show()
    return cam


def get_target_layer(model, model_name: str):
    """Convenience: return the recommended Grad-CAM layer for each architecture."""
    targets = {
        "vgg16":               lambda m: m.features[-3],
        "resnet50":            lambda m: m.layer4,
        "densenet":            lambda m: m.features.denseblock4,
        "googlenet":           lambda m: m.inception5b,
        "pretrained_googlenet":lambda m: m.inception5b,
        "pretrained_densenet": lambda m: m.features.denseblock4,
    }
    if model_name not in targets:
        raise ValueError(f"No default target layer for '{model_name}'. "
                         f"Pass target_layer directly to GradCAM().")
    return targets[model_name](model)
