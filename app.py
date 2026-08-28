"""
app.py — Streamlit UI for Human Action Recognition.

Run with:
    streamlit run app.py

Features:
  • Loads saved checkpoints — NO retraining required.
  • Single model or multi-model ensemble inference.
  • Test-Time Augmentation (TTA) toggle.
  • Confidence threshold → "Unknown" label for out-of-distribution images.
  • Top-K predictions with animated confidence bars.
  • Grad-CAM heatmap overlay to visualise what the model attends to.
  • All models cached in session so subsequent images are instant.
"""

import io
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import streamlit as st

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config
from models.vgg16 import make_vgg16
from models.resnet import make_resnet50
from models.densenet import make_densenet
from models.googlenet import make_googlenet
from utils.gradcam import GradCAM, get_target_layer

matplotlib.use("Agg")   # no display needed in Streamlit

# ── constants ─────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

ALL_MODELS = [
    "vgg16", "resnet50", "densenet", "googlenet",
    "pretrained_googlenet", "pretrained_densenet",
]
MODEL_LABELS = {
    "vgg16":               "VGG-16",
    "resnet50":            "ResNet-50",
    "densenet":            "DenseNet-121",
    "googlenet":           "GoogLeNet",
    "pretrained_googlenet":"GoogLeNet (ImageNet fine-tune)",
    "pretrained_densenet": "DenseNet (ImageNet fine-tune)",
}

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Human Action Recognition",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font smoothing */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Prediction result card */
.result-card {
    padding: 1.5rem 2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    text-align: center;
}
.result-known {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    color: white;
}
.result-unknown {
    background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
    color: white;
}
.result-card h1 { font-size: 2rem; margin: 0.2rem 0; }
.result-card p  { font-size: 1rem; opacity: 0.85; margin: 0; }

/* Confidence bar label */
.bar-row { display: flex; align-items: center; margin-bottom: 6px; }
.bar-label { width: 200px; font-size: 0.85rem; white-space: nowrap;
             overflow: hidden; text-overflow: ellipsis; }
.bar-pct { margin-left: 8px; font-size: 0.85rem; color: #888; }

/* Section headers */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    color: #e2e8f0;
}

/* Checkpoint status pills */
.pill-ok   { display:inline-block; background:#166534; color:#d1fae5;
             border-radius:9999px; padding:2px 10px; font-size:0.75rem; }
.pill-miss { display:inline-block; background:#7f1d1d; color:#fee2e2;
             border-radius:9999px; padding:2px 10px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def checkpoint_path(model_name: str) -> str:
    return os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_best.pth")


def available_models() -> list[str]:
    return [m for m in ALL_MODELS if os.path.exists(checkpoint_path(m))]


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load and cache a model from its checkpoint. Never retrains."""
    path = checkpoint_path(model_name)
    if not os.path.exists(path):
        return None, None, None, f"Checkpoint not found: {path}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
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
            try:
                model = tv_models.googlenet(
                    weights=tv_models.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=False)
            except AttributeError:
                model = tv_models.googlenet(pretrained=False, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "pretrained_densenet":
            try:
                model = tv_models.densenet121(
                    weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
            except AttributeError:
                model = tv_models.densenet121(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            return None, None, None, f"Unknown model: {model_name}"

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        meta = {"epoch": ckpt.get("epoch", "?"), "val_acc": ckpt.get("val_acc", None)}
        return model, classes, device, None, meta
    except Exception as exc:
        return None, None, None, str(exc), {}


def _val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _tta_transforms():
    norm = [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
    return [
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)] + norm),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.RandomHorizontalFlip(p=1.0)] + norm),
        transforms.Compose([transforms.Resize(280), transforms.CenterCrop(224)] + norm),
        transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224)] + norm),
    ]


def get_probs(model, img: Image.Image, device, use_tta: bool) -> torch.Tensor:
    tfms = _tta_transforms() if use_tta else [_val_transform()]
    total = None
    with torch.no_grad():
        for tfm in tfms:
            t = tfm(img).unsqueeze(0).to(device)
            p = F.softmax(model(t), dim=1).squeeze(0)
            total = p if total is None else total + p
    return total / len(tfms)


def run_inference(image: Image.Image, model_names: list[str],
                  use_tta: bool, threshold: float, top_k: int):
    """Run single or ensemble inference. Returns result dict or (None, error_str)."""
    probs_sum = None
    classes   = None
    loaded    = []

    for name in model_names:
        result = load_model(name)
        model, cls, device, err = result[0], result[1], result[2], result[3]
        if err:
            return None, f"Could not load {name}: {err}"
        p = get_probs(model, image, device, use_tta)
        probs_sum = p if probs_sum is None else probs_sum + p
        classes   = cls
        loaded.append(name)

    probs = probs_sum / len(loaded)
    top_probs, top_idxs = torch.topk(probs, k=min(top_k, len(classes)))

    best_conf  = top_probs[0].item()
    best_class = classes[top_idxs[0].item()]
    is_unknown = best_conf < threshold

    return {
        "predicted_class": best_class,
        "confidence":      best_conf,
        "unknown":         is_unknown,
        "top_k": [
            {"class": classes[i.item()], "confidence": p.item()}
            for p, i in zip(top_probs, top_idxs)
        ],
        "probs":   probs,
        "classes": classes,
        "models":  loaded,
    }, None


def gradcam_overlay(model_name: str, image: Image.Image) -> np.ndarray | None:
    """Return a (224, 224, 3) float32 overlay array, or None on failure."""
    result = load_model(model_name)
    model, _, device, err = result[0], result[1], result[2], result[3]
    if err or model is None:
        return None
    try:
        target_layer = get_target_layer(model, model_name)
        cam_obj = GradCAM(model, target_layer)

        tfm = _val_transform()
        tensor = tfm(image).unsqueeze(0).to(device)

        cam = cam_obj.generate(tensor)          # (224, 224) in [0, 1]

        orig = np.array(image.resize((224, 224))) / 255.0
        heatmap = plt.cm.jet(cam)[..., :3]      # (224, 224, 3)
        overlay = np.clip(0.55 * orig + 0.45 * heatmap, 0, 1)
        return (overlay * 255).astype(np.uint8)
    except Exception:
        return None


# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    avail = available_models()
    if not avail:
        st.error("No trained checkpoints found in `checkpoints/`.\n\nRun:\n```\npython train.py\n```")
        st.stop()

    # Model selection
    st.markdown("**Select Model(s)**")
    selected = []
    for m in avail:
        result = load_model(m)
        meta   = result[4] if len(result) > 4 else {}
        val_acc = meta.get("val_acc")
        acc_tag = f"  · {val_acc*100:.1f}% val" if val_acc else ""
        if st.checkbox(f"{MODEL_LABELS[m]}{acc_tag}", value=(m == avail[0]), key=f"chk_{m}"):
            selected.append(m)

    if not selected:
        st.warning("Select at least one model.")

    st.divider()

    # Inference options
    st.markdown("**Inference Options**")
    use_tta   = st.toggle("Test-Time Augmentation (TTA)", value=False,
                          help="Averages 4 augmented views — typically +1-2% accuracy, slower.")
    threshold = st.slider("Unknown Threshold", 0.0, 1.0,
                          float(config.CONFIDENCE_THRESHOLD), 0.05,
                          help="If max confidence < threshold the result is shown as UNKNOWN.")
    top_k     = st.slider("Top-K predictions to show", 1, 10, 5)
    show_gcam = st.toggle("Grad-CAM Heatmap", value=True,
                          help="Highlights image regions the model focused on.")

    st.divider()

    # Checkpoint status panel
    st.markdown("**Checkpoint Status**")
    for m in ALL_MODELS:
        exists = os.path.exists(checkpoint_path(m))
        badge  = "pill-ok" if exists else "pill-miss"
        icon   = "✓" if exists else "✗ missing"
        st.markdown(
            f'<span class="{badge}">{icon}</span> {MODEL_LABELS[m]}',
            unsafe_allow_html=True,
        )

    st.divider()
    device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.caption(f"Device: **{device_str}**")
    st.caption("Models are cached — no retraining on image upload.")


# ── main content ───────────────────────────────────────────────────────────────

st.markdown("# 🏃 Human Action Recognition")
st.markdown(
    "Upload any image and the model will identify the **human action** being performed. "
    "If the image doesn't match any known action, it's marked as **Unknown**."
)
st.divider()

uploaded = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="visible",
)

if uploaded is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

if not selected:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

# ── run inference ──────────────────────────────────────────────────────────────
image = Image.open(uploaded).convert("RGB")

col_img, col_res = st.columns([1, 2], gap="large")

with col_img:
    st.markdown('<p class="section-title">Input Image</p>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.caption(f"Size: {image.width} × {image.height} px")

with col_res:
    with st.spinner("Running inference…"):
        result, err = run_inference(image, selected, use_tta, threshold, top_k)

    if err:
        st.error(f"Inference error: {err}")
        st.stop()

    # ── Result banner ──────────────────────────────────────────────────────────
    action_label = result["predicted_class"].replace("_", " ").title()
    confidence   = result["confidence"]
    is_unknown   = result["unknown"]

    if is_unknown:
        st.markdown(f"""
        <div class="result-card result-unknown">
            <h1>⚠️ Unknown Action</h1>
            <p>Confidence {confidence*100:.1f}% is below the threshold ({threshold*100:.0f}%).</p>
            <p>This image does not clearly match any known action class.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-known">
            <h1>{action_label}</h1>
            <p>Confidence: <strong>{confidence*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # ── Metrics row ────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{confidence*100:.1f}%")
    m2.metric("Model(s)", ", ".join(MODEL_LABELS[m] for m in result["models"])
              if len(result["models"]) == 1
              else f"Ensemble ({len(result['models'])})")
    m3.metric("TTA", "ON" if use_tta else "OFF")

    st.divider()

    # ── Top-K predictions ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Top Predictions</p>', unsafe_allow_html=True)
    for rank, pred in enumerate(result["top_k"], 1):
        cls  = pred["class"].replace("_", " ").title()
        conf = pred["confidence"]
        col_lbl, col_bar, col_pct = st.columns([3, 6, 1])
        with col_lbl:
            st.markdown(f"**{rank}.** {cls}")
        with col_bar:
            color = "#2563eb" if rank == 1 and not is_unknown else "#64748b"
            st.markdown(
                f'<div style="background:#1e293b;border-radius:6px;height:18px;margin-top:8px;">'
                f'<div style="background:{color};width:{conf*100:.1f}%;height:100%;'
                f'border-radius:6px;"></div></div>',
                unsafe_allow_html=True,
            )
        with col_pct:
            st.markdown(f"<small style='line-height:2.8'>{conf*100:.1f}%</small>",
                        unsafe_allow_html=True)

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
if show_gcam and not is_unknown:
    st.divider()
    st.markdown('<p class="section-title">Grad-CAM — Where the model looks</p>',
                unsafe_allow_html=True)

    # Run GradCAM only on first selected model (ensemble GradCAM is ambiguous)
    gcam_model = selected[0]
    with st.spinner("Generating Grad-CAM…"):
        overlay = gradcam_overlay(gcam_model, image)

    if overlay is not None:
        orig_224 = np.array(image.resize((224, 224)))

        # Build a side-by-side matplotlib figure → Streamlit image
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="#0e1117")
        for ax in axes:
            ax.axis("off")
        axes[0].imshow(orig_224)
        axes[0].set_title("Original", color="white", fontsize=11, pad=6)
        axes[1].imshow(overlay)
        axes[1].set_title(
            f"Grad-CAM ({MODEL_LABELS[gcam_model]})",
            color="white", fontsize=11, pad=6,
        )
        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close(fig)
    else:
        st.info(f"Grad-CAM is not available for **{MODEL_LABELS[gcam_model]}**.")

# ── footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Stanford 40 Actions dataset · Models: VGG-16, ResNet-50, DenseNet-121, GoogLeNet "
    "· Pretrained on ImageNet and fine-tuned"
)
