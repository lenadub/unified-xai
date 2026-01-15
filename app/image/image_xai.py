from .gradcam import run_gradcam  # ou ton implément GradCAM
from .gradcam import run_gradcam

from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
import shap
import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.segmentation import slic


def run_image_xai(model, x, class_idx, methods, gradcam_layer):
    results = {}

    if "Grad-CAM" in methods:
        target_layer = get_module_by_name(model, gradcam_layer)

        heatmap = run_gradcam(
            model=model,
            x=x,
            class_idx=class_idx,
            target_layer=target_layer   # 🔥 MODULE, PAS STRING
        )

        results["Grad-CAM"] = heatmap

    if "LIME" in methods:
        results["LIME"] = run_lime_image(model, x)

    if "SHAP" in methods:
        results ["SHAP"] = compute_shap_image(model, x, class_idx=class_idx)

    return results

def get_module_by_name(model, layer_name: str):
    """
    Resolve a string like 'features.11' or 'features.denseblock4'
    into the corresponding PyTorch module.
    """
    module = model
    for attr in layer_name.split("."):
        if attr.isdigit():
            module = module[int(attr)]
        else:
            module = getattr(module, attr)
    return module

def run_lime_image(model, x):
    explainer = lime_image.LimeImageExplainer()

    # 🔥 FIX : remove batch dimension if present
    if x.dim() == 4:
        x_img = x[0]
    else:
        x_img = x

    def predict_fn(images):
        images = torch.tensor(images).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explanation = explainer.explain_instance(
        x_img.permute(1, 2, 0).cpu().numpy(),  # HWC
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=500
    )

    img, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    return mark_boundaries(img, mask)


def compute_shap_image(model, x, class_idx):
    """
    Compute SHAP explanation and return a 224x224 RGB visualization image
    using PiYG colormap (aligned with audio SHAP implementation).
    """

    model.eval()

    # -----------------------------
    # Ensure correct shapes
    # -----------------------------
    if x.dim() == 3:
        x = x.unsqueeze(0)  # (1, C, H, W)

    # Convert to numpy image for superpixels
    image = x[0].permute(1, 2, 0).detach().cpu().numpy()
    image = (image - image.min()) / (image.max() + 1e-8)

    # -----------------------------
    # SHAP baseline (neutral image)
    # -----------------------------
    background = torch.zeros_like(x)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x)

    # Handle binary / multi-class
    if isinstance(shap_values, list):
        sv = shap_values[class_idx][0]
    else:
        sv = shap_values[0]

    # sv shape: (C, H, W)
    sv = np.mean(sv, axis=0)  # → (H, W)

    # -----------------------------
    # Superpixels (for spatial coherence)
    # -----------------------------
    segments = slic(
        image,
        n_segments=60,
        compactness=10,
        start_label=0
    )

    shap_map = np.zeros_like(segments, dtype=np.float32)

    for seg_id in np.unique(segments):
        mask = segments == seg_id
        shap_map[mask] = np.mean(sv[mask])

    # -----------------------------
    # Normalize to [-1, 1]
    # -----------------------------
    max_val = np.max(np.abs(shap_map)) + 1e-8
    shap_map = shap_map / max_val

    # -----------------------------
    # Apply PiYG colormap manually
    # -----------------------------
    cmap = cm.get_cmap("PiYG")
    shap_rgb = cmap((shap_map + 1) / 2)[..., :3]  # RGB in [0,1]

    # -----------------------------
    # Render EXACT 224×224 image
    # -----------------------------
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(shap_rgb)
    ax.axis("off")

    fig.canvas.draw()

    shap_img = np.frombuffer(
        fig.canvas.buffer_rgba(), dtype=np.uint8
    ).reshape(224, 224, 4)[..., :3]

    plt.close(fig)

    return shap_img
