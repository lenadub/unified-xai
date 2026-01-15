from gradcam_utils import compute_gradcam, overlay_gradcam
from lime_utils import compute_lime
from shap_utils import compute_shap

import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def run_audio_xai(model, x, class_idx, gradcam_layer, methods):
    results = {}

    if "Grad-CAM" in methods:
        heatmap = compute_gradcam(model, x, class_idx, gradcam_layer)
        results["Grad-CAM"] = overlay_gradcam(x, heatmap)

    if "LIME" in methods:
        results["LIME"] = compute_lime(model, x, class_idx)

    if "SHAP" in methods:
        results["SHAP"] = compute_shap_audio(model, x, class_idx)

    return results

import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def compute_shap_audio(model, x, class_idx):
    """
    Robust SHAP for audio spectrograms.
    Handles all SHAP output shapes and returns (H, W, 3).
    """

    # x expected: (1, H, W, C)
    assert x.ndim == 4, f"Expected (1,H,W,C), got {x.shape}"

    # -----------------------------
    # Baseline
    # -----------------------------
    background = np.zeros_like(x)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x)

    # -----------------------------
    # STEP 1 — select class
    # -----------------------------
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    else:
        sv = shap_values

    # sv now could be:
    # (1, H, W, C)
    # (1, H, W, C, K)
    # (H, W, C, K)
    # etc.

    sv = np.array(sv)

    # -----------------------------
    # STEP 2 — remove batch dim
    # -----------------------------
    if sv.ndim >= 4 and sv.shape[0] == 1:
        sv = sv[0]

    # -----------------------------
    # STEP 3 — collapse class dim if still present
    # -----------------------------
    # If last dim = classes
    if sv.ndim == 4:
        # (H, W, C, K) → keep explained class OR mean
        sv = sv[..., 0]  # safe because class already selected

    # -----------------------------
    # STEP 4 — collapse channels
    # -----------------------------
    if sv.ndim == 3:
        # (H, W, C) → (H, W)
        sv = np.mean(sv, axis=-1)

    # -----------------------------
    # FINAL CHECK
    # -----------------------------
    if sv.ndim != 2:
        raise ValueError(f"SHAP map must be 2D after reduction, got {sv.shape}")

    # -----------------------------
    # Normalize to [-1, 1]
    # -----------------------------
    max_val = np.max(np.abs(sv)) + 1e-8
    shap_map = sv / max_val

    # -----------------------------
    # Apply PiYG colormap
    # -----------------------------
    cmap = cm.get_cmap("PiYG")
    shap_rgb = cmap((shap_map + 1) / 2)[..., :3]  # (H, W, 3)

    # -----------------------------
    # Render clean RGB image
    # -----------------------------
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(shap_rgb, aspect="auto")
    ax.axis("off")

    fig.canvas.draw()

    shap_img = np.frombuffer(
        fig.canvas.buffer_rgba(), dtype=np.uint8
    ).reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]

    plt.close(fig)

    return shap_img