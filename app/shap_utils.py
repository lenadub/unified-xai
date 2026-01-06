import shap
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from matplotlib import cm

def compute_shap(model, x, class_idx):
    """
    Compute SHAP explanation and return a 224x224 RGB visualization image
    using PiYG colormap (correctly).
    """

    # -----------------------------
    # Baseline
    # -----------------------------
    background = np.zeros((1,) + x.shape[1:], dtype=np.float32)
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x)

    if isinstance(shap_values, list):
        sv = shap_values[class_idx][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        sv = shap_values[0]

    image = x[0]

    # -----------------------------
    # Superpixels
    # -----------------------------
    segments = slic(image, n_segments=50, compactness=10, start_label=0)

    # -----------------------------
    # Aggregate SHAP
    # -----------------------------
    shap_map = np.zeros(segments.shape, dtype=np.float32)
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        shap_map[mask] = np.mean(sv[mask])

    # -----------------------------
    # Normalize to [-1, 1]
    # -----------------------------
    max_val = np.max(np.abs(shap_map)) + 1e-8
    shap_map = shap_map / max_val

    # -----------------------------
    # APPLY PiYG COLORMAP MANUALLY
    # -----------------------------
    cmap = cm.get_cmap("PiYG")
    shap_rgb = cmap((shap_map + 1) / 2)[..., :3]  # → RGB in [0,1]


    # -----------------------------
    # Render EXACT 224×224
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