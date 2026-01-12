from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms, models
from PIL import Image

# -------------------------
# Paths / Config
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = REPO_ROOT / "data" / "images" / "test"
CKPT_PATH = REPO_ROOT / "weights" / "images" / "densenet121" / "best_model.pt"
OUT_DIR = REPO_ROOT / "outputs" / "gradcam_densenet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# -------------------------
# Model
# -------------------------
def build_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    # 🔧 FIX Grad-CAM: disable inplace ReLU
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    return model



# -------------------------
# Grad-CAM helper
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # hooks
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out is tuple; take first
        self.gradients = grad_out[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, x, class_idx: int):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        # activations: [B, C, H, W]
        A = self.activations
        G = self.gradients

        # weights: global-average-pool gradients over spatial dims
        w = G.mean(dim=(2, 3), keepdim=True)          # [B, C, 1, 1]
        cam = (w * A).sum(dim=1, keepdim=True)        # [B, 1, H, W]
        cam = torch.relu(cam)

        # normalize per image
        cam = cam - cam.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        cam = cam / (cam.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-8)
        return cam, logits


# -------------------------
# Image utils
# -------------------------
def denorm(img_tensor):
    """Tensor CHW in normalized space -> numpy HWC [0..255]"""
    x = img_tensor.detach().cpu().numpy()
    x = x * np.array(STD)[:, None, None] + np.array(MEAN)[:, None, None]
    x = np.clip(x, 0, 1)
    x = (x.transpose(1, 2, 0) * 255).astype(np.uint8)
    return x


def overlay_heatmap(img_uint8, cam_2d, alpha=0.45):
    """
    img_uint8: HWC uint8
    cam_2d: HW float [0..1]
    """
    import cv2  # requires opencv-python

    h, w, _ = img_uint8.shape
    cam_resized = cv2.resize(cam_2d, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    out = (img_uint8 * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    return heatmap, out


def pick_one_image_per_class(ds, class_name):
    cls_idx = ds.class_to_idx[class_name]
    for i, (_, y) in enumerate(ds.samples):
        if y == cls_idx:
            return i
    raise RuntimeError(f"No image found for class {class_name}")


def main():
    # dataset
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    print("Classes:", ds.classes)

    # load model
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE)

    import torch.nn.functional as F

    # ===== PATCH DENSENET (FIX GRADCAM) =====
    orig_forward = model.forward

    def forward_no_inplace(x):
        old_relu = F.relu

        def relu_no_inplace(input, inplace=False):
            return old_relu(input, inplace=False)

        F.relu = relu_no_inplace
        try:
            return orig_forward(x)
        finally:
            F.relu = old_relu

    model.forward = forward_no_inplace
 

    model.eval()




    # target layer for DenseNet Grad-CAM
    target_layer = model.features[-1]

    cam_engine = GradCAM(model, target_layer)

    for class_name in ["normal", "pathology"]:
        idx = pick_one_image_per_class(ds, class_name)
        x, y = ds[idx]
        x = x.unsqueeze(0).to(DEVICE)

        # choose class to explain = predicted class (more natural)
        with torch.no_grad():
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())

        cam, logits2 = cam_engine(x, class_idx=pred)
        cam2d = cam[0, 0].detach().cpu().numpy()

        img_uint8 = denorm(x[0])

        # save outputs
        # original
        Image.fromarray(img_uint8).save(OUT_DIR / f"{class_name}_original.png")

        # overlay (needs opencv)
        heatmap, overlay = overlay_heatmap(img_uint8, cam2d, alpha=0.45)
        Image.fromarray(heatmap).save(OUT_DIR / f"{class_name}_heatmap.png")
        Image.fromarray(overlay).save(OUT_DIR / f"{class_name}_overlay.png")

        probs = torch.softmax(logits2, dim=1)[0].detach().cpu().numpy()
        print(f"[{class_name}] idx={idx} true={y} pred={pred} probs={probs}")

    cam_engine.remove()
    print("✅ Saved Grad-CAM outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
