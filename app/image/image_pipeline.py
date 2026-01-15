import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

def predict_image(model, image_file, device="cpu"):
    img = Image.open(image_file).convert("RGB")

    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        class_idx = int(torch.argmax(probs, dim=1))
        confidence = float(probs[0, class_idx])

    label = "pathology" if class_idx == 1 else "normal"

    return {
        "x": x,
        "class_idx": class_idx,
        "confidence": confidence,
        "label": label,
        "input_img": img
    }
