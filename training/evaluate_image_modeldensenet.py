from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm



REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "images" / "test"
CKPT_PATH = REPO_ROOT / "weights" / "images" / "densenet121" / "best_model.pt"

BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model


@torch.no_grad()
def main():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    ds = datasets.ImageFolder(DATA_DIR, transform=tf)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE)
    model.eval()

    probs, ytrue = [], []
    for x, y in tqdm(dl, desc="Evaluating test"):
        x = x.to(DEVICE)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
        ytrue.append(y.numpy())

    probs = np.concatenate(probs)
    ytrue = np.concatenate(ytrue)
    pred = (probs >= 0.5).astype(int)

    f1 = f1_score(ytrue, pred)
    auc = roc_auc_score(ytrue, probs) if len(np.unique(ytrue)) == 2 else float("nan")
    cm = confusion_matrix(ytrue, pred)

    print("TEST_F1:", f1)
    print("TEST_AUC:", auc)
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(ytrue, pred, target_names=ds.classes))


if __name__ == "__main__":
    main()
