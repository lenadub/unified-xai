from pathlib import Path
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

from sklearn.metrics import f1_score, roc_auc_score


# -------------------------
# Config
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "images"
WEIGHTS_DIR = REPO_ROOT / "weights" / "images" / "densenet121"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 3              # on commence petit, puis on augmentera
LR = 1e-4
NUM_WORKERS = 0         # Windows/OneDrive friendly
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_model(num_classes=2):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def make_weighted_sampler(train_ds: datasets.ImageFolder):
    # ImageFolder: targets = list of class indices
    targets = np.array(train_ds.targets)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)  # inverse frequency
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler, class_counts


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_y = []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]  # proba class 1 = pathology

        total_loss += loss.item() * x.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_y = np.concatenate(all_y)

    # metrics
    preds = (all_probs >= 0.5).astype(int)
    f1 = f1_score(all_y, preds)
    # ROC-AUC needs both classes present
    auc = roc_auc_score(all_y, all_probs) if len(np.unique(all_y)) == 2 else float("nan")

    return total_loss / max(len(all_y), 1), f1, auc


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / max(len(loader.dataset), 1)


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    train_tf, eval_tf = get_transforms()

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tf)

    print("Classes:", train_ds.classes)
    print("Train size:", len(train_ds), "| Val size:", len(val_ds))

    # Weighted sampler to handle imbalance
    sampler, class_counts = make_weighted_sampler(train_ds)
    print("Train class counts [normal, pathology]:", class_counts.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )

    model = build_model(num_classes=2).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = -1.0
    best_path = WEIGHTS_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1, val_auc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f}")

        # Save best by AUC (fallback to F1 if AUC is nan)
        score = val_auc if not np.isnan(val_auc) else val_f1
        if score > best_auc:
            best_auc = score
            torch.save({
                "model": model.state_dict(),
                "classes": train_ds.classes,
                "epoch": epoch,
                "val_auc": float(val_auc),
                "val_f1": float(val_f1),
            }, best_path)
            print("✅ Saved best:", best_path)

    print("Done. Best score:", best_auc)


if __name__ == "__main__":
    main()
