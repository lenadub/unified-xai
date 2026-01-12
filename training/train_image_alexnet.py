from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

# -------------------------
# Config
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "images"
WEIGHTS_DIR = REPO_ROOT / "weights" / "images" / "alexnet"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64          # AlexNet plus léger
EPOCHS = 3
LR = 1e-4
NUM_WORKERS = 0          # safe Windows
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_model(num_classes=2):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_y = [], []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item() * x.size(0)
        all_probs.append(probs.cpu().numpy())
        all_y.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_y = np.concatenate(all_y)
    preds = (all_probs >= 0.5).astype(int)

    f1 = f1_score(all_y, preds)
    auc = roc_auc_score(all_y, all_probs) if len(np.unique(all_y)) == 2 else float("nan")

    return total_loss / len(all_y), f1, auc


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

    return total_loss / len(loader.dataset)


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    train_tf, eval_tf = get_transforms()

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tf)

    print("Classes:", train_ds.classes)
    print("Train size:", len(train_ds), "| Val size:", len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = build_model(num_classes=2).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = -1.0
    best_path = WEIGHTS_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1, val_auc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_f1={val_f1:.4f} val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model": model.state_dict(),
                "classes": train_ds.classes,
                "epoch": epoch,
                "val_f1": float(val_f1),
                "val_auc": float(val_auc),
            }, best_path)
            print("✅ Saved best:", best_path)

    # Save final model
    torch.save(model.state_dict(), WEIGHTS_DIR / "model.pt")
    print("Done. Best AUC:", best_auc)


if __name__ == "__main__":
    main()
