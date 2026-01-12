import os
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# -----------------------------
# Paths (adapt to your repo)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_DIR = REPO_ROOT / "data" / "chexpert_raw" / "csv"
RAW_IMG_DIR = REPO_ROOT / "data" / "chexpert_raw" / "images"
OUT_DIR = REPO_ROOT / "data" / "images"

TRAIN_CSV = RAW_CSV_DIR / "train.csv"
VALID_CSV = RAW_CSV_DIR / "valid.csv"

# Output structure
SPLITS = ["train", "val", "test"]
CLASSES = {0: "normal", 1: "pathology"}

IMG_SIZE = (224, 224)
SEED = 42
TEST_SIZE = 0.15  # test split created from train.csv rows (patient-wise)

# Pulmonary-related labels (CheXpert columns)
PULM_COLS = [
    "Lung Opacity",
    "Lung Lesion",
    "Pneumonia",
    "Edema",
    "Atelectasis",
    "Consolidation",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumothorax",
]

def ensure_dirs():
    for split in SPLITS:
        for _, cls_name in CLASSES.items():
            (OUT_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

def extract_patient_id(path_str: str) -> str:
    # typical path contains: .../patient12345/...
    m = re.search(r"(patient\d+)", path_str)
    return m.group(1) if m else "unknown_patient"

def load_and_label(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic sanity checks
    required = {"Path", "No Finding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    # Keep only needed columns (Path + No Finding + pulmonary cols if present)
    cols = ["Path", "No Finding"] + [c for c in PULM_COLS if c in df.columns]
    df = df[cols].copy()

    # Create patient_id for patient-wise split
    df["patient_id"] = df["Path"].astype(str).apply(extract_patient_id)

    # Rule:
    # normal if No Finding == 1
    df["is_normal"] = df["No Finding"] == 1

    # pathology if any pulmonary col == 1
    pulm_present_cols = [c for c in PULM_COLS if c in df.columns]
    if not pulm_present_cols:
        raise ValueError("None of the pulmonary columns were found in the CSV. Check column names.")

    df["has_pulm_pos"] = (df[pulm_present_cols] == 1).any(axis=1)

    # Ambiguous rows:
    # - not normal AND no pulmonary positive => exclude
    df["keep"] = df["is_normal"] | df["has_pulm_pos"]

    df = df[df["keep"]].copy()

    # Final binary label
    # normal overrides everything (rare conflicts)
    df["label"] = df["has_pulm_pos"].astype(int)
    df.loc[df["is_normal"], "label"] = 0

    # Drop helpers
    df.drop(columns=["is_normal", "has_pulm_pos", "keep"], inplace=True)

    return df

def resize_and_save(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        im = im.resize(IMG_SIZE)
        im.save(dst_path, format="JPEG", quality=95)

def build_full_image_path(rel_path: str) -> Path:
    """
    CSV 'Path' is typically like: CheXpert-v1.0-small/train/patient.../view1_frontal.jpg
    But in our repo we placed train/ and valid/ under data/chexpert_raw/images/.
    So we remove leading 'CheXpert-v1.0-small/' if present and anchor at RAW_IMG_DIR.
    """
    rel = str(rel_path).replace("\\", "/")
    rel = re.sub(r"^CheXpert-v1\.0-small\/", "", rel)
    return RAW_IMG_DIR / rel

def patient_wise_split_train_test(df_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    groups = df_train["patient_id"]
    idx_train, idx_test = next(splitter.split(df_train, groups=groups))
    return df_train.iloc[idx_train].copy(), df_train.iloc[idx_test].copy()

def export_split(df: pd.DataFrame, split_name: str):
    n_ok = 0
    n_missing = 0

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Export {split_name}"):
        src = build_full_image_path(r["Path"])
        label = int(r["label"])
        cls_name = CLASSES[label]

        # destination file name: keep patient/study name but flatten to avoid deep folders
        rel = str(r["Path"]).replace("\\", "/")
        rel = rel.replace("CheXpert-v1.0-small/", "")
        safe_name = rel.replace("/", "__")
        dst = OUT_DIR / split_name / cls_name / safe_name


        if not src.exists():
            n_missing += 1
            continue

        resize_and_save(src, dst)
        n_ok += 1
        rows.append({"split": split_name, "src": str(src), "dst": str(dst), "label": label})

    manifest = pd.DataFrame(rows)
    manifest_path = OUT_DIR / f"manifest_{split_name}.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"[{split_name}] exported={n_ok} missing={n_missing} manifest={manifest_path}")

def main():
    ensure_dirs()

    print("Load + label train.csv ...")
    df_train = load_and_label(TRAIN_CSV)

    print("Load + label valid.csv ...")
    df_val = load_and_label(VALID_CSV)

    # Create test split from train (patient-wise)
    df_train_final, df_test = patient_wise_split_train_test(df_train)

    # Export images
    export_split(df_train_final, "train")
    export_split(df_val, "val")
    export_split(df_test, "test")

    print("Done. Dataset ready in:", OUT_DIR)

if __name__ == "__main__":
    main()
