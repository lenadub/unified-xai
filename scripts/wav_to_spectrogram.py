import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io

# ===============================
# Configuration
# ===============================
INPUT_ROOT = "data/audio"
OUTPUT_ROOT = "data/spectrograms"
IMG_SIZE = 224
N_MELS = 128

SPLITS = ["train", "val", "test"]
CLASSES = ["real", "fake"]

# ===============================
# Utility function
# ===============================
def wav_to_melspec(wav_path, out_path):
    y, sr = librosa.load(wav_path, sr=None)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sr, ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Read buffer into OpenCV
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite(out_path, img)
# ===============================
# Main loop
# ===============================
for split in SPLITS:
    for label in CLASSES:
        in_dir = os.path.join(INPUT_ROOT, split, label)
        out_dir = os.path.join(OUTPUT_ROOT, split, label)

        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_dir):
            print(f"Skipping missing folder: {in_dir}")
            continue

        for file in os.listdir(in_dir):
            if file.lower().endswith(".wav"):
                wav_path = os.path.join(in_dir, file)
                png_path = os.path.join(out_dir, file.replace(".wav", ".png"))

                wav_to_melspec(wav_path, png_path)
                print(f"[{split}/{label}] Saved {png_path}")

print("All spectrograms generated successfully.")