import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io

IMG_SIZE = 224
N_MELS = 128

def wav_to_spectrogram_image(wav_bytes, return_image=False):
    # Load audio from bytes
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None)

    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Render spectrogram with matplotlib
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sr, ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Decode PNG buffer → image
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Keep a copy for display (uint8, 0–255)
    display_img = img.copy()

    # Normalize for model
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

    if return_image:
        return img, display_img

    return img