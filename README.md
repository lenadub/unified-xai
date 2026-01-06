# 🎧 Deepfake Audio Detection with Explainable AI (XAI)

This project implements an **audio-only deepfake detection system** using deep learning and **multiple Explainable AI (XAI) techniques**.
Users can upload a `.wav` file, run inference using different models, and visually compare **Grad-CAM, LIME, and SHAP explanations** side-by-side in a Streamlit web application.

---

## 🚀 Running the Streamlit Application

### 1️⃣ Create and activate a virtual environment

From the project root:

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

---

### 2️⃣ Install dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

> ⚠️ This project was tested with **TensorFlow 2.12.0**.
> Using other versions may lead to incompatibilities.

---

### 3️⃣ Run the Streamlit app

```bash
streamlit run app/app.py
```

The application will open automatically in your browser.

---

## 🎵 Audio Dataset for Inference (Using the App)

To **use the app for inference only**, you may upload **any `.wav` file** directly through the Streamlit interface.

### 📌 Recommended dataset (optional, for testing)

You can use the **Fake-or-Real Audio Dataset** from Kaggle, specifically the **`for-2sec` version**:

🔗 **Dataset link:**
[https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

For inference and demo purposes, we recommend using:

```bash
for-2sec/for-2seconds/testing/
├── real/
└── fake/
```

This subset contains short, standardized 2-second audio samples and is suitable for:

* Manual testing
* Demo purposes
* Model evaluation

You **do NOT** need to place inference audio files inside the project folders —
they can be uploaded directly through the Streamlit interface.

---

## 🔁 Retraining the Models (Optional)

If you want to **retrain the audio classification models**, follow the steps below.

---

### 1️⃣ Download the training dataset

Download the **Fake-or-Real Audio Dataset** from Kaggle and use the **`for-2sec` version**:

🔗 [https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

You will work with:

```bash
for-2sec/for-2seconds/
├── training/
├── validation/
└── testing/
```

---

### 2️⃣ Place raw audio files (IMPORTANT)

Before converting to spectrograms, place the raw `.wav` files into the following structure:

```bash
data/
└── audio/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```

These folders should be populated using the corresponding splits from:

* `for-2seconds/training`
* `for-2seconds/validation`
* `for-2seconds/testing`

Folders must be renamed to `train`, `val`, and `test`

---

### 3️⃣ Convert WAV files to spectrograms

Use the provided script:

```bash
scripts/wav_to_spectrogram.py
```

This script:

* Reads `.wav` files from `data/audio/`
* Converts them into **Mel-spectrogram images**
* Automatically creates and populates:

```bash
data/
└── spectrograms/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
```

**Important notes:**

* Folder names must be exactly `train`, `val`, and `test`
* Images are saved as `.png`
* All spectrograms are resized to **224×224**
* This format is required for all models

---

### 4️⃣ Train the audio models

From the project root, run the following scripts (order does not matter):

#### ▶ MobileNet

```bash
python training/train_audio_mobilenet.py
```

#### ▶ VGG16

```bash
python training/train_audio_vgg.py
```

#### ▶ ResNet50

```bash
python training/train_audio_resnet.py
```

#### ▶ Custom CNN

```bash
python training/train_audio_custom_cnn.py
```

Each script will:

* Load spectrogram images from `data/spectrograms/`
* Train the model
* Save it automatically to:

```bash
weights/audio/
```

Example:

```bash
weights/audio/mobilenet/
weights/audio/vgg16/
weights/audio/resnet50/
weights/audio/custom_cnn/
```

---

### 5️⃣ Use retrained models in the app

No code changes are required.

The Streamlit app automatically loads models from:

```bash
weights/audio/
```

As long as the folder names remain unchanged, newly trained models will be used automatically.

---

## 🧠 Explainability Methods Included

The application provides **side-by-side explainability comparison** using:

* **Grad-CAM** — highlights discriminative spectrogram regions
* **LIME** — local, superpixel-based explanations
* **SHAP** — contribution-based explanations with neutral (gray) regions

All XAI methods are applied automatically to the selected model.

---

## 📁 Project Structure (Current Scope)

```bash
unified-xai/
├── app/
│   ├── app.py
│   ├── audio_utils.py
│   ├── gradcam_utils.py
│   ├── lime_utils.py
│   └── shap_utils.py
├── training/
│   ├── train_audio_mobilenet.py
│   ├── train_audio_vgg.py
│   ├── train_audio_resnet.py
│   └── train_audio_custom_cnn.py
├── scripts/
│   └── wav_to_spectrogram.py
├── data/
│   ├── audio/
│   └── spectrograms/
├── weights/
│   └── audio/
├── requirements.txt
└── README.md
```

---

## ✅ Current Project Scope

✔ Audio-based deepfake detection
✔ Multiple CNN architectures
✔ Explainable AI (Grad-CAM, LIME, SHAP)
✔ Unified Streamlit interface
✔ Side-by-side XAI comparison