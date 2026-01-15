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

## Chest X-Ray Image Detection with Explainable AI (XAI)

In addition to audio deepfake detection, this unified project also includes an image-based chest X-ray classification pipeline (Normal vs. Malignant), paired with Explainable AI (XAI) methods to interpret model decisions.

Users can upload a chest X-ray image (.png, .jpg, .jpeg), run inference with a supported model, and visualize explanations such as Grad-CAM, LIME, and SHAP (when compatible).

## Image Dataset (CheXpert)

The image pipeline is based on the CheXpert chest radiograph dataset from Stanford ML Group.

Dataset reference:
https://stanfordmlgroup.github.io/competitions/chexpert/

For this project, the task is simplified to a binary classification setup:

- Normal
- Malignant / Lung Cancer
- Note: You may use any chest X-ray dataset for demo/inference purposes, as long as inputs follow the expected format.

## Image Data Organization

If you want to run training or evaluation, place images using the following structure:

```bash
data/
└── images/
    ├── train/
    │   ├── normal/
    │   └── cancer/
    ├── val/
    │   ├── normal/
    │   └── cancer/
    └── test/
        ├── normal/
        └── cancer/
```

Requirements:

- Supported formats: .png, .jpg, .jpeg
- Images are resized to 224×224 during preprocessing
- Normalization is handled automatically by the pipeline

## Training the Image Models (Optional)

If you want to retrain the image classification models, run the training scripts (PyTorch):

AlexNet
```bash
python training/train_image_alexnet.py
```
DenseNet121
```bash
python training/train_image_densenet.py
```

Each script will:

- load images from data/images/
- train the model
- save weights to:
```bash
weights/image/
├── alexnet/
└── densenet121/
```

Models are stored as PyTorch checkpoints (.pth).

## Explainability Methods for Images

The image pipeline supports the following explainability techniques:

- Grad-CAM — highlights the most discriminative regions in the X-ray
- LIME — local explanations using superpixel perturbations
- SHAP — contribution-based explanations (availability may depend on model/input constraints)

All methods are automatically filtered based on the input type and selected model.

## Unified Multi-Modal XAI Interface (Audio + Image)

This project refactors and merges two repositories into a single Streamlit interface that supports:

- Audio deepfake detection (.wav)
- Chest X-ray classification (.png, .jpg, .jpeg)
- Multiple deep learning models per modality
- Multiple XAI methods with automatic compatibility filtering
- A dedicated comparison tab for side-by-side analysis

## Streamlit Tabs Overview
### Tab 1 — Prediction

The Prediction tab is designed for fast interaction and demo:
- Select input type (Audio or Image)
- Upload a file
- Select a compatible model
- Select one or multiple XAI methods
- Run inference and display:
 - predicted label + confidence
 - selected XAI visualizations

This tab ensures basic functionality (as required in the instructions):
1 dataset + 1 model + 1 XAI method → explanation output.

### Tab 2 — XAI Comparison (All Compatible Methods)

The XAI Comparison tab implements the project requirement for systematic comparison:

Even if the user selects only one method in Tab 1, Tab 2 computes and displays all XAI methods compatible with the last prediction (input type + selected model).

Workflow:
- Tab 1 stores the latest prediction context (model, preprocessed input x, predicted class index)
- Tab 2 retrieves this context and recomputes all compatible XAI methods
- Results are displayed side-by-side (or in a stacked grid) with clear method labels

This ensures a fair and consistent comparison of explainability techniques for the exact same input and model.

## Automatic XAI Compatibility Filtering

XAI options are filtered automatically so that only applicable methods appear for a given modality/model.

Example:
- If the user uploads an audio file, image-only XAI methods are hidden/disabled
- If the user uploads an image, audio-specific logic is not exposed

This avoids invalid selections and improves usability.

## 📁 Project Structure (Current Scope)

```bash
unified-xai/
├── app/
│   ├── app.py
│   ├── audio/
│       ├── audio_pipeline.py
│       ├── audio_utils.py
│       └── audio_xai.py
│   ├── image/
│       ├── image_pipeline.py
│       ├── gradcam.py.py
│       └── image_xai.py
│   ├── gradcam_utils.py
│   ├── lime_utils.py
│   └── shap_utils.py
├── training/
│   ├── train_audio_mobilenet.py
│   ├── train_audio_vgg.py
│   ├── train_audio_resnet.py
│   ├── train_audio_custom_cnn.py
│   ├── train_image_alexnet.py
│   ├── train_image_densenet.py
│   ├── evaluate_image_model_alexnet.py
│   └── evaluate_image_modeldensenet.py
├── scripts/
│   ├── wav_to_spectrogram.py
│   ├── demo_gradcam_densenet.py
│   └── prepare_chexpert.py
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