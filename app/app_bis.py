import streamlit as st
import tensorflow as tf
import json
import torch
from torchvision import models
from torch import nn
import numpy as np

from audio.audio_pipeline import predict_audio
from audio.audio_xai import run_audio_xai
from audio.audio_utils import wav_to_spectrogram_image

from image.image_pipeline import predict_image
from image.image_xai import run_image_xai

import keras
import os

def normalize_for_display(img):
    """
    Normalize XAI outputs for Streamlit display.
    """

    img = np.asarray(img)

    # 🔥 CASE: (H, W, C, K) → keep only first / predicted class
    if img.ndim == 4 and img.shape[-1] in [2, 3, 4]:
        # assume last dim = classes
        img = img[..., 0]  # keep class 0 or predicted class

    # Case 1: (H, W) → OK (grayscale heatmap)
    if img.ndim == 2:
        img = img.astype(np.float32)
        img -= img.min()
        img /= (img.max() + 1e-8)
        return img

    # Case 2: (H, W, 1) → squeeze to (H, W)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
        img -= img.min()
        img /= (img.max() + 1e-8)
        return img

    # Case 3: (H, W, 3) → RGB image
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.astype(np.float32)
        img -= img.min()
        img /= (img.max() + 1e-8)
        return img

    # Anything else = bug
    raise ValueError(f"Unsupported image shape for display: {img.shape}")


@st.cache_resource
def load_tf_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_pt_model(path, model_name):
    if model_name == "AlexNet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, 2
        )

    elif model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features, 2
        )

    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model


if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "xai_comparison" not in st.session_state:
    st.session_state.xai_comparison = None

# CONFIG --------------------------------
st.set_page_config(
    page_title="Deepfake Audio Detection",
    layout="centered"
)

with open("app/models_config.json", "r") as f:
    MODELS_CONFIG = json.load(f)

# UI ------------------------------------
st.title("Deepfake Detection")

tab_pred, tab_xai = st.tabs(["Prediction", "XAI Comparison"])

# Prediction Tab -----------------------
with tab_pred:
    st.header("Prediction")
    st.subheader("Input selection")

    input_type = st.radio(
        "Select input type",
        ["Audio", "Image"],
        horizontal=True
    )

    uploaded_file = None

    if input_type == "Audio":
        uploaded_file = st.file_uploader(
            "Upload an audio file (.wav)",
            type=["wav"]
        )

    elif input_type == "Image":
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray image (.png, .jpg)",
            type=["png", "jpg", "jpeg"]
        )

    if uploaded_file is not None:
        st.subheader("Input preview")

        if input_type == "Audio":
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format="audio/wav")

            x, spec_img = wav_to_spectrogram_image(
                audio_bytes,
                return_image=True
            )

            st.subheader("Spectrogram")
            st.image(spec_img, use_container_width=True)

        elif input_type == "Image":
            st.image(uploaded_file, use_container_width=True)

        st.divider()
        st.subheader("Model selection")
        model_name = st.selectbox(
        "Select model",
            list(MODELS_CONFIG[input_type]["models"].keys())
        )

        st.divider()
        st.subheader("Explainability (XAI) selection")

        st.caption(
            "You can select one or multiple XAI methods for comparison."
        )

        # Récupération de la config du modèle sélectionné
        model_cfg = MODELS_CONFIG[input_type]["models"][model_name]
        available_xai = model_cfg.get("xai_methods", [])

        selected_xai = st.multiselect(
            "Select XAI method(s)",
            available_xai,
            help="Only XAI methods compatible with the selected input and model are shown."
        )

        if st.button("Run prediction"):
            st.session_state.run_clicked = True

        if st.session_state.run_clicked:

            if not selected_xai:
                st.warning("Please select at least one XAI method.")
                st.stop()

            with st.spinner("Running prediction..."):

                if input_type == "Audio":
                    model = load_tf_model(model_cfg["path"])
                    result = predict_audio(model, audio_bytes)

                    xai_results = run_audio_xai(
                        model=model,
                        x=result["x"],
                        class_idx=result["class_idx"],
                        gradcam_layer=model_cfg["gradcam_layer"],
                        methods=selected_xai
                    )

                    st.session_state.last_prediction = {
                        "input_type": input_type,
                        "model_name": model_name,
                        "model_cfg": model_cfg,
                        "model": model,
                        "x": result["x"],
                        "class_idx": result["class_idx"],
                        "label": result["label"],
                        "confidence": float(result["confidence"]),
                    }

                elif input_type == "Image":
                    model = load_pt_model(
                        model_cfg["path"],
                        model_name=model_name
                    )
                    result = predict_image(model, uploaded_file)

                    xai_results = run_image_xai(
                        model=model,
                        x=result["x"],
                        class_idx=result["class_idx"],
                        gradcam_layer=model_cfg["gradcam_layer"],
                        methods=selected_xai
                    )

                    st.session_state.last_prediction = {
                        "input_type": input_type,
                        "model_name": model_name,
                        "model_cfg": model_cfg,
                        "model": model,
                        "x": result["x"],
                        "class_idx": result["class_idx"],
                        "label": result["label"],
                        "confidence": float(result["confidence"]),
                    }

            st.success(
                f"Prediction: **{result['label']}** "
                f"(confidence: {result['confidence']:.3f})"
            )

            st.session_state.last_run = {
                "input_type": input_type,
                "model_name": model_name,
                "label": result["label"],
                "confidence": float(result["confidence"]),
                "xai_results": xai_results,
                "file_preview": audio_bytes if input_type == "Audio" else uploaded_file,
            }

            st.divider()
            st.subheader("Explainability results")

            for xai_name, xai_img in xai_results.items():
                st.markdown(f"### {xai_name}")
                st.image(normalize_for_display(xai_img), width="stretch")

    with tab_xai:
        st.header("XAI Comparison")
        st.caption("Comparison of all compatible XAI methods for the latest prediction.")

        last = st.session_state.last_prediction

        if last is None:
            st.info("Run a prediction in the Prediction tab first.")
            st.stop()

        st.subheader("Prediction summary")
        st.write(
            f"**Input type:** {last['input_type']}  \n"
            f"**Model:** {last['model_name']}  \n"
            f"**Prediction:** {last['label']} "
            f"(confidence: {last['confidence']:.3f})"
        )

        st.divider()

        # 🔹 Get ALL compatible XAI methods (not only selected ones)
        model_cfg = last["model_cfg"]
        all_xai_methods = model_cfg.get("xai_methods", [])

        if st.button("Compute all XAI methods"):
            with st.spinner("Computing XAI explanations..."):

                if last["input_type"] == "Audio":
                    xai_results = run_audio_xai(
                        model=last["model"],
                        x=last["x"],
                        class_idx=last["class_idx"],
                        gradcam_layer=model_cfg["gradcam_layer"],
                        methods=all_xai_methods
                    )
                else:
                    xai_results = run_image_xai(
                        model=last["model"],
                        x=last["x"],
                        class_idx=last["class_idx"],
                        gradcam_layer=model_cfg["gradcam_layer"],
                        methods=all_xai_methods
                    )

                st.session_state.xai_comparison = xai_results

        xai_results = st.session_state.xai_comparison

        if xai_results:
            st.subheader("Side-by-side comparison")

            cols = st.columns(len(xai_results))

            for col, (method, img) in zip(cols, xai_results.items()):
                with col:
                    st.markdown(f"### {method}")
                    st.image(
                        normalize_for_display(img),
                        use_container_width=True
                    )
