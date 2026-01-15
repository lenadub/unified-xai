import streamlit as st
import tensorflow as tf

from app.audio.audio_utils import wav_to_spectrogram_image
from gradcam_utils import compute_gradcam, overlay_gradcam
from lime_utils import compute_lime
from shap_utils import compute_shap

# ===============================
# Config
# ===============================
st.set_page_config(
    page_title="Deepfake Audio Detection",
    layout="centered"
)

MODEL_PATHS = {
    "MobileNet": "weights/audio/mobilenet",
    "VGG16": "weights/audio/vgg16",
    "ResNet50": "weights/audio/resnet50",
    "Custom CNN": "weights/audio/custom_cnn"
}

GRADCAM_LAYERS = {
    "MobileNet": "conv_pw_13_relu",
    "VGG16": "block5_conv3",
    "ResNet50": "conv5_block3_out",
    "Custom CNN": "conv2d_2"
}

CLASS_NAMES = ["real", "fake"]

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# ===============================
# UI
# ===============================
st.title("Deepfake Audio Detection (Audio Only)")

tab_pred, tab_xai = st.tabs(["Prediction", "XAI Comparison"])

# ===============================
# Prediction Tab
# ===============================
with tab_pred:
    st.header("Prediction")

    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")

        x, spec_img = wav_to_spectrogram_image(
            audio_bytes,
            return_image=True
        )

        # Store in session
        st.session_state["x"] = x
        st.session_state["spec_img"] = spec_img

        # Show spectrogram
        st.subheader("Mel-Spectrogram")
        st.image(spec_img, use_container_width=True)

        # Model selection
        model_name = st.selectbox("Select model", list(MODEL_PATHS.keys()))
        st.session_state["model_name"] = model_name

        if st.button("Run Prediction"):
            with st.spinner("Processing..."):
                model = load_model(MODEL_PATHS[model_name])
                preds = model.predict(x)

                class_idx = int(preds.argmax(axis=1)[0])
                confidence = float(preds.max(axis=1)[0])

                st.session_state["model"] = model
                st.session_state["class_idx"] = class_idx
                st.session_state["confidence"] = confidence

            st.success(
                f"Prediction: **{CLASS_NAMES[class_idx]}** "
                f"(confidence: {confidence:.3f})"
            )
    else:
        st.info("Please upload a WAV file.")

# ===============================
# XAI Comparison Tab
# ===============================
with tab_xai:
    st.header("Explainability Comparison")

    if "model" not in st.session_state:
        st.info("Run a prediction first in the Prediction tab.")
    else:
        model = st.session_state["model"]
        x = st.session_state["x"]
        class_idx = st.session_state["class_idx"]
        model_name = st.session_state["model_name"]

        col1, col2, col3 = st.columns(3)

        # -------- Grad-CAM --------
        with col1:
            st.subheader("Grad-CAM")

            heatmap = compute_gradcam(
                model,
                x,
                class_idx,
                GRADCAM_LAYERS[model_name]
            )
            cam_overlay = overlay_gradcam(x, heatmap)
            st.image(cam_overlay, use_container_width=True)

        # -------- LIME --------
        with col2:
            st.subheader("LIME")

            lime_img = compute_lime(model, x, class_idx)
            st.image(lime_img, use_container_width=True)

        # -------- SHAP --------
        with col3:
            st.subheader("SHAP")

            shap_img = compute_shap(model, x, class_idx)
            st.image(shap_img, use_container_width=True)