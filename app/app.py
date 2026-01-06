import streamlit as st
import tensorflow as tf
from audio_utils import wav_to_spectrogram_image

# ===============================
# Config
# ===============================
st.set_page_config(page_title="Deepfake Audio Detection", layout="centered")

MODEL_PATHS = {
    "MobileNet": "weights/audio/mobilenet",
    "VGG16": "weights/audio/vgg16",
    "ResNet50": "weights/audio/resnet50",
    "Custom CNN": "weights/audio/custom_cnn"
}

CLASS_NAMES = ["real", "fake"]

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# ===============================
# UI
# ===============================
st.title("Deepfake Audio Detection (Audio Only)")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

model_name = st.selectbox("Select model", list(MODEL_PATHS.keys()))

if uploaded_file is not None:
    st.audio(uploaded_file.read(), format="audio/wav")

    if st.button("Run Prediction"):
        with st.spinner("Processing..."):
            wav_bytes = uploaded_file.getvalue()
            x = wav_to_spectrogram_image(wav_bytes)

            model = load_model(MODEL_PATHS[model_name])
            preds = model.predict(x)

            class_idx = int(preds.argmax(axis=1)[0])
            confidence = float(preds.max(axis=1)[0])

        st.success(
            f"Prediction: **{CLASS_NAMES[class_idx]}** "
            f"(confidence: {confidence:.3f})"
        )