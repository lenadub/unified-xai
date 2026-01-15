from audio.audio_utils import wav_to_spectrogram_image

CLASS_NAMES = ["real", "fake"]

def predict_audio(model, audio_bytes):
    x, spec_img = wav_to_spectrogram_image(audio_bytes, return_image=True)

    preds = model.predict(x)

    if isinstance(preds, dict):
        preds = list(preds.values())[0]

    class_idx = int(preds.argmax(axis=1)[0])
    confidence = float(preds.max(axis=1)[0])    
        
    return {
        "x": x,
        "spec_img": spec_img,
        "class_idx": class_idx,
        "confidence": confidence,
        "label": CLASS_NAMES[class_idx]
    }