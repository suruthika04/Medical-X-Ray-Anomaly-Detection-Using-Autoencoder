from pathlib import Path
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow.keras.models import load_model

from src.autoencoder import IMG_SIZE

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "autoencoder.keras"
THRESHOLD_PATH = ROOT / "artifacts" / "threshold.npy"

if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
    raise FileNotFoundError("Run `python train.py` before starting the app.")

model = load_model(MODEL_PATH)
threshold = float(np.load(THRESHOLD_PATH))


def predict(image: Image.Image):
    if image is None:
        return None, None, "Please upload an X-ray image.", 0.0

    image = image.convert("L").resize((IMG_SIZE, IMG_SIZE))
    img = np.asarray(image).astype("float32") / 255.0
    batch = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    reconstruction = model.predict(batch, verbose=0)
    error = float(np.mean(np.square(batch - reconstruction)))
    result = "ANOMALY" if error > threshold else "NORMAL"

    return img, reconstruction[0], result, error


with gr.Blocks(title="Medical X-Ray Anomaly Detection") as ui:
    gr.Markdown("# Medical X-Ray Anomaly Detection")
    gr.Markdown(
        "Upload a chest X-ray to reconstruct it with the trained autoencoder "
        "and calculate its reconstruction error."
    )

    image = gr.Image(type="pil", label="Upload X-Ray")
    button = gr.Button("Analyze X-Ray")

    with gr.Row():
        original = gr.Image(label="Original X-Ray")
        reconstructed = gr.Image(label="Reconstructed X-Ray")

    result = gr.Textbox(label="Prediction")
    error = gr.Number(label="Reconstruction Error")
    threshold_box = gr.Number(
        label="Anomaly Threshold",
        value=threshold,
        interactive=False,
    )

    button.click(
        predict,
        inputs=image,
        outputs=[original, reconstructed, result, error],
    )

    gr.Markdown(
        "**Research prototype:** This system is not a medical diagnostic tool."
    )


if __name__ == "__main__":
    ui.launch()
