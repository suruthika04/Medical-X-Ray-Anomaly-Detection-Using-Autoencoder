from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.autoencoder import IMG_SIZE

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "chest_xray"
ARTIFACTS = ROOT / "artifacts"


def load_images(path: Path):
    extensions = {".jpg", ".jpeg", ".png"}
    images = []

    for file in sorted(path.iterdir()):
        if file.suffix.lower() not in extensions:
            continue
        image = load_img(
            file,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode="grayscale",
        )
        images.append(img_to_array(image).astype("float32") / 255.0)

    return np.stack(images) if images else np.empty(
        (0, IMG_SIZE, IMG_SIZE, 1), dtype="float32"
    )


def main():
    model_path = ARTIFACTS / "autoencoder.keras"
    threshold_path = ARTIFACTS / "threshold.npy"

    if not model_path.exists() or not threshold_path.exists():
        raise FileNotFoundError("Run `python train.py` first.")

    model = load_model(model_path)
    threshold = float(np.load(threshold_path))

    normal = load_images(DATASET / "test" / "NORMAL")
    anomaly = load_images(DATASET / "test" / "PNEUMONIA")

    if len(normal) == 0 or len(anomaly) == 0:
        raise ValueError(
            "Both test/NORMAL and test/PNEUMONIA must contain images."
        )

    x = np.concatenate([normal, anomaly], axis=0)
    y_true = np.concatenate([
        np.zeros(len(normal), dtype=int),
        np.ones(len(anomaly), dtype=int),
    ])

    reconstructed = model.predict(x, verbose=0)
    errors = np.mean(np.square(x - reconstructed), axis=(1, 2, 3))
    y_pred = (errors > threshold).astype(int)

    plt.figure(figsize=(8, 5))
    plt.hist(errors[y_true == 0], bins=40, alpha=0.7, label="Normal")
    plt.hist(errors[y_true == 1], bins=40, alpha=0.7, label="Pneumonia / Anomaly")
    plt.axvline(threshold, linestyle="--", label=f"Threshold = {threshold:.6f}")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Number of Images")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "reconstruction_error_distribution.png", dpi=200)
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Anomaly"],
    )
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "confusion_matrix.png", dpi=200)
    plt.close()

    print("Evaluation complete.")
    print(f"Threshold: {threshold:.8f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
