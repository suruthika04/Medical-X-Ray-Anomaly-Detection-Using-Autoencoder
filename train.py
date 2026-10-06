from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.autoencoder import build_autoencoder, IMG_SIZE

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "chest_xray"
ARTIFACTS = ROOT / "artifacts"

BATCH_SIZE = 32
EPOCHS = 20
SEED = 42


def load_normal_images(path: Path) -> np.ndarray:
    """Load normal X-rays as 128x128 grayscale arrays normalized to 0-1."""
    extensions = {".jpg", ".jpeg", ".png"}
    images = []

    if not path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {path}")

    for file in sorted(path.iterdir()):
        if file.suffix.lower() not in extensions:
            continue
        image = load_img(
            file,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode="grayscale",
        )
        images.append(img_to_array(image).astype("float32") / 255.0)

    if not images:
        raise ValueError(f"No X-ray images found in: {path}")

    return np.stack(images)


def reconstruction_errors(model, images):
    reconstructed = model.predict(images, verbose=0)
    return np.mean(np.square(images - reconstructed), axis=(1, 2, 3))


def main():
    ARTIFACTS.mkdir(exist_ok=True)

    x = load_normal_images(DATASET / "train" / "NORMAL")

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x))
    split = max(1, int(0.8 * len(x)))

    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(x) else indices[:1]

    x_train = x[train_idx]
    x_val = x[val_idx]

    model = build_autoencoder()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        x_train,
        validation_data=(x_val, x_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
    )

    val_errors = reconstruction_errors(model, x_val)
    threshold = float(np.percentile(val_errors, 95))

    model.save(ARTIFACTS / "autoencoder.keras")
    np.save(ARTIFACTS / "threshold.npy", threshold)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "training_history.png", dpi=200)
    plt.close()

    print("Training complete.")
    print(f"Normal validation images: {len(x_val)}")
    print(f"95th-percentile threshold: {threshold:.8f}")
    print(f"Model saved to: {ARTIFACTS / 'autoencoder.keras'}")
    print(f"Threshold saved to: {ARTIFACTS / 'threshold.npy'}")


if __name__ == "__main__":
    main()
