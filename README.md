# Medical X-Ray Image Reconstruction and Anomaly Detection Using Autoencoder

An autoencoder-based deep learning project for reconstructing chest X-ray images and detecting anomalous images using reconstruction error.

## Objective

The model learns visual patterns of **normal chest X-ray images** using an encoder-decoder architecture. For a new image:

1. Convert the X-ray to grayscale.
2. Resize it to **128 × 128** pixels.
3. Normalize pixel values to **0–1**.
4. Encode the image into a compact latent representation.
5. Reconstruct the image with the decoder.
6. Calculate Mean Squared Error (MSE) between input and reconstruction.
7. Use the **95th percentile of normal reconstruction errors** as the anomaly threshold.
8. Error > threshold → `ANOMALY`; otherwise → `NORMAL`.

This follows the workflow described in the supplied project report.

## Repository structure

```text
medical-xray-autoencoder/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
├── src/
│   ├── __init__.py
│   └── autoencoder.py
├── artifacts/
│   └── README.md
└── data/
    └── chest_xray/
        ├── train/NORMAL/
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/
```

The dataset and trained model weights are intentionally not included in this repository.

## Dataset

Place your chest X-ray dataset in this structure:

```text
data/chest_xray/
├── train/
│   └── NORMAL/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Only `train/NORMAL` images are used to learn the normal reconstruction pattern.

## Installation

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

This generates:

```text
artifacts/
├── autoencoder.keras
├── threshold.npy
└── training_history.png
```

## Evaluate

```bash
python evaluate.py
```

This generates:

```text
artifacts/
├── reconstruction_error_distribution.png
└── confusion_matrix.png
```

## Run the Gradio application

After training:

```bash
python app.py
```

The interface displays the original X-ray, reconstructed X-ray, prediction, reconstruction error, and threshold.

## Team

- Jeevika K
- Suruthika C D
- Dharun K C
- Saravana Kumar U

## Medical disclaimer

This is an academic research prototype and **not a medical diagnostic system**. Predictions must not be used for clinical diagnosis or treatment decisions.
