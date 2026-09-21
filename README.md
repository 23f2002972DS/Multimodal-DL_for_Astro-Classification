# Multimodal-DL_for_Astro-Classification

A PyTorch spatial-spectral fusion model for classifying SDSS objects as stars, galaxies, or quasars. The pipeline downloads paired SDSS image cutouts and spectra, encodes both modalities, and fuses them with cross-attention.

## Setup

Use Python 3.12 or 3.13. PyTorch currently fails to load its native Windows DLLs in the Python 3.14 environment, so Python 3.14 is not supported for this project.

Create and activate a virtual environment in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the CUDA-compatible PyTorch build recommended at [pytorch.org](https://pytorch.org/get-started/locally/) instead of the generic `torch` package in `requirements.txt`.

## Run

### First Run

From the project folder in PowerShell, activate the environment and download a
small test dataset:

```powershell
.\.venv\Scripts\Activate.ps1
python multimodal_astro_classification.py --download --sample-limit 5 --epochs 1
```

This queries SDSS, downloads five paired image/spectrum samples, performs one
training epoch, and saves the results in `outputs/`. Open the generated PNG
files from the `outputs` folder in VS Code to view the plots.

After the test works, run a larger experiment:

```powershell
python multimodal_astro_classification.py --download --sample-limit 20 --epochs 10
```

Download a small SDSS sample and train for 10 epochs:

```powershell
python multimodal_astro_classification.py --download
```

The first run needs internet access and creates `multimodal_dataset/`. Model weights and plots are written to `outputs/`. Both folders are ignored by Git because they can be regenerated and may be large.

Train again from the downloaded data without querying SDSS:

```powershell
python multimodal_astro_classification.py --epochs 10
```

Useful options:

```powershell
python multimodal_astro_classification.py --help
python multimodal_astro_classification.py --download --sample-limit 100 --epochs 20
python multimodal_astro_classification.py --epochs 1 --skip-plots
```

The script uses CUDA automatically when PyTorch detects a compatible GPU; otherwise it runs on CPU.

## Outputs

- `outputs/astro_fusion_model.pt`: trained model weights
- `outputs/learning_curves.png`: loss and accuracy curves
- `outputs/sample_spectrum.png`: example spectrum
- `outputs/sample_image.png`: example multi-band image
- `outputs/network_architecture.png`: conceptual model diagram

## Project Layout

```text
.
├── multimodal_astro_classification.py
├── requirements.txt
├── README.md
└── .gitignore
```
