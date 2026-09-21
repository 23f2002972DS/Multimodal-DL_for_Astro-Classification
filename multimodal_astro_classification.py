"""Multimodal deep learning for SDSS astronomical classification.

The pipeline downloads paired SDSS image cutouts and spectra, preprocesses them,
trains a spatial-spectral fusion model, and optionally saves diagnostic plots.
Run ``python multimodal_astro_classification.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from astropy import coordinates as coords
from astropy.io import fits
import astropy.units as u
from astroquery.sdss import SDSS
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", module="astropy")

BANDS = ("g", "r", "z")
CLASS_NAMES = ("STAR", "GALAXY", "QSO")
LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Config:
    """Runtime configuration for data preparation and training."""

    data_dir: Path = Path("multimodal_dataset")
    output_dir: Path = Path("outputs")
    sample_limit: int = 20
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0


def set_seed(seed: int) -> None:
    """Make the training run as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sdss_query(sample_limit: int) -> str:
    """Build the query used to find objects with both SDSS modalities."""
    return f"""
    SELECT TOP {sample_limit}
        p.objid, p.ra, p.dec,
        s.specobjid, s.plate, s.mjd, s.fiberid, s.class, s.z
    FROM PhotoObjAll AS p
    JOIN SpecObjAll AS s ON p.objid = s.bestobjid
    WHERE s.class IN ('GALAXY', 'QSO', 'STAR')
      AND p.clean = 1
      AND s.zWarning = 0
    """


def required_paths(data_dir: Path, uid: str) -> list[Path]:
    """Return every file required for one paired sample."""
    image_paths = [
        data_dir / "images_2d" / f"img_{uid}_{band}.fits" for band in BANDS
    ]
    return [data_dir / "spectra_1d" / f"spec_{uid}.fits", *image_paths]


def download_sdss_dataset(config: Config) -> tuple[list[str], list[str]]:
    """Download paired SDSS spectra and image cutouts."""
    spectra_dir = config.data_dir / "spectra_1d"
    images_dir = config.data_dir / "images_2d"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Querying the SDSS database...")
    matches = SDSS.query_sql(build_sdss_query(config.sample_limit))
    downloaded_uids: list[str] = []
    downloaded_labels: list[str] = []

    for row in matches:
        obj_class = str(row["class"])
        uid = f"{row['plate']}-{row['mjd']}-{row['fiberid']}"

        try:
            spectra = SDSS.get_spectra(
                plate=row["plate"],
                mjd=row["mjd"],
                fiberID=row["fiberid"],
            )
            if not spectra:
                raise RuntimeError("SDSS returned no spectrum")
            spectra[0].writeto(spectra_dir / f"spec_{uid}.fits", overwrite=True)

            position = coords.SkyCoord(
                row["ra"] * u.deg,
                row["dec"] * u.deg,
                frame="icrs",
            )
            images = SDSS.get_images(coordinates=position, band=list(BANDS))
            if images is None or len(images) != len(BANDS):
                raise RuntimeError("SDSS returned incomplete image cutouts")
            for image, band in zip(images, BANDS):
                image.writeto(images_dir / f"img_{uid}_{band}.fits", overwrite=True)

            if not all(path.exists() for path in required_paths(config.data_dir, uid)):
                raise RuntimeError("one or more downloaded files are missing")
            downloaded_uids.append(uid)
            downloaded_labels.append(obj_class)
            print(f"Downloaded {obj_class}: {uid}")
        except Exception as error:
            print(f"Skipped {uid}: {error}")

    print(f"Prepared {len(downloaded_uids)} paired samples.")
    return downloaded_uids, downloaded_labels


def find_existing_samples(data_dir: Path) -> tuple[list[str], list[str]]:
    """Load labels from an existing dataset directory.

    Labels are stored in ``labels.csv`` with columns ``uid,class``. The file is
    intentionally simple so it can be inspected and versioned independently.
    """
    labels_path = data_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"No {labels_path} found. Run with --download to create the dataset."
        )

    records = np.genfromtxt(labels_path, delimiter=",", names=True, dtype=str)
    if records.ndim == 0:
        records = np.array([records])
    uids: list[str] = []
    labels: list[str] = []
    for record in records:
        uid = str(record["uid"])
        label = str(record["class"])
        if label in LABEL_MAP and all(
            path.exists() for path in required_paths(data_dir, uid)
        ):
            uids.append(uid)
            labels.append(label)
    return uids, labels


class AstroMultimodalDataset(Dataset):
    """Load paired SDSS images and spectra as PyTorch tensors."""

    def __init__(
        self,
        data_dir: Path,
        uids: Iterable[str],
        labels: Iterable[str],
        image_size: int = 64,
        fixed_spec_len: int = 3800,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.uids = list(uids)
        self.labels = list(labels)
        self.image_size = image_size
        self.fixed_spec_len = fixed_spec_len

    def __len__(self) -> int:
        return len(self.uids)

    def _load_image(self, uid: str) -> np.ndarray:
        channels = []
        half_size = self.image_size // 2
        for band in BANDS:
            path = self.data_dir / "images_2d" / f"img_{uid}_{band}.fits"
            with fits.open(path) as hdul:
                image = np.nan_to_num(np.asarray(hdul[0].data, dtype=np.float32))
            center_y, center_x = np.array(image.shape[:2]) // 2
            image = image[
                center_y - half_size : center_y + half_size,
                center_x - half_size : center_x + half_size,
            ]
            if image.shape != (self.image_size, self.image_size):
                raise ValueError(f"Unexpected image shape in {path}: {image.shape}")
            image = (image - image.mean()) / (image.std() + 1e-8)
            channels.append(image)
        return np.stack(channels)

    def _load_spectrum(self, uid: str) -> np.ndarray:
        path = self.data_dir / "spectra_1d" / f"spec_{uid}.fits"
        with fits.open(path) as hdul:
            spectrum = np.nan_to_num(
                np.asarray(hdul[1].data["flux"], dtype=np.float32)
            )
        spectrum = spectrum[: self.fixed_spec_len]
        if len(spectrum) < self.fixed_spec_len:
            spectrum = np.pad(spectrum, (0, self.fixed_spec_len - len(spectrum)))
        return (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid = self.uids[index]
        image = torch.tensor(self._load_image(uid), dtype=torch.float32)
        spectrum = torch.tensor(self._load_spectrum(uid), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(LABEL_MAP[self.labels[index]], dtype=torch.long)
        return image, spectrum, label


class SpectralEncoder1D(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.AdaptiveAvgPool1d(16),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).permute(0, 2, 1)


class SpatialEncoder2D(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.stem(inputs).flatten(2).permute(0, 2, 1)


class AstroCrossAttentionFusion(nn.Module):
    """Fuse image and spectrum tokens with cross-attention."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.spatial_encoder = SpatialEncoder2D(embed_dim=embed_dim)
        self.spectral_encoder = SpectralEncoder1D(embed_dim=embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, image: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        spatial_tokens = self.spatial_encoder(image)
        spectral_tokens = self.spectral_encoder(spectrum)
        attention_output, _ = self.cross_attention(
            query=spatial_tokens,
            key=spectral_tokens,
            value=spectral_tokens,
        )
        fused = self.norm1(spatial_tokens + attention_output)
        fused = self.norm2(fused + self.ffn(fused))
        return self.classifier(fused.mean(dim=1))


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> tuple[list[float], list[float]]:
    """Train the model and return loss and accuracy per epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    accuracies: list[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, spectra, targets in dataloader:
            images, spectra, targets = (
                images.to(device),
                spectra.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()
            outputs = model(images, spectra)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / max(len(dataloader), 1)
        epoch_accuracy = 100.0 * correct / max(total, 1)
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%"
        )
    return losses, accuracies


def plot_learning_curves(
    losses: list[float], accuracies: list[float], output_dir: Path
) -> None:
    """Save the training loss and accuracy plot."""
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(losses) + 1)
    loss_axis.plot(epochs, losses, marker="o", color="purple")
    loss_axis.set(title="Training Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    loss_axis.grid(True, linestyle="--", alpha=0.6)
    accuracy_axis.plot(epochs, accuracies, marker="s", color="teal")
    accuracy_axis.set(title="Classification Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    accuracy_axis.set_ylim(0, 105)
    accuracy_axis.grid(True, linestyle="--", alpha=0.6)
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(figure)


def plot_sample(data_dir: Path, uid: str, label: str, output_dir: Path) -> None:
    """Save one spectrum plot and one multi-band image plot."""
    spectrum_path = data_dir / "spectra_1d" / f"spec_{uid}.fits"
    with fits.open(spectrum_path) as hdul:
        spectrum_data = hdul[1].data
        flux = spectrum_data["flux"]
        wavelength = 10 ** spectrum_data["loglam"]

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(wavelength, flux, color="crimson", linewidth=0.8)
    axis.set(title=f"SDSS Spectrum | Class: {label} | ID: {uid}", xlabel="Wavelength (Angstroms)", ylabel="Flux")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "sample_spectrum.png", dpi=150)
    plt.close(figure)

    channels = []
    for band in BANDS:
        with fits.open(data_dir / "images_2d" / f"img_{uid}_{band}.fits") as hdul:
            channels.append(np.nan_to_num(hdul[0].data))
    image = np.dstack(channels)
    low, high = np.percentile(image, [0.5, 99.5])
    normalized = np.clip((image - low) / (high - low + 1e-8), 0, 1)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(normalized)
    axis.set(title=f"SDSS Multi-Band Image | Class: {label}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "sample_image.png", dpi=150)
    plt.close(figure)


def draw_labeled_neural_net(axis: plt.Axes) -> None:
    """Draw a conceptual overview of the multimodal network."""
    graph = nx.DiGraph()
    layers = [
        ["g-band\nImage", "r-band\nImage", "z-band\nImage", "1D\nSpectrum"],
        ["Feature\nExtraction"] * 5,
        ["Cross-Attention\nFusion"] * 5,
        ["STAR", "GALAXY", "QSO"],
    ]
    positions = {}
    labels = {}
    colors = []
    for layer_index, layer in enumerate(layers):
        top = 1.2 * (len(layer) - 1) / 2
        for node_index, label in enumerate(layer):
            node_id = f"L{layer_index}_N{node_index}"
            graph.add_node(node_id)
            positions[node_id] = (layer_index * 2.5, top - node_index * 1.2)
            labels[node_id] = label
            colors.append("#98FB98" if layer_index == 0 else "#FFA07A" if layer_index == 3 else "#87CEFA")
            if layer_index:
                graph.add_edges_from(
                    (f"L{layer_index - 1}_N{k}", node_id)
                    for k in range(len(layers[layer_index - 1]))
                )
    nx.draw(graph, positions, ax=axis, labels=labels, node_size=3500, node_color=colors, arrows=True, font_weight="bold")
    axis.axis("off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download fresh SDSS data")
    parser.add_argument("--data-dir", type=Path, default=Config.data_dir)
    parser.add_argument("--output-dir", type=Path, default=Config.output_dir)
    parser.add_argument("--sample-limit", type=int, default=Config.sample_limit)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--skip-plots", action="store_true", help="Skip diagnostic plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.download:
        uids, labels = download_sdss_dataset(config)
        labels_path = config.data_dir / "labels.csv"
        np.savetxt(
            labels_path,
            np.column_stack((uids, labels)),
            delimiter=",",
            header="uid,class",
            comments="",
            fmt="%s",
        )
    else:
        uids, labels = find_existing_samples(config.data_dir)

    if not uids:
        raise RuntimeError("No complete paired samples are available.")

    dataset = AstroMultimodalDataset(config.data_dir, uids, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    model = AstroCrossAttentionFusion().to(device)
    losses, accuracies = train_model(
        model, dataloader, device, config.epochs, config.learning_rate
    )
    torch.save(model.state_dict(), config.output_dir / "astro_fusion_model.pt")

    if not args.skip_plots:
        plot_learning_curves(losses, accuracies, config.output_dir)
        plot_sample(config.data_dir, uids[0], labels[0], config.output_dir)
        figure, axis = plt.subplots(figsize=(11, 7))
        draw_labeled_neural_net(axis)
        figure.tight_layout()
        figure.savefig(config.output_dir / "network_architecture.png", dpi=150)
        plt.close(figure)

    print(f"Saved outputs to {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
