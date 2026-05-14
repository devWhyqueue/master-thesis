from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.synthetic.image_gan_data import GanSettings


class PatchDataset(Dataset):
    """Load histopathology patch images for one class."""

    def __init__(self, image_paths: list[Path], image_size: int) -> None:
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1)


class Generator(nn.Module):
    """Small DCGAN-style generator for patch synthesis."""

    def __init__(self, latent_dim: int, channels: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 8, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Generate image tensors in [-1, 1]."""
        return self.model(noise)


class Discriminator(nn.Module):
    """Small DCGAN discriminator for patch synthesis."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 4, channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one real/fake logit per image."""
        return self.model(images).squeeze(1)


def train_class_gan(
    image_paths: list[Path],
    settings: GanSettings,
    device: torch.device,
    seed: int,
) -> Generator:
    """Train one per-class patch GAN."""
    torch.manual_seed(seed)
    dataset = PatchDataset(image_paths, settings.image_size)
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True)
    generator = Generator(settings.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt_g = _optimizer(generator, settings)
    opt_d = _optimizer(discriminator, settings)
    for _ in range(settings.epochs):
        _train_epoch(
            loader, discriminator, generator, criterion, opt_d, opt_g, settings, device
        )
    return generator


def _optimizer(model: nn.Module, settings: GanSettings) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        betas=(settings.beta1, 0.999),
    )


def _train_epoch(
    loader: DataLoader,
    discriminator: Discriminator,
    generator: Generator,
    criterion: nn.Module,
    opt_d: torch.optim.Optimizer,
    opt_g: torch.optim.Optimizer,
    settings: GanSettings,
    device: torch.device,
) -> None:
    for real_images in loader:
        real_images = real_images.to(device)
        _train_discriminator_step(
            discriminator, generator, real_images, criterion, opt_d, settings, device
        )
        _train_generator_step(
            discriminator,
            generator,
            len(real_images),
            criterion,
            opt_g,
            settings,
            device,
        )


def _train_discriminator_step(
    discriminator: Discriminator,
    generator: Generator,
    real_images: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    settings: GanSettings,
    device: torch.device,
) -> None:
    batch_size = len(real_images)
    real_targets = torch.ones(batch_size, device=device)
    fake_targets = torch.zeros(batch_size, device=device)
    noise = torch.randn(batch_size, settings.latent_dim, 1, 1, device=device)
    fake_images = generator(noise).detach()
    loss = criterion(discriminator(real_images), real_targets) + criterion(
        discriminator(fake_images), fake_targets
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _train_generator_step(
    discriminator: Discriminator,
    generator: Generator,
    batch_size: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    settings: GanSettings,
    device: torch.device,
) -> None:
    targets = torch.ones(batch_size, device=device)
    noise = torch.randn(batch_size, settings.latent_dim, 1, 1, device=device)
    loss = criterion(discriminator(generator(noise)), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def write_generated_images(
    generator: Generator,
    output_dir: Path,
    class_name: str,
    settings: GanSettings,
    device: torch.device,
) -> list[dict[str, object]]:
    """Write generated JPEG patches and return manifest rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    generator.eval()
    remaining = settings.generated_patches_per_class
    written = 0
    with torch.no_grad():
        while remaining > 0:
            batch_size = min(settings.batch_size, remaining)
            noise = torch.randn(batch_size, settings.latent_dim, 1, 1, device=device)
            images = generator(noise).cpu()
            for image in images:
                path = output_dir / f"{class_name}_{written:05d}.jpg"
                _save_image(image, path)
                rows.append({"cancer_type": class_name, "image_path": str(path)})
                written += 1
            remaining -= batch_size
    return rows


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    array = ((tensor.clamp(-1, 1) + 1.0) * 127.5).byte()
    image = Image.fromarray(array.permute(1, 2, 0).numpy())
    image.save(path, quality=95)
