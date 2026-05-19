from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from scripts.progan.core import (
    ProgressiveDiscriminator,
    ProgressiveGenerator,
    ProgressivePatchDataset,
    ProGanSettings,
)


def paper_batch_size(depth: int) -> int:
    """Return Ruiz-Casado et al.'s ProGAN batch schedule."""
    return {1: 64, 2: 64, 3: 32, 4: 16, 5: 4, 6: 2, 7: 1}[depth]


def train_class_progan(
    image_paths: list[Path], settings: ProGanSettings, device: torch.device, seed: int
) -> ProgressiveGenerator:
    """Train one class-specific ProGAN with progressive resolution growth."""
    torch.manual_seed(seed)
    generator, discriminator, opt_g, opt_d = _build_models(settings, device)
    criterion = nn.BCEWithLogitsLoss()
    for depth in range(1, settings.max_depth + 1):
        _train_depth(
            image_paths,
            settings,
            device,
            depth,
            generator,
            discriminator,
            criterion,
            opt_d,
            opt_g,
        )
    return generator


def _build_models(
    settings: ProGanSettings, device: torch.device
) -> tuple[
    ProgressiveGenerator,
    ProgressiveDiscriminator,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    generator = ProgressiveGenerator(
        settings.latent_dim, settings.max_depth, settings.base_channels
    ).to(device)
    discriminator = ProgressiveDiscriminator(
        settings.max_depth, settings.base_channels
    ).to(device)
    opt_g = torch.optim.Adam(
        generator.parameters(), lr=settings.learning_rate, betas=(settings.beta1, 0.999)
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=settings.learning_rate,
        betas=(settings.beta1, 0.999),
    )
    return generator, discriminator, opt_g, opt_d


def _train_depth(
    image_paths: list[Path],
    settings: ProGanSettings,
    device: torch.device,
    depth: int,
    generator: ProgressiveGenerator,
    discriminator: ProgressiveDiscriminator,
    criterion: nn.Module,
    opt_d: torch.optim.Optimizer,
    opt_g: torch.optim.Optimizer,
) -> None:
    loader = _depth_loader(image_paths, depth)
    for epoch in range(settings.epochs_per_depth):
        alpha = _fade_alpha(epoch, settings.epochs_per_depth, settings.fade_in_fraction)
        for real in loader:
            _train_step(
                generator,
                discriminator,
                real.to(device),
                depth,
                alpha,
                criterion,
                opt_d,
                opt_g,
                settings,
            )


def _depth_loader(image_paths: list[Path], depth: int) -> DataLoader:
    resolution = 2 ** (depth + 1)
    return DataLoader(
        ProgressivePatchDataset(image_paths, resolution),
        batch_size=min(paper_batch_size(depth), len(image_paths)),
        shuffle=True,
    )


def _fade_alpha(epoch: int, epochs: int, fade_fraction: float) -> float:
    fade_epochs = max(1, round(epochs * fade_fraction))
    return min(1.0, (epoch + 1) / fade_epochs)


def _train_step(
    generator: ProgressiveGenerator,
    discriminator: ProgressiveDiscriminator,
    real: torch.Tensor,
    depth: int,
    alpha: float,
    criterion: nn.Module,
    opt_d: torch.optim.Optimizer,
    opt_g: torch.optim.Optimizer,
    settings: ProGanSettings,
) -> None:
    batch_size = len(real)
    device = real.device
    noise = torch.randn(batch_size, settings.latent_dim, 1, 1, device=device)
    fake = generator(noise, depth, alpha).detach()
    loss_d = criterion(
        discriminator(real, depth, alpha), torch.ones(batch_size, device=device)
    )
    loss_d += criterion(
        discriminator(fake, depth, alpha), torch.zeros(batch_size, device=device)
    )
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()

    noise = torch.randn(batch_size, settings.latent_dim, 1, 1, device=device)
    fake = generator(noise, depth, alpha)
    loss_g = criterion(
        discriminator(fake, depth, alpha), torch.ones(batch_size, device=device)
    )
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()


def write_generated_images(
    generator: ProgressiveGenerator,
    output_dir: Path,
    class_name: str,
    settings: ProGanSettings,
    device: torch.device,
    n_images: int,
) -> list[dict[str, object]]:
    """Write final-resolution generated patches and return manifest rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    generator.eval()
    with torch.no_grad():
        for index in range(n_images):
            noise = torch.randn(1, settings.latent_dim, 1, 1, device=device)
            image = generator(noise, settings.max_depth, 1.0)[0].cpu()
            path = output_dir / f"{class_name}_{index:05d}.jpg"
            array = ((image.clamp(-1, 1) + 1.0) * 127.5).byte()
            Image.fromarray(array.permute(1, 2, 0).numpy()).save(path, quality=95)
            rows.append({"cancer_type": class_name, "image_path": str(path)})
    return rows
