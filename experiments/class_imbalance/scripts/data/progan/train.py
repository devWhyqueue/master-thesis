from __future__ import annotations

import copy
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from scripts.data.progan.core import (
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
) -> tuple[dict[int, ProgressiveGenerator], list[dict[str, object]]]:
    """Train one class-specific ProGAN; return generator snapshots keyed by final-depth epoch.

    Depths 1..max_depth-1 train for settings.epochs_per_depth each.  The final depth
    trains to max(settings.final_depth_epoch_grid) and the generator state is captured
    after each epoch in the grid.  Fade-in at the final depth is pinned to the schedule
    defined by settings.epochs_per_depth so all snapshots share identical early-fade
    behavior regardless of the longest training run.
    """
    torch.manual_seed(seed)
    generator, discriminator, opt_g, opt_d = _build_models(settings, device)
    criterion = nn.BCEWithLogitsLoss()
    diagnostics: list[dict[str, object]] = []
    for depth in range(1, settings.max_depth + 1):
        if depth < settings.max_depth:
            diagnostics.append(
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
            )
        else:
            diag, snapshots = _train_final_depth(
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
            diagnostics.append(diag)
            return snapshots, diagnostics
    # Reached only when max_depth == 0 (should never happen in practice).
    return {}, diagnostics


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
) -> dict[str, object]:
    loader = _depth_loader(image_paths, depth)
    discriminator_losses: list[float] = []
    generator_losses: list[float] = []
    for epoch in range(settings.epochs_per_depth):
        alpha = _fade_alpha(epoch, settings.epochs_per_depth, settings.fade_in_fraction)
        for real in loader:
            loss_d, loss_g = _train_step(
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
            discriminator_losses.append(loss_d)
            generator_losses.append(loss_g)
    return _depth_diagnostics(
        image_paths,
        depth,
        settings.epochs_per_depth,
        loader.batch_size,
        discriminator_losses,
        generator_losses,
    )


def _train_final_depth(
    image_paths: list[Path],
    settings: ProGanSettings,
    device: torch.device,
    depth: int,
    generator: ProgressiveGenerator,
    discriminator: ProgressiveDiscriminator,
    criterion: nn.Module,
    opt_d: torch.optim.Optimizer,
    opt_g: torch.optim.Optimizer,
) -> tuple[dict[str, object], dict[int, ProgressiveGenerator]]:
    """Train the final progressive depth to max(grid) epochs and capture snapshots.

    The fade-in alpha schedule is pinned to settings.epochs_per_depth (the reference
    schedule) so that all snapshots see the same ramp regardless of total run length.
    """
    loader = _depth_loader(image_paths, depth)
    grid = sorted(settings.final_depth_epoch_grid)
    max_epochs = grid[-1]
    grid_set = set(grid)
    fade_epochs = max(1, round(settings.epochs_per_depth * settings.fade_in_fraction))

    discriminator_losses: list[float] = []
    generator_losses: list[float] = []
    snapshots: dict[int, ProgressiveGenerator] = {}

    for epoch in range(max_epochs):
        alpha = min(1.0, (epoch + 1) / fade_epochs)
        for real in loader:
            loss_d, loss_g = _train_step(
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
            discriminator_losses.append(loss_d)
            generator_losses.append(loss_g)
        if (epoch + 1) in grid_set:
            snapshots[epoch + 1] = copy.deepcopy(generator)

    diag = _depth_diagnostics(
        image_paths,
        depth,
        max_epochs,
        loader.batch_size,
        discriminator_losses,
        generator_losses,
    )
    return diag, snapshots


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
) -> tuple[float, float]:
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
    return float(loss_d.detach().cpu().item()), float(loss_g.detach().cpu().item())


def _depth_diagnostics(
    image_paths: list[Path],
    depth: int,
    epochs: int,
    batch_size: int | None,
    discriminator_losses: list[float],
    generator_losses: list[float],
) -> dict[str, object]:
    """Return training diagnostics for one progressive depth."""
    return {
        "batch_size": int(batch_size or 0),
        "depth": depth,
        "discriminator_loss_mean": _mean_loss(discriminator_losses),
        "epochs": epochs,
        "generator_loss_mean": _mean_loss(generator_losses),
        "n_real_images": len(image_paths),
        "resolution": 2 ** (depth + 1),
        "steps": len(discriminator_losses),
    }


def _mean_loss(values: list[float]) -> float | None:
    """Return a stable mean for optional loss traces."""
    return sum(values) / len(values) if values else None


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
