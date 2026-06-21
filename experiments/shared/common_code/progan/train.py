from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from common_code.progan.core import (
    ProgressiveDiscriminator,
    ProgressiveGenerator,
    ProgressivePatchDataset,
    ProGanSettings,
    decode_raw_images,
)

_WGAN_GP_LAMBDA = 10.0
_WGAN_GP_DRIFT = 0.001


def paper_batch_size(depth: int) -> int:
    """Return the Ruiz-Casado style ProGAN batch schedule."""
    return {1: 64, 2: 64, 3: 32, 4: 16, 5: 4, 6: 2, 7: 1}[depth]


def train_class_progan(
    image_paths: list[Path], settings: ProGanSettings, device: torch.device, seed: int
) -> tuple[dict[int, ProgressiveGenerator], list[dict[str, object]]]:
    """Train one class-specific ProGAN and return final-depth snapshots."""
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    models = _build_models(settings, device)
    raw_images = decode_raw_images(image_paths)
    diagnostics: list[dict[str, object]] = []
    context = (image_paths, raw_images, settings, device)
    for depth in range(1, settings.max_depth + 1):
        if depth < settings.max_depth:
            diagnostics.append(_train_depth(context, depth, models))
            continue
        final_diag, snapshots = _train_final(context, depth, models)
        diagnostics.append(final_diag)
        return snapshots, diagnostics
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
    optimizer_generator = torch.optim.Adam(
        generator.parameters(),
        lr=settings.learning_rate,
        betas=(settings.beta1, 0.999),
    )
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=settings.learning_rate,
        betas=(settings.beta1, 0.999),
    )
    return generator, discriminator, optimizer_generator, optimizer_discriminator


def _train_depth(context: tuple, depth: int, models: tuple) -> dict[str, object]:
    image_paths, raw_images, settings, device = context
    loader = _depth_loader(raw_images, depth)
    sum_loss_d = torch.zeros(1, device=device)
    sum_loss_g = torch.zeros(1, device=device)
    n_steps = 0
    for epoch in range(settings.epochs_per_depth):
        alpha = _fade_alpha(epoch, settings.epochs_per_depth, settings.fade_in_fraction)
        for real in loader:
            loss_d, loss_g = _train_step(
                models, real.to(device), depth, alpha, settings
            )
            sum_loss_d += loss_d
            sum_loss_g += loss_g
            n_steps += 1
    denom = max(n_steps, 1)
    return _depth_diagnostics(
        image_paths,
        depth,
        settings.epochs_per_depth,
        loader.batch_size,
        (sum_loss_d / denom).item() if n_steps else None,
        (sum_loss_g / denom).item() if n_steps else None,
        n_steps,
    )


def _train_final(
    context: tuple, depth: int, models: tuple
) -> tuple[dict, dict[int, ProgressiveGenerator]]:
    image_paths, raw_images, settings, device = context
    generator = models[0]
    loader = _depth_loader(raw_images, depth)
    grid = sorted(settings.final_depth_epoch_grid)
    grid_set = set(grid)
    fade_epochs = max(1, round(settings.epochs_per_depth * settings.fade_in_fraction))
    sum_loss_d = torch.zeros(1, device=device)
    sum_loss_g = torch.zeros(1, device=device)
    n_steps = 0
    snapshots: dict[int, ProgressiveGenerator] = {}
    for epoch in range(grid[-1]):
        alpha = min(1.0, (epoch + 1) / fade_epochs)
        for real in loader:
            loss_d, loss_g = _train_step(
                models, real.to(device), depth, alpha, settings
            )
            sum_loss_d += loss_d
            sum_loss_g += loss_g
            n_steps += 1
        if (epoch + 1) in grid_set:
            snapshots[epoch + 1] = copy.deepcopy(generator)
    denom = max(n_steps, 1)
    diagnostics = _depth_diagnostics(
        image_paths,
        depth,
        grid[-1],
        loader.batch_size,
        (sum_loss_d / denom).item() if n_steps else None,
        (sum_loss_g / denom).item() if n_steps else None,
        n_steps,
    )
    return diagnostics, snapshots


def _depth_loader(raw_images: list, depth: int) -> DataLoader:
    num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "0"))
    return DataLoader(
        ProgressivePatchDataset([], 2 ** (depth + 1), raw=raw_images),
        batch_size=min(paper_batch_size(depth), len(raw_images)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() and num_workers > 0,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _fade_alpha(epoch: int, epochs: int, fade_fraction: float) -> float:
    fade_epochs = max(1, round(epochs * fade_fraction))
    return min(1.0, (epoch + 1) / fade_epochs)


def _gradient_penalty(
    discriminator: ProgressiveDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    depth: int,
    alpha: float,
) -> torch.Tensor:
    batch_size = len(real)
    eps = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interp = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    scores = discriminator(interp, depth, alpha)
    gradient = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return _WGAN_GP_LAMBDA * ((gradient.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def _train_step(
    models: tuple,
    real: torch.Tensor,
    depth: int,
    alpha: float,
    settings: ProGanSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator, discriminator, optimizer_generator, optimizer_discriminator = models
    batch_size = len(real)
    noise = torch.randn(batch_size, settings.latent_dim, 1, 1, device=real.device)
    fake = generator(noise, depth, alpha).detach()
    real_scores = discriminator(real, depth, alpha)
    fake_scores = discriminator(fake, depth, alpha)
    loss_discriminator = (
        fake_scores.mean()
        - real_scores.mean()
        + _WGAN_GP_DRIFT * real_scores.pow(2).mean()
        + _gradient_penalty(discriminator, real, fake, depth, alpha)
    )
    optimizer_discriminator.zero_grad()
    loss_discriminator.backward()
    optimizer_discriminator.step()
    noise = torch.randn(batch_size, settings.latent_dim, 1, 1, device=real.device)
    fake = generator(noise, depth, alpha)
    loss_generator = -discriminator(fake, depth, alpha).mean()
    optimizer_generator.zero_grad()
    loss_generator.backward()
    optimizer_generator.step()
    return loss_discriminator.detach(), loss_generator.detach()


def _depth_diagnostics(
    image_paths: list[Path],
    depth: int,
    epochs: int,
    batch_size: int | None,
    discriminator_loss_mean: float | None,
    generator_loss_mean: float | None,
    steps: int,
) -> dict[str, object]:
    return {
        "batch_size": int(batch_size or 0),
        "depth": depth,
        "discriminator_loss_mean": discriminator_loss_mean,
        "epochs": epochs,
        "generator_loss_mean": generator_loss_mean,
        "n_real_images": len(image_paths),
        "resolution": 2 ** (depth + 1),
        "steps": steps,
    }


def write_generated_images(
    generator: ProgressiveGenerator,
    output_dir: Path,
    class_name: str,
    settings: ProGanSettings,
    device: torch.device,
    n_images: int,
) -> list[dict[str, object]]:
    """Write generated JPEGs and return manifest rows."""
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
