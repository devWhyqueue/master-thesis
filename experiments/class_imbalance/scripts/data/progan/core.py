from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ProGanSettings:
    image_size: int
    latent_dim: int
    epochs_per_depth: int
    learning_rate: float
    beta1: float
    max_real_patches_per_class: int
    balance_target: str
    max_classes: int | None
    fade_in_fraction: float
    base_channels: int
    final_depth_epoch_grid: tuple[int, ...] = (10, 25, 50)

    @property
    def max_depth(self) -> int:
        """Return the final progressive depth for the configured image size."""
        return int(np.log2(self.image_size)) - 1


class ProgressivePatchDataset(Dataset):
    """Load one class of patches at the active ProGAN resolution."""

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


class PixelNorm(nn.Module):
    """Normalize generator activations across channels."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        """Apply per-pixel feature-vector normalization."""
        denominator = torch.sqrt(
            torch.mean(x.pow(2), dim=1, keepdim=True) + self.epsilon
        )
        return x / denominator


class MinibatchStdDev(nn.Module):
    """Append ProGAN minibatch standard deviation as one feature channel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        """Append one constant channel with batch-level feature variation."""
        if len(x) == 1:
            stddev = torch.zeros(1, 1, x.shape[2], x.shape[3], device=x.device)
            return torch.cat([x, stddev], dim=1)
        deviation = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-8)
        mean_deviation = deviation.mean().view(1, 1, 1, 1)
        stddev = mean_deviation.repeat(len(x), 1, x.shape[2], x.shape[3])
        return torch.cat([x, stddev], dim=1)


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        """Upsample and refine one generator feature map."""
        return self.block(x)


class ProgressiveGenerator(nn.Module):
    """ProGAN generator with progressive growth and fade-in."""

    def __init__(self, latent_dim: int, max_depth: int, base_channels: int) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels, 4),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            PixelNorm(),
        )
        channels = [max(base_channels // (2**idx), 32) for idx in range(max_depth)]
        self.blocks = nn.ModuleList(
            GeneratorBlock(channels[idx - 1], channels[idx])
            for idx in range(1, max_depth)
        )
        self.to_rgb = nn.ModuleList(nn.Conv2d(channel, 3, 1) for channel in channels)

    def forward(self, noise: torch.Tensor, depth: int, alpha: float) -> torch.Tensor:  # noqa
        """Generate images at one active progressive depth."""
        x = self.initial(noise)
        if depth == 1:
            return torch.tanh(self.to_rgb[0](x))
        previous = x
        for block_idx in range(depth - 1):
            previous = x
            x = self.blocks[block_idx](x)
        current = self.to_rgb[depth - 1](x)
        previous_rgb = nn.functional.interpolate(
            self.to_rgb[depth - 2](previous), scale_factor=2, mode="nearest"
        )
        return torch.tanh(alpha * current + (1.0 - alpha) * previous_rgb)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        """Downsample one discriminator feature map."""
        return self.block(x)


class ProgressiveDiscriminator(nn.Module):
    """ProGAN discriminator matching the generator's active depth."""

    def __init__(self, max_depth: int, base_channels: int) -> None:
        super().__init__()
        channels = [max(base_channels // (2**idx), 32) for idx in range(max_depth)]
        self.from_rgb = nn.ModuleList(nn.Conv2d(3, channel, 1) for channel in channels)
        self.blocks = nn.ModuleList(
            DiscriminatorBlock(channels[idx], channels[idx - 1])
            for idx in range(1, max_depth)
        )
        self.minibatch_stddev = MinibatchStdDev()
        self.final = nn.Sequential(
            nn.Conv2d(channels[0] + 1, channels[0], 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[0], 1, 4),
            nn.Flatten(),
        )

    def forward(self, images: torch.Tensor, depth: int, alpha: float) -> torch.Tensor:  # noqa
        """Score images at one active progressive depth."""
        if depth == 1:
            x = self.minibatch_stddev(self.from_rgb[0](images))
            return self.final(x).squeeze(1)
        x = self.blocks[depth - 2](self.from_rgb[depth - 1](images))
        downsampled = nn.functional.avg_pool2d(images, 2)
        previous = self.from_rgb[depth - 2](downsampled)
        x = alpha * x + (1.0 - alpha) * previous
        for block_idx in range(depth - 3, -1, -1):
            x = self.blocks[block_idx](x)
        return self.final(self.minibatch_stddev(x)).squeeze(1)
