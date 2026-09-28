"""CUDA: curriculum of data augmentation, per-class levels of learning (Eq. cuda-aug/-lol)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torchvision.transforms.v2 import functional as F2

__all__ = ["augment", "cuda_batch", "update_levels"]

# Magnitude at full strength (s=S) for each of the K=10 ops; sign is randomized for the
# symmetric ones (rotate/translate/shear/color-factor deltas), fixed direction for the two
# that have no natural negative (blur, sharpen-toward-2x only goes one way from identity).
_MAX_ROTATE = 30.0
_MAX_TRANSLATE = 0.3
_MAX_SHEAR = 20.0
_MAX_COLOR = 0.5
_MAX_HUE = 0.5
_MAX_BLUR_SIGMA = 3.0
_MAX_SHARPNESS = 1.0


def _sign(generator: torch.Generator) -> float:
    """Random +-1, drawn from ``generator``."""
    device = generator.device if generator.device.type != "cpu" else "cpu"
    return (
        1.0 if torch.rand((), generator=generator, device=device).item() < 0.5 else -1.0
    )


def _mag(s: int, max_strength: int, max_k: float) -> float:
    """Magnitude function m_k(s) = s/S * max_k (Eq. cuda-aug)."""
    return s / max_strength * max_k


def _op_rotate(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    return F2.rotate(img, angle=_sign(g) * _mag(s, max_strength, _MAX_ROTATE))


def _op_translate_x(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    dx = _sign(g) * _mag(s, max_strength, _MAX_TRANSLATE) * img.shape[-1]
    return F2.affine(
        img, angle=0.0, translate=[int(dx), 0], scale=1.0, shear=[0.0, 0.0]
    )


def _op_translate_y(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    dy = _sign(g) * _mag(s, max_strength, _MAX_TRANSLATE) * img.shape[-2]
    return F2.affine(
        img, angle=0.0, translate=[0, int(dy)], scale=1.0, shear=[0.0, 0.0]
    )


def _op_shear(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    shear = _sign(g) * _mag(s, max_strength, _MAX_SHEAR)
    return F2.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[shear, 0.0])


def _op_brightness(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    factor = max(0.0, 1.0 + _sign(g) * _mag(s, max_strength, _MAX_COLOR))
    return F2.adjust_brightness(img, brightness_factor=factor)


def _op_contrast(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    factor = max(0.0, 1.0 + _sign(g) * _mag(s, max_strength, _MAX_COLOR))
    return F2.adjust_contrast(img, contrast_factor=factor)


def _op_saturation(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    factor = max(0.0, 1.0 + _sign(g) * _mag(s, max_strength, _MAX_COLOR))
    return F2.adjust_saturation(img, saturation_factor=factor)


def _op_hue(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    hue = _sign(g) * _mag(s, max_strength, _MAX_HUE)
    return F2.adjust_hue(img, hue_factor=max(-0.5, min(0.5, hue)))


def _op_blur(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    sigma = _mag(s, max_strength, _MAX_BLUR_SIGMA)
    if sigma <= 0:
        return img
    kernel = max(3, int(2 * round(3 * sigma) + 1))
    return F2.gaussian_blur(img, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def _op_sharpness(
    img: torch.Tensor, s: int, max_strength: int, g: torch.Generator
) -> torch.Tensor:
    factor = max(0.0, 1.0 + _mag(s, max_strength, _MAX_SHARPNESS))
    return F2.adjust_sharpness(img, sharpness_factor=factor)


_OPS: tuple[Callable[[torch.Tensor, int, int, torch.Generator], torch.Tensor], ...] = (
    _op_rotate,
    _op_translate_x,
    _op_translate_y,
    _op_shear,
    _op_brightness,
    _op_contrast,
    _op_saturation,
    _op_hue,
    _op_blur,
    _op_sharpness,
)


def augment(
    img: torch.Tensor, strength: int, max_strength: int, generator: torch.Generator
) -> torch.Tensor:
    """Apply ``strength`` ops drawn uniformly, each at magnitude m_k(strength) (Eq. cuda-aug).

    ``strength=0`` leaves the patch unchanged.
    """
    for _ in range(strength):
        k = int(
            torch.randint(
                len(_OPS), (1,), generator=generator, device=img.device
            ).item()
        )
        img = _OPS[k](img, strength, max_strength, generator)
    return img


def cuda_batch(
    images_u8: torch.Tensor,
    levels_per_row: torch.Tensor,
    p_aug: float,
    max_strength: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Replace each row by ``augment(img, L_y)`` with probability ``p_aug`` (Eq. cuda-aug/-lol).

    # ponytail: per-sample loop, group by (op, level) if step time matters
    """
    if p_aug <= 0:
        return images_u8
    mask = (
        torch.rand(images_u8.shape[0], generator=generator, device=images_u8.device)
        < p_aug
    )
    out = images_u8.clone()
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        level = int(levels_per_row[i].item())
        out[i] = augment(images_u8[i], level, max_strength, generator)
    return out


def _class_passes(
    rows: torch.Tensor,
    up_to: int,
    predict: Callable[[torch.Tensor], torch.Tensor],
    class_id: int,
    gamma: float,
    check_size: int,
    max_strength: int,
    generator: torch.Generator,
) -> bool:
    """True if every strength up to ``up_to`` passes the gamma-accuracy check (Eq. cuda-lol)."""
    for l in range(up_to + 1):
        draw_idx = torch.randint(
            rows.shape[0], (check_size,), generator=generator, device=rows.device
        )
        batch = torch.stack(
            [augment(rows[i], l, max_strength, generator) for i in draw_idx]
        )
        correct = int((predict(batch) == class_id).sum().item())
        if correct < gamma * check_size:
            return False
    return True


def update_levels(
    levels: torch.Tensor,
    class_rows: list[torch.Tensor],
    predict: Callable[[torch.Tensor], torch.Tensor],
    gamma: float,
    check_size: int,
    max_strength: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Update every class's level of learning from its own model check (Eq. cuda-lol).

    ``class_rows[c]`` holds class ``c``'s raw uint8 training images. The level rises only
    if every strength up to and including the current one passes the ``gamma``-accuracy
    check; otherwise it falls. Clipped to ``[0, max_strength]``.
    """
    new_levels = levels.clone()
    for c, rows in enumerate(class_rows):
        if rows.shape[0] == 0:
            continue
        l_c = int(levels[c].item())
        passed = _class_passes(
            rows, l_c, predict, c, gamma, check_size, max_strength, generator
        )
        new_levels[c] = min(max_strength, l_c + 1) if passed else max(0, l_c - 1)
    return new_levels
