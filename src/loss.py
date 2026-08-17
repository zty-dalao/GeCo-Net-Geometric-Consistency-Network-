"""Loss functions for the projection interpolation model (Fig. 3).

Contains:

* ``ssim`` / ``ssim_loss`` - differentiable SSIM and ``1 - SSIM``.
* ``VGGPerceptualLoss``   - VGG19 perceptual loss (MSE of features).
* ``InterpolationLoss``   - ``1 - SSIM + alpha * L_VGG`` (paper Eq. 1).
* ``consistency_loss``    - view-selection / rotation consistency.

Image tensors are assumed to be in the raw projection range (e.g. 0..255);
pass ``val_range`` (and ``input_range`` for the VGG loss) accordingly.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------
def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    g = torch.tensor(
        [math.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
         for x in range(window_size)],
        dtype=torch.float32,
    )
    return g / g.sum()


def _create_window(window_size: int, channels: int) -> torch.Tensor:
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
    return _2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11,
         val_range: float = 1.0) -> torch.Tensor:
    """Mean structural similarity in [0, 1] (differentiable, higher=better)."""
    channels = x.size(1)
    window = _create_window(window_size, channels).to(x.device).to(x.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(x, window, padding=pad, groups=channels)
    mu2 = F.conv2d(y, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11,
              val_range: float = 1.0) -> torch.Tensor:
    """``1 - SSIM`` (lower is better)."""
    return 1.0 - ssim(x, y, window_size=window_size, val_range=val_range)


# ---------------------------------------------------------------------------
# VGG19 perceptual loss
# ---------------------------------------------------------------------------
class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG19 features.

    ``L_VGG(x, y) = MSE(rho(x), rho(y))``, where ``rho`` is the first
    ``feature_layers`` layers of VGG19 (frozen).  Single-channel inputs are
    tiled to 3 channels and normalized to [0, 1] before the (optional)
    ImageNet mean/std normalization.
    """

    def __init__(self, feature_layers: int = 9, input_range: float = 255.0,
                 imagenet_norm: bool = True, weights: Optional[str] = "IMAGENET1K_V1") -> None:
        super().__init__()
        from torchvision import models

        if weights == "IMAGENET1K_V1":
            weights = models.VGG19_Weights.IMAGENET1K_V1
        else:
            weights = None  # random init (no download)
        vgg = models.vgg19(weights=weights)
        self.features = vgg.features[:feature_layers].eval()
        for p in self.features.parameters():
            p.requires_grad = False

        self.input_range = float(input_range)
        self.imagenet_norm = imagenet_norm
        if imagenet_norm:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.input_range                    # -> [0, 1]
        if x.size(1) == 1:                          # single-channel -> 3-channel
            x = x.repeat(1, 3, 1, 1)
        if self.imagenet_norm:
            x = (x - self.mean) / self.std
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.features(self._prepare(x)),
                          self.features(self._prepare(y)))


# ---------------------------------------------------------------------------
# Interpolation loss (paper Eq. 1)
# ---------------------------------------------------------------------------
class InterpolationLoss(nn.Module):
    """``1 - SSIM + alpha * L_VGG``."""

    def __init__(self, alpha: float = 1e-2, val_range: float = 1.0,
                 window_size: int = 11, vgg: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.val_range = val_range
        self.window_size = window_size
        self.vgg = vgg

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(pred, target)[-1]

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """Return ``(l_ssim, l_vgg, total)`` for logging."""
        l_ssim = ssim_loss(pred, target, self.window_size, self.val_range)
        l_vgg = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if self.vgg is not None:
            l_vgg = self.vgg(pred, target)
        return l_ssim, l_vgg, l_ssim + self.alpha * l_vgg


# ---------------------------------------------------------------------------
# Consistency loss (self-supervised)
# ---------------------------------------------------------------------------
def consistency_loss(pred_a: torch.Tensor, pred_b: torch.Tensor, use_ssim: bool = True,
                     val_range: float = 1.0, window_size: int = 11) -> torch.Tensor:
    """View-selection / rotation consistency between two interpolations.

    Must be combined with a supervised loss; used alone it can be satisfied by
    a degenerate constant-output model.
    """
    if use_ssim:
        return ssim_loss(pred_a, pred_b, window_size=window_size, val_range=val_range)
    return F.mse_loss(pred_a, pred_b)
