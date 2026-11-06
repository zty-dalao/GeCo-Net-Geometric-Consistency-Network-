"""GPU wavelet preprocessing for projection images."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward
except ImportError as exc:  # pragma: no cover
    DWTForward = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class WaveletProcess(nn.Module):
    """Configurable GPU DWT with all subbands resized to input H x W.

    Input is ``[N, 1, H, W]``. Output is ``[N, 1 + 3 * level, H, W]``;
    with the default one-level transform this is ``[LL, LH, HL, HH]``.
    """

    def __init__(self, wavelet="db4", level=1, mode="symmetric", resize_mode="bilinear"):
        super().__init__()
        if DWTForward is None:
            raise ImportError(
                "WaveletProcess requires pytorch-wavelets and its PyWavelets "
                "dependency. Install them with `python -m pip install "
                "pytorch-wavelets PyWavelets`."
            ) from _IMPORT_ERROR
        if not isinstance(level, int) or level < 1:
            raise ValueError(f"level must be a positive integer, got {level!r}")
        if resize_mode not in {"bilinear", "bicubic", "nearest"}:
            raise ValueError(f"Unsupported resize_mode: {resize_mode!r}")

        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.resize_mode = resize_mode
        self.out_channels = 1 + 3 * level
        self.dwt = DWTForward(J=level, wave=wavelet, mode=mode)

    def forward(self, image: Tensor) -> Tensor:
        if not isinstance(image, Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"Expected [N, 1, H, W], got {tuple(image.shape)}")
        if not image.is_floating_point():
            image = image.float()

        target_size = image.shape[-2:]
        yl, yh = self.dwt(image)
        bands = [yl]
        for high in yh:  # [N, 1, 3, h, w] -> [N, 3, h, w]
            bands.append(high.flatten(start_dim=1, end_dim=2))

        bands = [
            F.interpolate(
                band,
                size=target_size,
                mode=self.resize_mode,
                align_corners=False if self.resize_mode != "nearest" else None,
            )
            for band in bands
        ]
        return torch.cat(bands, dim=1)


@lru_cache(maxsize=16)
def _cached_wavelet(wavelet, level, mode):
    return WaveletProcess(wavelet=wavelet, level=level, mode=mode)


def real_value_wavelet_transformer(image: Tensor, wavelet="db4", level=1, mode="symmetric") -> Tensor:
    """Backward-compatible API used by ``trainer.py``."""
    transform = _cached_wavelet(wavelet, level, mode).to(device=image.device)
    return transform(image)


__all__ = ["WaveletProcess", "real_value_wavelet_transformer"]
