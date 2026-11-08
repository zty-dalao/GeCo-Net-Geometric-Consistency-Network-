"""QUAVE/QWT to real four-channel image adapter.

Input:  grayscale image ``[H, W]``, ``[1, H, W]``, or ``[B, 1, H, W]``.
Output: real tensor ``[B, 4, H, W]`` in ``[real, i, j, k]`` order.

By default this module applies the repository's ``QWTForward`` and uses its
low-frequency quaternion map.  A trained/external QUAVE module can instead be
passed through ``quave=...``; its quaternion output is then converted by the
same adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

_QUAVE_SRC = Path(__file__).resolve().parent / "src" / "quave"
if str(_QUAVE_SRC) not in sys.path:
    sys.path.insert(0, str(_QUAVE_SRC))

try:
    from .src.quave.qwt import QWTForward  # package import
except ImportError:
    from qwt import QWTForward  # direct source-tree import

try:
    from .src.quave.quaternion_to_real import QuaternionToRealChannels
except ImportError:
    from quaternion_to_real import QuaternionToRealChannels


class QUAVEToRealChannels(nn.Module):
    """Run QWT/QUAVE and return a geometry-aligned real 4-channel tensor.

    Args:
        quave: Optional callable/module representing a trained QUAVE block.
            If omitted, the local ``QWTForward`` is used and its low-frequency
            quaternion map is selected.
        resize_mode: Interpolation used to restore the original ``H x W``.
    """

    def __init__(self, quave: Any = None, resize_mode: str = "bilinear", device: Any = None) -> None:
        super().__init__()
        if resize_mode not in {"bilinear", "bicubic", "nearest"}:
            raise ValueError(f"Unsupported resize_mode: {resize_mode!r}")
        self.quave = quave
        self.resize_mode = resize_mode
        self.to_real = QuaternionToRealChannels()
        self.qwt = None
        if self.quave is None:
            qwt_device = torch.device(device).type if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.qwt = QWTForward(qwt_device)

    @staticmethod
    def _select_quaternion(output: Any) -> Any:
        # QWTForward returns (low_quaternion, high_subbands).  A QUAVE block
        # may return the quaternion directly or a four-component tuple.
        if isinstance(output, tuple) and len(output) == 2:
            low, _high = output
            return low
        return output

    def forward(self, image: Tensor) -> Tensor:
        if not isinstance(image, Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
        elif image.ndim == 3 and image.shape[0] == 1:
            image = image.unsqueeze(0)               # [1,H,W] -> [1,1,H,W]
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"Expected grayscale image [H,W], [1,H,W], or [B,1,H,W], got {tuple(image.shape)}")
        if not image.is_floating_point():
            image = image.float()

        if self.quave is None:
            if self.qwt is None or self.qwt.device != image.device.type:
                # Rebuild only if the input is moved to a different device.
                self.qwt = QWTForward(image.device.type)
            quaternion = self._select_quaternion(self.qwt(image))
        else:
            quaternion = self._select_quaternion(self.quave(image))

        real_channels = self.to_real(quaternion)
        if real_channels.shape[-2:] != image.shape[-2:]:
            real_channels = F.interpolate(
                real_channels,
                size=image.shape[-2:],
                mode=self.resize_mode,
                align_corners=False if self.resize_mode != "nearest" else None,
            )
        return real_channels


def quave_to_real_channels(image: Tensor, quave: Any = None) -> Tensor:
    """Functional wrapper returning ``[B, 4, H, W]``."""
    return QUAVEToRealChannels(quave=quave)(image)


__all__ = ["QUAVEToRealChannels", "quave_to_real_channels"]
