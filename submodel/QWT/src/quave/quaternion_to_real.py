"""Convert QUAVE quaternion feature maps to real-valued channels.

The QWT/QUAVE code represents a quaternion as four real components.  This
module converts that representation to a conventional PyTorch tensor with
channel order ``[real, i, j, k]`` and shape ``[B, 4, H, W]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _stack_components(components: Sequence[Tensor]) -> Tensor:
    if len(components) != 4:
        raise ValueError(
            f"A quaternion must have four components [real, i, j, k], got {len(components)}"
        )
    if not all(isinstance(component, Tensor) for component in components):
        raise TypeError("Quaternion components must all be torch.Tensor objects")

    reference = components[0]
    if any(component.shape != reference.shape for component in components[1:]):
        raise ValueError("All quaternion components must have the same shape")
    if any(component.device != reference.device for component in components[1:]):
        raise ValueError("All quaternion components must be on the same device")
    if any(component.dtype != reference.dtype for component in components[1:]):
        raise ValueError("All quaternion components must have the same dtype")

    # Components may be [B, H, W] or [B, 1, H, W].  The returned channel
    # dimension is always inserted at position 1.
    if reference.ndim == 3:
        return torch.stack(tuple(components), dim=1)
    if reference.ndim == 4 and reference.shape[1] == 1:
        return torch.cat(tuple(components), dim=1)
    raise ValueError(
        "Quaternion components must have shape [B, H, W] or [B, 1, H, W]"
    )


class QuaternionToRealChannels(nn.Module):
    """Convert a QUAVE quaternion image to a real 4-channel image.

    Accepted inputs:
        * ``[B, 4, H, W]``: already channel-first ``[real, i, j, k]``;
        * ``[B, H, W, 4]``: channel-last quaternion components;
        * ``(real, i, j, k)``: four tensors shaped ``[B,H,W]`` or
          ``[B,1,H,W]``.

    Returns:
        A tensor with shape ``[B, 4, H, W]`` and channel order
        ``[real, i, j, k]``.  No numerical normalization is performed.
    """

    def forward(self, quaternion_map: Tensor | Sequence[Tensor]) -> Tensor:
        if isinstance(quaternion_map, (tuple, list)):
            output = _stack_components(quaternion_map)
        elif isinstance(quaternion_map, Tensor):
            if quaternion_map.is_complex():
                raise TypeError(
                    "A native complex tensor has only real/imaginary parts; "
                    "QUAVE requires four real quaternion components [real, i, j, k]."
                )
            if quaternion_map.ndim == 4 and quaternion_map.shape[1] == 4:
                output = quaternion_map
            elif quaternion_map.ndim == 4 and quaternion_map.shape[-1] == 4:
                output = quaternion_map.permute(0, 3, 1, 2)
            elif quaternion_map.ndim == 5 and quaternion_map.shape[1] == 1 and quaternion_map.shape[2] == 4:
                output = quaternion_map[:, 0]
            elif quaternion_map.ndim == 5 and quaternion_map.shape[1] == 4 and quaternion_map.shape[2] == 1:
                output = quaternion_map[:, :, 0]
            else:
                raise ValueError(
                    "Expected quaternion map [B,4,H,W], [B,H,W,4], "
                    "[B,1,4,H,W], [B,4,1,H,W], or a four-component tuple"
                )
        else:
            raise TypeError("quaternion_map must be a Tensor or a four-component tuple/list")

        if output.ndim != 4 or output.shape[1] != 4:
            raise RuntimeError(f"Internal conversion produced invalid shape {tuple(output.shape)}")
        return output.contiguous()


def quaternion_to_real_channels(quaternion_map: Tensor | Sequence[Tensor]) -> Tensor:
    """Functional wrapper returning ``[B, 4, H, W]`` in ``[real,i,j,k]`` order."""
    return QuaternionToRealChannels()(quaternion_map)


__all__ = ["QuaternionToRealChannels", "quaternion_to_real_channels"]
