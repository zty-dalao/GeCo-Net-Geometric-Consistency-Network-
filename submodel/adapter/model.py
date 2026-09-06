"""Adapter between the projection aggregator and the pretrained 3-D decoder."""

import torch
import torch.nn as nn


class LatentAdapter(nn.Module):
    """A lightweight, identity-initialized residual adapter for 3-D latents.

    The final convolution is zero initialized, so inserting this module does
    not perturb an existing projection latent before the adapter is trained.
    """

    def __init__(self, channels: int = 256, hidden_channels: int = 64):
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("channels and hidden_channels must be positive")

        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv3d(self.channels, self.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, self.channels, kernel_size=1),
        )

        # Start as an exact identity mapping: forward(z) == z.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError(
                f"LatentAdapter expects [B,C,X,Y,Z], got shape {tuple(z.shape)}"
            )
        if z.shape[1] != self.channels:
            raise ValueError(
                f"LatentAdapter expects {self.channels} channels, got {z.shape[1]}"
            )
        return z + self.net(z)
