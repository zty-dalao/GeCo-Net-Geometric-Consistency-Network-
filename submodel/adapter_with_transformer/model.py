"""CNN-local and pooled-Transformer-global latent adapter."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from submodel.adapter import LatentAdapter


class PooledGlobalTransformer(nn.Module):
    """Extract global context after pooling a 3-D feature volume to few tokens."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        pool_size: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(in_channels, hidden_channels, pool_size, num_layers, num_heads) <= 0:
            raise ValueError("channel, pool, layer, and head counts must be positive")
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_channels = int(hidden_channels)
        self.pool_size = int(pool_size)
        self.input_projection = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(
            (self.pool_size, self.pool_size, self.pool_size)
        )
        token_count = self.pool_size ** 3
        self.position_embedding = nn.Parameter(
            torch.empty(1, token_count, hidden_channels)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        output_size = z.shape[-3:]
        feature = self.pool(self.input_projection(z))
        batch, channels, depth, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        position_embedding = self.position_embedding.to(dtype=tokens.dtype)
        tokens = self.transformer(tokens + position_embedding)
        feature = tokens.transpose(1, 2).reshape(
            batch, channels, depth, height, width
        )
        return F.interpolate(
            feature,
            size=output_size,
            mode="trilinear",
            align_corners=False,
        )


class TransformerLatentAdapter(nn.Module):
    """Fuse the existing CNN adapter feature with pooled global attention.

    The local branch is an actual ``LatentAdapter`` instance. Its ``encode``
    method produces the local 64-channel feature and its zero-initialized
    ``project`` layer maps the fused feature back to 256 channels. Therefore
    this adapter also starts as an exact identity mapping.
    """

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 64,
        pool_size: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_global_alpha: bool = False,
        global_alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.local_adapter = LatentAdapter(
            channels=channels,
            hidden_channels=hidden_channels,
        )
        self.global_branch = PooledGlobalTransformer(
            in_channels=channels,
            hidden_channels=hidden_channels,
            pool_size=pool_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.use_global_alpha = bool(use_global_alpha)
        if not math.isfinite(global_alpha_init):
            raise ValueError("global_alpha_init must be finite")
        if self.use_global_alpha:
            # Zero gates the entire global branch at initialization without
            # zeroing or otherwise changing its Transformer parameters.
            self.global_alpha = nn.Parameter(
                torch.tensor(float(global_alpha_init), dtype=torch.float32)
            )
        else:
            # None is omitted from state_dict, preserving the old key layout.
            self.register_parameter("global_alpha", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError(
                f"TransformerLatentAdapter expects [B,C,X,Y,Z], got {tuple(z.shape)}"
            )
        if z.shape[1] != self.channels:
            raise ValueError(
                f"TransformerLatentAdapter expects {self.channels} channels, "
                f"got {z.shape[1]}"
            )
        local_feature = self.local_adapter.encode(z)
        global_feature = self.global_branch(z)
        if self.global_alpha is not None:
            global_feature = (
                self.global_alpha.to(dtype=global_feature.dtype) * global_feature
            )
        residual = self.local_adapter.project(local_feature + global_feature)
        return z + residual
