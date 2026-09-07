"""Learned high-frequency pCT prior encoder with the original SRGAN decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SRGAN import generator


class LearnedPriorEncoder(nn.Module):
    """Encode fixed low-frequency means and learned high-frequency residuals.

    The first low-resolution channel is always the fixed 4x4x4 block mean.
    The remaining 31 channels are learned measurements of the block-wise
    high-frequency residual ``volume - nearest_upsample(block_mean)``.
    """

    def __init__(
        self,
        inplanes: int = 256,
        scale: int = 4,
        detail_channels: int = 31,
    ) -> None:
        super().__init__()
        if int(scale) != 4:
            raise ValueError(f"LearnedPriorEncoder expects scale=4, got {scale}.")
        if int(inplanes) != 256:
            raise ValueError(
                f"LearnedPriorEncoder expects 256 output channels, got {inplanes}."
            )
        if int(detail_channels) != 31:
            raise ValueError(
                "This architecture expects 31 detail channels so mean+detail=32."
            )

        self.scale = int(scale)
        self.inplanes = int(inplanes)
        self.detail_channels = int(detail_channels)
        self.detail_downsample = nn.Conv3d(
            1,
            self.detail_channels,
            kernel_size=self.scale,
            stride=self.scale,
            padding=0,
        )
        self.feature_stem = nn.Sequential(
            nn.Conv3d(1 + self.detail_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(64, self.inplanes, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def make_low_feature(
        self, volume: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return concatenated low feature, mean channel, and detail channels."""
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise ValueError(
                "LearnedPriorEncoder expects [B,1,X,Y,Z], "
                f"got {tuple(volume.shape)}."
            )
        mean = F.avg_pool3d(
            volume,
            kernel_size=self.scale,
            stride=self.scale,
        )
        mean_up = F.interpolate(
            mean,
            size=volume.shape[-3:],
            mode="nearest",
        )
        high_frequency = volume - mean_up
        detail = self.detail_downsample(high_frequency)
        if mean.shape[-3:] != detail.shape[-3:]:
            raise RuntimeError(
                "Mean/detail shape mismatch. Input spatial dimensions must be "
                f"compatible with scale={self.scale}: {tuple(mean.shape)} vs "
                f"{tuple(detail.shape)}."
            )
        low_feature = torch.cat([mean, detail], dim=1)
        return low_feature, mean, detail

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        low_feature, _, _ = self.make_low_feature(volume)
        return self.feature_stem(low_feature)


# Backward-compatible import name used by the first deep_encoder draft.
DeepPriorFeatureStem = LearnedPriorEncoder


class DecoderPretrainer(nn.Module):
    """Learned mean/detail prior encoder followed by the original decoder."""

    def __init__(self, decoder_conf) -> None:
        super().__init__()
        self.scale = int(decoder_conf.scale)
        self.inplanes = int(decoder_conf.inplanes)
        if self.scale != 4:
            raise ValueError(
                f"Deep decoder pretraining expects scale=4, got {self.scale}."
            )
        if self.inplanes != 256:
            raise ValueError(
                f"Deep decoder pretraining expects 256 latent channels, got {self.inplanes}."
            )

        # Keep this attribute name because the shared checkpoint writer saves
        # model.feature_stem. It now contains the complete prior encoder,
        # including the learned detail-downsampling convolution.
        self.feature_stem = LearnedPriorEncoder(
            inplanes=self.inplanes,
            scale=self.scale,
        )
        self.decoder = generator(decoder_conf)
        self.output_activation = nn.GELU()

    def degrade(self, volume: torch.Tensor) -> torch.Tensor:
        """Return the 32-channel mean/detail low-resolution representation."""
        low_feature, _, _ = self.feature_stem.make_low_feature(volume)
        return low_feature

    def forward(
        self, volume: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_feature = self.degrade(volume)
        latent = self.feature_stem.feature_stem(low_feature)
        prediction = self.output_activation(self.decoder(latent))
        return prediction, low_feature, latent
