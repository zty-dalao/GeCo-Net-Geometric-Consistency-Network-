import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SRGAN import generator


class PriorFeatureStem(nn.Sequential):
    """Map a fixed low-resolution pCT volume to the decoder latent space."""

    def __init__(self, inplanes: int = 256) -> None:
        super().__init__(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(64, int(inplanes), kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )


class DecoderPretrainer(nn.Module):
    """Pretrain the project's decoder from irreversibly degraded pCT volumes.

    Input:  [B, 1, X, Y, Z]
    Latent: [B, 256, X/4, Y/4, Z/4]
    Output: [B, 1, X, Y, Z]

    Average pooling is deliberately fixed. A learned full-resolution encoder
    could hide the complete input in the 256 latent channels and reduce the
    task to an identity mapping.
    """

    def __init__(self, decoder_conf) -> None:
        super().__init__()
        self.scale = int(decoder_conf.scale)
        self.inplanes = int(decoder_conf.inplanes)
        if self.scale != 4:
            raise ValueError(
                f"Decoder pretraining expects scale=4 to match the project, got {self.scale}."
            )
        if self.inplanes != 256:
            raise ValueError(
                f"Decoder pretraining expects 256 latent channels, got {self.inplanes}."
            )

        self.feature_stem = PriorFeatureStem(self.inplanes)
        # Reuse the exact decoder class used by models/model.py.
        self.decoder = generator(decoder_conf)
        self.output_activation = nn.GELU()

    def degrade(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply fixed low-pass filtering and 4x spatial downsampling."""
        return F.avg_pool3d(volume, kernel_size=self.scale, stride=self.scale)

    def forward(
        self, volume: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_resolution = self.degrade(volume)
        latent = self.feature_stem(low_resolution)
        prediction = self.output_activation(self.decoder(latent))
        return prediction, low_resolution, latent
