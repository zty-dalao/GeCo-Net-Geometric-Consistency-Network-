"""Angle/geometry-conditioned continuous prior completion for 3-D latents."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _DepthwiseResidualBlock(nn.Module):
    """Memory-efficient spatial context block at a fixed channel width."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(self.activation(self.depthwise(feature)))
        return feature + residual


class GeometrySetEncoder(nn.Module):
    """Encode an unordered set of cone-beam projection geometries.

    Each pose contains four 3-D vectors: source position, detector-center
    position, detector-u pixel step, and detector-v pixel step.  The derived
    representation is translation/scale normalized and also contains Fourier
    features of the source angle. Mean/max pooling makes the result invariant
    to the ordering of views and supports a variable number of views.
    """

    def __init__(self, hidden_channels: int = 32, output_channels: int = 64) -> None:
        super().__init__()
        if hidden_channels <= 0 or output_channels <= 0:
            raise ValueError("geometry channel counts must be positive")
        self.output_channels = int(output_channels)

        # 14 normalized geometric values + 8 angular Fourier values.
        input_channels = 22
        self.view_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, output_channels),
            nn.GELU(),
        )

    @staticmethod
    def _estimate_isocenter(
        sources: torch.Tensor,
        detectors: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the common center closest to all central-ray lines.

        Using ``sources.mean`` as a trajectory center is biased for limited or
        non-uniform angle subsets. The least-squares intersection of the
        source-to-detector-center lines remains meaningful for those scans and
        is translation equivariant. ``pinv`` also gives a finite result for a
        narrow-angle or otherwise rank-deficient set.
        """
        directions = F.normalize(detectors - sources, dim=-1, eps=1e-6)
        identity = torch.eye(
            3,
            device=sources.device,
            dtype=sources.dtype,
        ).view(1, 1, 3, 3)
        projectors = identity - directions.unsqueeze(-1) * directions.unsqueeze(-2)
        normal_matrix = projectors.sum(dim=1)
        rhs = torch.matmul(projectors, sources.unsqueeze(-1)).sum(dim=1)
        return torch.matmul(torch.linalg.pinv(normal_matrix), rhs).transpose(-1, -2)

    @staticmethod
    def _derived_geometry(poses: torch.Tensor) -> torch.Tensor:
        sources = poses[..., :3]
        detectors = poses[..., 3:6]
        uvectors = poses[..., 6:9]
        vvectors = poses[..., 9:12]

        trajectory_center = GeometrySetEncoder._estimate_isocenter(
            sources,
            detectors,
        )
        sid = torch.linalg.vector_norm(detectors - sources, dim=-1, keepdim=True)
        scale = sid.mean(dim=1, keepdim=True).clamp_min(1e-6)

        source_relative = (sources - trajectory_center) / scale
        detector_relative = (detectors - trajectory_center) / scale
        u_length = torch.linalg.vector_norm(uvectors, dim=-1, keepdim=True).clamp_min(1e-8)
        v_length = torch.linalg.vector_norm(vvectors, dim=-1, keepdim=True).clamp_min(1e-8)
        u_direction = uvectors / u_length
        v_direction = vvectors / v_length

        source_angle = torch.atan2(source_relative[..., 1], source_relative[..., 0])
        angular_features = []
        for harmonic in range(1, 5):
            angular_features.extend((
                torch.sin(harmonic * source_angle).unsqueeze(-1),
                torch.cos(harmonic * source_angle).unsqueeze(-1),
            ))

        return torch.cat(
            (
                source_relative,
                detector_relative,
                u_direction,
                v_direction,
                u_length / scale,
                v_length / scale,
                *angular_features,
            ),
            dim=-1,
        )

    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        if poses.ndim != 3 or poses.shape[-1] != 12:
            raise ValueError(
                "GeometrySetEncoder expects [V,12] or [B,V,12], "
                f"got {tuple(poses.shape)}"
            )
        if poses.shape[1] == 0:
            raise ValueError("at least one projection view is required")

        view_features = self.view_mlp(self._derived_geometry(poses.float()))
        pooled = torch.cat(
            (view_features.mean(dim=1), view_features.amax(dim=1)),
            dim=-1,
        )
        return self.output_mlp(pooled)


class ContinuousPriorCompletion(nn.Module):
    """Predict a continuous residual from an aligned latent and view geometry.

    The final projection is zero initialized, so enabling this module does not
    perturb an existing model before completion training starts.
    """

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 16,
        geometry_hidden_channels: int = 32,
        geometry_channels: int = 64,
        residual_scale: float = 1.0,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if min(channels, hidden_channels, geometry_hidden_channels, geometry_channels) <= 0:
            raise ValueError("completion and geometry channel counts must be positive")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")

        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_scale = float(residual_scale)
        self.use_checkpoint = bool(use_checkpoint)
        self.geometry_encoder = GeometrySetEncoder(
            hidden_channels=geometry_hidden_channels,
            output_channels=geometry_channels,
        )
        self.reduce = nn.Conv3d(channels, hidden_channels, kernel_size=1)
        self.geometry_film = nn.Sequential(
            nn.Linear(geometry_channels, hidden_channels * 2),
        )
        self.context = nn.ModuleList(
            _DepthwiseResidualBlock(hidden_channels, dilation)
            for dilation in (1, 2, 4)
        )
        self.expand = nn.Conv3d(hidden_channels, channels, kernel_size=1)

        # Geometry starts as neutral FiLM and the complete branch starts as an
        # exact zero residual. Internal convolutions retain normal randomized
        # initialization so gradients can enter after the first update.
        nn.init.zeros_(self.geometry_film[-1].weight)
        nn.init.zeros_(self.geometry_film[-1].bias)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def predict_residual(self, latent: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != self.channels:
            raise ValueError(
                f"ContinuousPriorCompletion expects [B,{self.channels},D,H,W], "
                f"got {tuple(latent.shape)}"
            )
        geometry = self.geometry_encoder(poses).to(
            device=latent.device,
            dtype=latent.dtype,
        )
        if geometry.shape[0] == 1 and latent.shape[0] != 1:
            geometry = geometry.expand(latent.shape[0], -1)
        if geometry.shape[0] != latent.shape[0]:
            raise ValueError(
                "geometry batch size must be 1 or match the latent batch size"
            )

        gamma, beta = self.geometry_film(geometry).chunk(2, dim=1)
        gamma = gamma[..., None, None, None]
        beta = beta[..., None, None, None]
        feature = self.reduce(latent)
        feature = feature * (1.0 + gamma) + beta
        for block in self.context:
            if self.training and torch.is_grad_enabled() and self.use_checkpoint:
                feature = checkpoint(block, feature, use_reentrant=False)
            else:
                feature = block(feature)
        raw_residual = self.expand(feature)
        return self.residual_scale * torch.tanh(raw_residual)

    def forward(
        self,
        latent: torch.Tensor,
        poses: torch.Tensor,
        return_residual: bool = False,
    ):
        residual = self.predict_residual(latent, poses)
        completed = latent + residual
        if return_residual:
            return completed, residual
        return completed
