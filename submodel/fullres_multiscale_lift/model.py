"""Full-resolution multi-scale 2D-to-3D lifting decoder.

Every encoder level is sampled on the same full physical 3-D grid. The
per-view features are concatenated in the original encoder order and fused by
the main model's existing Aggregator. The fused 256-channel volume is then
split back into F0..F4 channel groups, processed by independent concat-residual
blocks, concatenated coarse-to-fine, and reconstructed directly with one final
1x1x1 convolution. H01 deliberately has no additional residual block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.render import get_pixel00_center


class ConcatResidualBlock3D(nn.Module):
    """Preserve the input by concatenation, then reduce 2C back to C."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("ConcatResidualBlock3D channels must be positive")
        self.channels = channels
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.reduce = nn.Conv3d(channels * 2, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        branch = self.act(self.conv1(x))
        branch = self.act(self.conv2(branch))
        return self.act(self.reduce(torch.cat([x, branch], dim=1)))


class FullResolutionMultiScaleLiftDecoder(nn.Module):
    """Lift F0..F4 to one full grid and decode without spatial upsampling."""

    feature_channels = (16, 16, 32, 64, 128)

    def __init__(
        self,
        decoder_scale=4,
        query_chunk_size=4000,
        use_query_checkpoint=True,
        use_block_checkpoint=True,
        metric_stride=8,
    ):
        super().__init__()
        # The renderer still reads decoder.scale to construct xyz_sample. This
        # decoder itself uses xyz_full and performs no spatial upsampling.
        self.scale = int(decoder_scale)
        self.inplanes = sum(self.feature_channels)
        self.query_chunk_size = int(query_chunk_size)
        self.use_query_checkpoint = bool(use_query_checkpoint)
        self.use_block_checkpoint = bool(use_block_checkpoint)
        self.metric_stride = max(1, int(metric_stride))
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")

        self.scale_blocks = nn.ModuleList(
            ConcatResidualBlock3D(channels)
            for channels in self.feature_channels
        )
        # Deliberately the only 1x1x1 convolution in this decoder.
        self.output = nn.Conv3d(self.inplanes, 1, 1)

    @staticmethod
    def _project_uv(xyz, poses, image_shape):
        point_count = xyz.shape[0]
        xyz = torch.repeat_interleave(xyz.unsqueeze(0), poses.shape[0], dim=0)
        vecs = torch.repeat_interleave(poses.unsqueeze(1), point_count, dim=1)
        sources, detectors = vecs[..., :3], vecs[..., 3:6]
        uvectors, vvectors = vecs[..., 6:9], vecs[..., 9:]
        ray_dir = xyz - sources
        normals = torch.cross(uvectors, vvectors, dim=2)
        numerator = torch.sum(normals * (detectors - sources), dim=2, keepdim=True)
        denominator = torch.sum(normals * ray_dir, dim=2, keepdim=True)
        projected = sources + numerator / (denominator + 1e-6) * ray_dir

        height, width = image_shape[1], image_shape[0]
        pixel00 = get_pixel00_center(detectors, uvectors, vvectors, height, width)
        offset = projected - pixel00
        u = torch.sum(offset * uvectors, dim=2) / torch.sum(
            uvectors * uvectors, dim=2
        )
        v = torch.sum(offset * vvectors, dim=2) / torch.sum(
            vvectors * vvectors, dim=2
        )
        u = 2 * (u / width) - 1
        v = 2 * (v / height) - 1
        return torch.cat([u.unsqueeze(-1), v.unsqueeze(-1)], dim=-1).unsqueeze(2)

    def _sample_and_fuse(
        self, points, feature_maps, poses, image_shape, view_fuser,
    ):
        uv = self._project_uv(points, poses, image_shape)
        sampled_scales = []
        for feature in feature_maps:
            sampled = F.grid_sample(
                feature,
                uv,
                align_corners=True,
                mode="bilinear",
                padding_mode="zeros",
            )[:, :, :, 0]
            sampled_scales.append(sampled)
        # Reproduce ResEncoder.queryfeature's original 256-channel ordering so
        # the existing Aggregator can be reused without a second fusion model.
        sampled = torch.cat(sampled_scales, dim=1)
        return view_fuser(sampled)

    def _lift_full_volume(
        self, feature_maps, poses, image_shape, xyz_full, view_fuser,
    ):
        if len(feature_maps) != len(self.feature_channels):
            raise ValueError(
                "Full-resolution lift expects five encoder maps F0..F4, got "
                f"{len(feature_maps)}"
            )
        actual_channels = tuple(int(feature.shape[1]) for feature in feature_maps)
        if actual_channels != self.feature_channels:
            raise ValueError(
                "Unexpected F0..F4 channels: expected "
                f"{self.feature_channels}, got {actual_channels}"
            )

        points = xyz_full.contiguous().reshape(-1, 3)
        fused_chunks = []
        for point_chunk in torch.split(points, self.query_chunk_size):
            if self.training and torch.is_grad_enabled() and self.use_query_checkpoint:
                fused = checkpoint(
                    lambda p, *maps: self._sample_and_fuse(
                        p, maps, poses, image_shape, view_fuser,
                    ),
                    point_chunk,
                    *feature_maps,
                    use_reentrant=False,
                )
            else:
                fused = self._sample_and_fuse(
                    point_chunk, feature_maps, poses, image_shape, view_fuser,
                )
            fused_chunks.append(fused)

        fused = torch.cat(fused_chunks, dim=1)
        if fused.shape[0] != self.inplanes:
            raise ValueError(
                f"Aggregator must return {self.inplanes} channels, got {fused.shape[0]}"
            )
        return fused.reshape(1, self.inplanes, *xyz_full.shape[:3])

    def _run_block(self, block, volume):
        if (
            self.training
            and torch.is_grad_enabled()
            and self.use_block_checkpoint
        ):
            return checkpoint(block, volume, use_reentrant=False)
        return block(volume)

    def _sample_for_metrics(self, volume):
        stride = self.metric_stride
        return volume.detach()[..., ::stride, ::stride, ::stride]

    def _record_branch_metrics(self, aux, index, source, result):
        source_sample = self._sample_for_metrics(source)
        result_sample = self._sample_for_metrics(result)
        aux[f"fullres_x{index}_abs_mean"] = source_sample.abs().mean()
        aux[f"fullres_r{index}_abs_mean"] = result_sample.abs().mean()
        aux[f"fullres_r{index}_change_l1"] = (
            result_sample - source_sample
        ).abs().mean()

    def forward(
        self,
        feature_maps,
        poses,
        image_shape,
        xyz_full,
        view_fuser,
        return_aux=False,
    ):
        if xyz_full is None:
            raise ValueError(
                "FullResolutionMultiScaleLiftDecoder requires xyz_full"
            )
        full_volume = self._lift_full_volume(
            feature_maps, poses, image_shape, xyz_full, view_fuser,
        )
        scale_volumes = torch.split(full_volume, self.feature_channels, dim=1)

        aux = {}
        residuals = []
        for index, (volume, block) in enumerate(
            zip(scale_volumes, self.scale_blocks)
        ):
            result = self._run_block(block, volume)
            residuals.append(result)
            self._record_branch_metrics(aux, index, volume, result)

        # Coarse-to-fine accumulation: R4 + R3 + R2 + R1 + R0.
        h4 = residuals[4]
        h3 = torch.cat([h4, residuals[3]], dim=1)
        h2 = torch.cat([h3, residuals[2]], dim=1)
        h01 = torch.cat([h2, residuals[1], residuals[0]], dim=1)
        if h01.shape[1] != self.inplanes:
            raise RuntimeError(
                f"Final concatenation must have {self.inplanes} channels, "
                f"got {h01.shape[1]}"
            )

        # H01 already contains all five independently refined scales. Mapping
        # it directly avoids another full-resolution 256-channel residual
        # block and its dominant activation peak.
        output = self.output(h01)

        h01_sample = self._sample_for_metrics(h01)
        aux.update({
            "fullres_h3_abs_mean": self._sample_for_metrics(h3).abs().mean(),
            "fullres_h2_abs_mean": self._sample_for_metrics(h2).abs().mean(),
            "fullres_h01_abs_mean": h01_sample.abs().mean(),
            "fullres_output_abs_mean": self._sample_for_metrics(output).abs().mean(),
            "latent": h01,
        })
        if return_aux:
            return output, aux
        return output


__all__ = ["ConcatResidualBlock3D", "FullResolutionMultiScaleLiftDecoder"]
