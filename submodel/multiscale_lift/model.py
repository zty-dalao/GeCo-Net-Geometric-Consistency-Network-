"""Geometry-aware multi-scale 2D-to-3D lifting decoder.

The first implementation deliberately starts with the stable E2/E3/E4 path:
F2, F3 and F4 are lifted on progressively coarser 3D grids and fused while
the 3D decoder upsamples.  F0/F1 support is optional and uses a 2D fusion
stem before one high-resolution lift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.render import get_pixel00_center


class ScaleViewFusion(nn.Module):
    """Ada-style view fusion for one feature scale."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        self.channels = channels
        self.input_fc = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.GELU(),
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(channels, 1),
            nn.GELU(),
        )
        self.output_fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
        )

    def forward(self, latent, parent_logits=None, return_details=False):
        # latent: [views, channels, points]
        mean = latent.mean(dim=0, keepdim=True).expand_as(latent)
        var = latent.var(dim=0, unbiased=False, keepdim=True).expand_as(latent)
        feat = torch.cat([latent, mean, var], dim=1).transpose(1, 2)
        feat = self.input_fc(feat)
        delta_logits = self.weight_fc(feat).squeeze(-1)
        logits = delta_logits if parent_logits is None else delta_logits + parent_logits
        weights = torch.softmax(logits, dim=0)
        weights_expanded = weights.unsqueeze(-1)
        fused = torch.sum(feat * weights_expanded, dim=0)
        fused = self.output_fc(fused).transpose(0, 1)
        if not return_details:
            return fused

        view_count = max(2, latent.shape[0])
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=0)
        entropy = entropy / torch.log(
            latent.new_tensor(float(view_count))
        )
        variance = latent.var(dim=0, unbiased=False).mean(dim=0)
        variance = variance / (variance.mean().detach() + 1e-6)
        uncertainty = torch.stack([entropy, variance], dim=0)
        correction_loss = (
            delta_logits.abs().mean()
            if parent_logits is not None else delta_logits.new_zeros(())
        )
        return fused, logits, uncertainty, correction_loss


class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, size):
        x = self.act(self.conv1(x))
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=True)
        return self.act(self.conv2(x))


class FusionBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, mode="concat", use_uncertainty=False,
    ):
        super().__init__()
        self.mode = mode
        self.out_channels = int(out_channels)
        self.use_uncertainty = bool(use_uncertainty)
        if mode == "concat":
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.GELU(),
            )
        elif mode == "gated_add":
            # The input is [upsampled feature, projected observation].
            self.project = nn.Conv3d(in_channels - out_channels, out_channels, 1)
            self.gate = nn.Sequential(
                nn.Conv3d(
                    out_channels * 2 + (2 if self.use_uncertainty else 0),
                    out_channels,
                    1,
                ),
                nn.GELU(),
                nn.Conv3d(out_channels, out_channels, 1),
                nn.Sigmoid(),
            )
            self.refine = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.GELU(),
            )
        else:
            raise ValueError("multiscale fusion must be 'concat' or 'gated_add'")

    def forward(self, up, observation, uncertainty=None):
        if self.mode == "concat":
            return self.net(torch.cat([up, observation], dim=1))
        projected = self.project(observation)
        gate_inputs = [up, projected]
        if self.use_uncertainty:
            if uncertainty is None:
                raise ValueError("uncertainty-aware fusion requires uncertainty maps")
            gate_inputs.append(uncertainty)
        gate = self.gate(torch.cat(gate_inputs, dim=1))
        return self.refine(up + gate * projected)


class AuxiliaryHead(nn.Module):
    """Small prediction head used only by E5 training supervision."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(4, int(channels) // 2)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleLiftDecoder(nn.Module):
    """Lift F2/F3/F4 to 3D and decode with scale-matched observations.

    E2/E3/E4 are built from the same full-resolution physical coordinate grid
    at strides 2/4/8.  This keeps every scale registered to the same volume
    origin and physical extent instead of inferring geometry from tensor size.
    """

    def __init__(
        self,
        decoder_scale=4,
        fusion="concat",
        use_shallow_2d_fusion=False,
        shallow_channels=16,
        highres_fusion="gated_add",
        use_multiscale_supervision=False,
        use_hierarchical_view_weights=False,
        use_uncertainty_gate=False,
        use_query_checkpoint=True,
        query_chunk_size=25000,
    ):
        super().__init__()
        if int(decoder_scale) != 4:
            raise ValueError("MultiScaleLiftDecoder currently requires decoder scale=4")
        self.scale = int(decoder_scale)
        self.inplanes = 32
        self.fusion_mode = str(fusion)
        self.use_shallow_2d_fusion = bool(use_shallow_2d_fusion)
        self.shallow_channels = int(shallow_channels)
        self.highres_fusion = str(highres_fusion)
        self.use_multiscale_supervision = bool(use_multiscale_supervision)
        self.use_hierarchical_view_weights = bool(use_hierarchical_view_weights)
        self.use_uncertainty_gate = bool(use_uncertainty_gate)
        self.use_query_checkpoint = bool(use_query_checkpoint)
        self.query_chunk_size = int(query_chunk_size)
        self.scale_channels = {2: 32, 3: 64, 4: 128}
        self.view_fusion = nn.ModuleDict({
            str(index): ScaleViewFusion(channels)
            for index, channels in self.scale_channels.items()
        })
        self.up4 = ConvUpBlock(128, 64)
        self.fuse3 = FusionBlock(
            128, 64, fusion, use_uncertainty=self.use_uncertainty_gate,
        )
        self.up3 = ConvUpBlock(64, 32)
        self.fuse2 = FusionBlock(
            64, 32, fusion, use_uncertainty=self.use_uncertainty_gate,
        )
        self.up_full = ConvUpBlock(32, 16)
        self.output = nn.Conv3d(16, 1, 3, padding=1)
        if self.use_multiscale_supervision:
            self.aux_e4 = AuxiliaryHead(128)
            self.aux_e3 = AuxiliaryHead(64)
            self.aux_e2 = AuxiliaryHead(32)
        if self.use_shallow_2d_fusion:
            self.shallow_2d = nn.Sequential(
                nn.Conv2d(32, self.shallow_channels, 1),
                nn.GELU(),
                nn.Conv2d(self.shallow_channels, self.shallow_channels, 3, padding=1),
                nn.GELU(),
            )
            self.shallow_project = nn.Conv3d(self.shallow_channels, 16, 1)
            if self.highres_fusion == "gated_add":
                self.shallow_gate = nn.Sequential(
                    nn.Conv3d(
                        32 + (2 if self.use_uncertainty_gate else 0), 16, 1,
                    ),
                    nn.GELU(),
                    nn.Conv3d(16, 16, 1),
                    nn.Sigmoid(),
                )
                self.shallow_refine = nn.Sequential(
                    nn.Conv3d(16, 16, 3, padding=1),
                    nn.GELU(),
                )
                self.shallow_alpha = nn.Parameter(torch.zeros(()))
            elif self.highres_fusion == "concat":
                self.shallow_concat = nn.Sequential(
                    nn.Conv3d(32, 16, 3, padding=1),
                    nn.GELU(),
                    nn.Conv3d(16, 16, 3, padding=1),
                    nn.GELU(),
                )
            else:
                raise ValueError("highres fusion must be 'concat' or 'gated_add'")
            self.shallow_view_fusion = ScaleViewFusion(self.shallow_channels)

    @staticmethod
    def _project_uv(xyz, poses, image_shape):
        n = xyz.shape[0]
        xyz = torch.repeat_interleave(xyz.unsqueeze(0), poses.shape[0], dim=0)
        vecs = torch.repeat_interleave(poses.unsqueeze(1), n, dim=1)
        sources, detectors = vecs[..., :3], vecs[..., 3:6]
        uvectors, vvectors = vecs[..., 6:9], vecs[..., 9:]
        ray_dir = xyz - sources
        normals = torch.cross(uvectors, vvectors, dim=2)
        numerator = torch.sum(normals * (detectors - sources), dim=2, keepdim=True)
        denominator = torch.sum(normals * ray_dir, dim=2, keepdim=True)
        t = numerator / (denominator + 1e-6)
        uv_proj = sources + t * ray_dir
        h, w = image_shape[1], image_shape[0]
        pixel00 = get_pixel00_center(detectors, uvectors, vvectors, h, w)
        offset = uv_proj - pixel00
        u = torch.sum(offset * uvectors, dim=2) / torch.sum(uvectors * uvectors, dim=2)
        v = torch.sum(offset * vvectors, dim=2) / torch.sum(vvectors * vvectors, dim=2)
        u = 2 * (u / w) - 1
        v = 2 * (v / h) - 1
        return torch.cat([u.unsqueeze(-1), v.unsqueeze(-1)], dim=-1).unsqueeze(2)

    def _sample_and_fuse(
        self, chunk, feature, view_fusion, poses, image_shape,
        parent_logits=None, return_details=False,
    ):
        uv = self._project_uv(chunk, poses, image_shape)
        sampled = F.grid_sample(
            feature,
            uv,
            align_corners=True,
            mode="bilinear",
            padding_mode="zeros",
        )[:, :, :, 0]
        return view_fusion(
            sampled,
            parent_logits=parent_logits,
            return_details=return_details,
        )

    def _lift_feature(
        self, feature, view_fusion, poses, image_shape, xyz,
        parent_logits=None, return_details=False, collect_logits=True,
    ):
        points = xyz.reshape(-1, 3)
        fused_outputs = []
        logits_outputs = []
        uncertainty_outputs = []
        delta_losses = []
        point_chunks = torch.split(points, self.query_chunk_size)
        if parent_logits is None:
            parent_chunks = [None] * len(point_chunks)
        else:
            parent_chunks = torch.split(parent_logits, self.query_chunk_size, dim=1)
        for chunk, parent_chunk in zip(point_chunks, parent_chunks):
            if self.training and torch.is_grad_enabled() and self.use_query_checkpoint:
                if return_details and parent_chunk is not None:
                    result = checkpoint(
                        lambda p, fmap, prior: self._sample_and_fuse(
                            p, fmap, view_fusion, poses, image_shape,
                            parent_logits=prior, return_details=True,
                        ),
                        chunk, feature, parent_chunk, use_reentrant=False,
                    )
                else:
                    result = checkpoint(
                        lambda p, fmap, pc=parent_chunk: self._sample_and_fuse(
                            p, fmap, view_fusion, poses, image_shape,
                            parent_logits=pc,
                            return_details=return_details,
                        ),
                        chunk, feature, use_reentrant=False,
                    )
            else:
                result = self._sample_and_fuse(
                    chunk, feature, view_fusion, poses, image_shape,
                    parent_logits=parent_chunk,
                    return_details=return_details,
                )
            if return_details:
                fused, logits, uncertainty, delta_loss = result
                fused_outputs.append(fused)
                if collect_logits:
                    logits_outputs.append(logits)
                uncertainty_outputs.append(uncertainty)
                delta_losses.append(delta_loss)
            else:
                fused_outputs.append(result)
        fused = torch.cat(fused_outputs, dim=1)
        volume = fused.reshape(1, feature.shape[1], *xyz.shape[:3])
        if not return_details:
            return volume
        logits = torch.cat(logits_outputs, dim=1) if collect_logits else None
        uncertainty = torch.cat(uncertainty_outputs, dim=1).reshape(
            1, 2, *xyz.shape[:3]
        )
        delta_loss = torch.stack(delta_losses).mean()
        return volume, logits, uncertainty, delta_loss

    def _lift(
        self, feature_maps, poses, image_shape, xyz, index,
        parent_logits=None, return_details=False, collect_logits=True,
    ):
        return self._lift_feature(
            feature_maps[index], self.view_fusion[str(index)], poses, image_shape, xyz,
            parent_logits=parent_logits,
            return_details=return_details,
            collect_logits=collect_logits,
        )

    @staticmethod
    def _upsample_logits(logits, source_shape, target_shape):
        logits = logits.reshape(logits.shape[0], 1, *source_shape)
        logits = F.interpolate(
            logits, size=target_shape, mode="trilinear", align_corners=True,
        )
        return logits.reshape(logits.shape[0], -1)

    def forward(
        self, feature_maps, poses, image_shape, xyz_full, return_aux=False,
    ):
        if xyz_full is None:
            raise ValueError("MultiScaleLiftDecoder requires the full physical coordinate grid")
        e2_xyz = xyz_full[::2, ::2, ::2]
        e3_xyz = xyz_full[::4, ::4, ::4]
        e4_xyz = xyz_full[::8, ::8, ::8]
        need_details = (
            self.use_hierarchical_view_weights or self.use_uncertainty_gate
        )
        if need_details:
            e4, logits4, uncertainty4, delta4 = self._lift(
                feature_maps, poses, image_shape, e4_xyz, 4,
                return_details=True,
                collect_logits=self.use_hierarchical_view_weights,
            )
            parent3 = None
            if self.use_hierarchical_view_weights:
                parent3 = self._upsample_logits(
                    logits4, e4_xyz.shape[:3], e3_xyz.shape[:3],
                )
            e3, logits3, uncertainty3, delta3 = self._lift(
                feature_maps, poses, image_shape, e3_xyz, 3,
                parent_logits=parent3, return_details=True,
                collect_logits=self.use_hierarchical_view_weights,
            )
            parent2 = None
            if self.use_hierarchical_view_weights:
                parent2 = self._upsample_logits(
                    logits3, e3_xyz.shape[:3], e2_xyz.shape[:3],
                )
            e2, logits2, uncertainty2, delta2 = self._lift(
                feature_maps, poses, image_shape, e2_xyz, 2,
                parent_logits=parent2, return_details=True,
                collect_logits=False,
            )
        else:
            e2 = self._lift(feature_maps, poses, image_shape, e2_xyz, 2)
            e3 = self._lift(feature_maps, poses, image_shape, e3_xyz, 3)
            e4 = self._lift(feature_maps, poses, image_shape, e4_xyz, 4)
            uncertainty2 = uncertainty3 = uncertainty4 = None
            delta2 = delta3 = delta4 = e2.new_zeros(())

        x = self.up4(e4, e3.shape[-3:])
        x = self.fuse3(x, e3, uncertainty=uncertainty3)
        fused_e3 = x
        x = self.up3(x, e2.shape[-3:])
        x = self.fuse2(x, e2, uncertainty=uncertainty2)
        fused_e2 = x
        aux = {
            "e2": fused_e2,
            "view_delta_l1": (delta2 + delta3 + delta4) / 3.0,
        }
        if uncertainty2 is not None:
            aux.update({
                "uncertainty_e2_mean": uncertainty2.mean(),
                "uncertainty_e3_mean": uncertainty3.mean(),
                "uncertainty_e4_mean": uncertainty4.mean(),
            })
        if self.use_multiscale_supervision:
            aux.update({
                "pred_e4": self.aux_e4(e4),
                "pred_e3": self.aux_e3(fused_e3),
                "pred_e2": self.aux_e2(fused_e2),
            })
        x = self.up_full(x, xyz_full.shape[:3])

        if self.use_shallow_2d_fusion:
            shallow = self.shallow_2d(torch.cat([feature_maps[0], feature_maps[1]], dim=1))
            if xyz_full is None:
                raise ValueError("shallow 2D fusion requires xyz_full from the renderer")
            if self.use_uncertainty_gate and self.highres_fusion == "gated_add":
                e01, _, uncertainty01, shallow_delta = self._lift_feature(
                    shallow, self.shallow_view_fusion, poses, image_shape, xyz_full,
                    return_details=True, collect_logits=False,
                )
                aux["view_delta_l1"] = (
                    aux["view_delta_l1"] * 3.0 + shallow_delta
                ) / 4.0
                aux["uncertainty_e01_mean"] = uncertainty01.mean()
            else:
                e01 = self._lift_feature(
                    shallow, self.shallow_view_fusion, poses, image_shape, xyz_full,
                )
                uncertainty01 = None
            projected = self.shallow_project(e01)
            if self.highres_fusion == "concat":
                x = self.shallow_concat(torch.cat([x, projected], dim=1))
            else:
                gate_inputs = [x, projected]
                if self.use_uncertainty_gate:
                    gate_inputs.append(uncertainty01)
                gate = self.shallow_gate(torch.cat(gate_inputs, dim=1))
                correction = self.shallow_refine(gate * projected)
                x = x + self.shallow_alpha * correction
        output = self.output(x)
        if return_aux:
            return output, aux
        return output
