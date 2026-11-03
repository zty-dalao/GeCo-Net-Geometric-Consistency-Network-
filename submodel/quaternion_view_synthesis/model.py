"""Quaternion-conditioned single-view projection synthesiser + ensemble fusion.

    b_i = F(a_i, q_{i->y})       single view, single forward
    b   = sum_i w_i b_i          quaternion softmax over the 20 measured views

The network contains **only** quaternion operators (QConv / QLinear / QNorm)
plus one parameter-free geometric path: the plane-induced homography from the
source view to the target view.  Grid sampling is a geometry operator, not a
learned layer, so the "QCNN + QMLP only" constraint is respected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quaternion_geometry.pose import pose_vec_to_quaternion
from . import geometry
from .qconv import (
    QBlock,
    QConv2d,
    QLinear,
    QNorm,
    hamilton_matrix,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    rotate_feature_vector,
    split_relu,
)


def _channelwise_qmul(gamma, h):
    """Per-channel quaternion left multiplication.

    ``gamma`` is ``[B, C, 4]`` and ``h`` is ``[B, C, 4, H, W]``.
    """
    matrix = hamilton_matrix(quaternion_normalize(gamma))
    return torch.einsum("bcij,bcjhw->bcihw", matrix, h)


class QuaternionViewSynthesizer(nn.Module):
    """``F(a, q_{s->t}) -> b`` plus a per-view confidence scalar."""

    def __init__(self, base_channels=16, levels=2):
        super().__init__()
        self.base_channels = base_channels
        self.levels = levels

        channels = [base_channels * (2 ** level) for level in range(levels + 1)]
        self.stem = QConv2d(1, channels[0], 3)
        self.stem_norm = QNorm(channels[0])

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.modulation = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(QBlock(channels[level], channels[level]))
            self.downs.append(QBlock(channels[level], channels[level + 1], stride=2))
            # QMLP: q_rel -> quaternion modulation (gamma, beta) for level + 1
            self.modulation.append(
                nn.Sequential(
                    QLinear(1, 8),
                    nn.ReLU(),
                    QLinear(8, channels[level + 1] * 2),
                )
            )

        self.bottleneck = QBlock(channels[-1], channels[-1])
        self.up_convs = nn.ModuleList()
        for level in reversed(range(levels)):
            self.up_convs.append(
                QConv2d(channels[level + 1] + channels[level], channels[level], 3)
            )

        self.head = QConv2d(channels[0], 1, 3)
        self.confidence = QLinear(channels[-1], 1)

    @staticmethod
    def _to_quaternion_layout(tensor):
        batch, channel4, height, width = tensor.shape
        return tensor.view(batch, channel4 // 4, 4, height, width)

    def _modulate(self, h, gamma_beta, q_rel):
        batch, channel4, height, width = h.shape
        channel = channel4 // 4
        gamma_beta = gamma_beta.view(batch, channel, 8)
        gamma = gamma_beta[:, :, :4]
        beta = gamma_beta[:, :, 4:]
        h = h.view(batch, channel, 4, height, width)
        h = _channelwise_qmul(gamma, h) + beta.unsqueeze(-1).unsqueeze(-1)
        h = h.reshape(batch, channel4, height, width)
        return rotate_feature_vector(self._to_quaternion_layout(h), q_rel).reshape(h.shape)

    def forward(self, image, vec_s, vec_t, isocenter=None):
        """``image`` [B,1,H,W]; ``vec_s``/``vec_t`` [B,12]."""
        height, width = image.shape[-2], image.shape[-1]

        homography = geometry.homography_source_to_target(
            vec_s, vec_t, height, width, isocenter
        )
        warped = geometry.warp_with_homography(image, homography)

        q_s = pose_vec_to_quaternion(vec_s)
        q_t = pose_vec_to_quaternion(vec_t)
        q_rel = quaternion_normalize(quaternion_multiply(quaternion_inverse(q_s), q_t))
        gate = geometry.residual_gate(q_rel).view(-1, 1, 1, 1)

        zeros = torch.zeros_like(image)
        quaternion_input = torch.cat((image, zeros, zeros, zeros), dim=1)  # [B,4,H,W]

        features = split_relu(self.stem_norm(self.stem(quaternion_input)))
        skips = []
        for level in range(self.levels):
            features = self.encoders[level](features)
            # parameter-free geometric action G(q): h -> q (x) h (x) q^-1
            features = rotate_feature_vector(
                self._to_quaternion_layout(features), q_rel
            ).reshape(features.shape)
            skips.append(features)

            features = self.downs[level](features)
            gamma_beta = self.modulation[level](q_rel.view(-1, 1, 4))
            features = self._modulate(features, gamma_beta, q_rel)

        features = self.bottleneck(features)

        pooled = self._to_quaternion_layout(features).mean(dim=(-1, -2))
        confidence = self.confidence(pooled.reshape(features.shape[0], -1)).norm(dim=-1)

        for index, up_conv in enumerate(self.up_convs):
            skip = skips[-1 - index]
            features = F.interpolate(features, size=skip.shape[-2:], mode="nearest")
            features = torch.cat((features, skip), dim=1)
            features = split_relu(up_conv(features))

        residual = self.head(features)
        residual = residual.view(residual.shape[0], 1, 4, *residual.shape[-2:]).norm(dim=2)
        out = warped + gate * residual
        return {
            "out": out,
            "warped": warped,
            "residual": residual,
            "gate": gate,
            "q_rel": q_rel,
            "confidence": confidence,
        }


class QuaternionSoftmaxEnsemble(nn.Module):
    """Fuse the 20 per-view predictions with a quaternion-derived softmax.

    ``logit_i = kappa * cos(Delta_i / 2) + lambda * confidence_i``

    ``cos(Delta/2)`` is exactly the real part of the quaternion inner product
    with the identity, i.e. the geometric "how close is this view to the
    target" term; ``confidence_i`` is produced by the QCNN itself.
    """

    def __init__(self, angle_scale=4.0, use_confidence=True):
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(float(angle_scale)))
        self.lam = nn.Parameter(torch.tensor(1.0))
        self.use_confidence = use_confidence

    def forward(self, predictions, q_rel, confidence=None, return_weights=False):
        """``predictions`` [V, B, 1, H, W]; ``q_rel`` [V, B, 4]."""
        cos_half = quaternion_normalize(q_rel)[..., 0]
        logits = self.kappa.abs() * cos_half
        if self.use_confidence and confidence is not None:
            logits = logits + self.lam * confidence
        weights = torch.softmax(logits, dim=0)
        fused = (weights[..., None, None, None] * predictions).sum(dim=0)
        if return_weights:
            return fused, weights
        return fused


def build_model(base_channels=16, levels=2):
    return QuaternionViewSynthesizer(base_channels=base_channels, levels=levels)
