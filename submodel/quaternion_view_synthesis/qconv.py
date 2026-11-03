"""Minimal quaternion operators: Hamilton product, QConv, QLinear, QNorm.

Feature layout is ``[B, 4 * C, H, W]`` with channel index ``4 * c + k``, i.e.
each of the ``C`` quaternion channels stores ``(a, b, c, d)`` contiguously.
Every weight is a quaternion, so a QConv with ``C_in``/``C_out`` quaternion
channels has ``C_out * C_in * k * k * 4`` parameters, which is exactly one
quarter of a real convolution with ``4 C_in`` / ``4 C_out`` channels.

Multiplication by a quaternion kernel is the 4x4 Hamilton matrix

    H(a, b, c, d) = [[ a, -b, -c, -d],
                     [ b,  a, -d,  c],
                     [ c,  d,  a, -b],
                     [ d, -c,  b,  a]]

so a QConv is implemented as *one* ``conv2d`` on a kernel assembled from that
matrix, keeping the parameter count at 1/4 and the arithmetic dense.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


def quaternion_normalize(q, eps=_EPS):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_multiply(q1, q2):
    """Hamilton product of two ``(..., 4)`` quaternion tensors."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def quaternion_inverse(q):
    q = quaternion_normalize(q)
    sign = torch.tensor((1.0, -1.0, -1.0, -1.0), device=q.device, dtype=q.dtype)
    return q * sign


def hamilton_matrix(q):
    """Return the ``(..., 4, 4)`` left-multiplication matrix of ``q``."""
    a, b, c, d = q.unbind(-1)
    rows = (
        torch.stack((a, -b, -c, -d), dim=-1),
        torch.stack((b, a, -d, c), dim=-1),
        torch.stack((c, d, a, -b), dim=-1),
        torch.stack((d, -c, b, a), dim=-1),
    )
    return torch.stack(rows, dim=-2)


def split_relu(x):
    return F.relu(x)


def rotate_feature_vector(h, q):
    """Apply ``h -> q (x) h (x) q^-1`` to a quaternion feature map.

    ``q`` is a per-sample unit quaternion ``[B, 4]`` and ``h`` is
    ``[B, C, 4, H, W]``.  The rotation only acts on the imaginary 3-vector part,
    which is the natural ``SO(3)`` action of a quaternion.  Because the layer is
    parameter free it satisfies exactly

        G(q2) o G(q1) = G(q2 (x) q1)

    so "x -> y1 then y1 -> y2" and "x -> y1 + y2" are algebraically identical.
    No real-valued conditioning (FiLM, concatenation) has this property.
    """
    q = quaternion_normalize(q)
    batch = q.shape[0]
    weight = q[:, 0].view(batch, 1, 1, 1, 1)
    axis = q[:, 1:].view(batch, 1, 3, 1, 1)

    real = h[:, :, 0:1]
    vector = h[:, :, 1:4]
    axis_sq = (axis * axis).sum(dim=2, keepdim=True)
    axis_dot = (axis * vector).sum(dim=2, keepdim=True)
    axis_cross = torch.cross(axis.expand_as(vector), vector, dim=2)
    rotated = (weight * weight - axis_sq) * vector + 2.0 * axis * axis_dot + 2.0 * weight * axis_cross
    return torch.cat((real, rotated), dim=2)


class QConv2d(nn.Module):
    """Quaternion convolution with Hamilton-product weight sharing."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 unit_weights=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.unit_weights = unit_weights

        shape = (out_channels, in_channels, kernel_size, kernel_size, 4)
        self.weight = nn.Parameter(torch.randn(shape) * 0.1)
        fan_in = in_channels * 4 * kernel_size * kernel_size
        self.scale = nn.Parameter(torch.full((out_channels, in_channels, 1, 1, 1),
                                             math.sqrt(2.0 / fan_in)))
        self.bias = nn.Parameter(torch.zeros(out_channels * 4))

    def _assembled_weight(self):
        weight = self.weight
        if self.unit_weights:
            weight = quaternion_normalize(weight, eps=1e-6)
        weight = weight * self.scale
        a, b, c, d = weight.unbind(-1)
        blocks = torch.stack((
            torch.stack((a, -b, -c, -d), dim=-1),
            torch.stack((b, a, -d, c), dim=-1),
            torch.stack((c, d, a, -b), dim=-1),
            torch.stack((d, -c, b, a), dim=-1),
        ), dim=-2)  # [Co, Ci, k, k, 4, 4]
        blocks = blocks.permute(0, 4, 1, 5, 2, 3)  # [Co, Kr, Ci, Kc, k, k]
        return blocks.reshape(
            self.out_channels * 4, self.in_channels * 4, self.kernel_size, self.kernel_size
        )

    def forward(self, x):
        kernel = self._assembled_weight()
        out = F.conv2d(x, kernel, bias=self.bias, stride=self.stride, padding=self.padding)
        return out


class QLinear(nn.Module):
    """Quaternion linear layer."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        shape = (out_channels, in_channels, 4)
        self.weight = nn.Parameter(torch.randn(shape) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels * 4)) if bias else None

    def forward(self, x):
        weight = quaternion_normalize(self.weight, eps=1e-6)
        a, b, c, d = weight.unbind(-1)
        blocks = torch.stack((
            torch.stack((a, -b, -c, -d), dim=-1),
            torch.stack((b, a, -d, c), dim=-1),
            torch.stack((c, d, a, -b), dim=-1),
            torch.stack((d, -c, b, a), dim=-1),
        ), dim=-2)  # [Co, Ci, 4, 4]
        blocks = blocks.permute(0, 2, 1, 3).reshape(self.out_channels * 4, self.in_channels * 4)
        return F.linear(x, blocks, self.bias)


class QNorm(nn.Module):
    """Group normalisation over each 4-component quaternion channel."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=channels, num_channels=channels * 4, eps=eps)

    def forward(self, x):
        return self.norm(x)


class QBlock(nn.Module):
    """Two QConvs with normalisation and split-ReLU, optional stride."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = QConv2d(in_channels, out_channels, 3, stride=stride)
        self.norm1 = QNorm(out_channels)
        self.conv2 = QConv2d(out_channels, out_channels, 3)
        self.norm2 = QNorm(out_channels)

    def forward(self, x):
        x = split_relu(self.norm1(self.conv1(x)))
        x = split_relu(self.norm2(self.conv2(x)))
        return x
