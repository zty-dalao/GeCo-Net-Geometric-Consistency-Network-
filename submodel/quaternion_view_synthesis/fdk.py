"""Weighted Feldkamp-Davis-Kress (FDK) reconstruction for the fan/cone geometry.

Views are not equivalent: the 20 measured projections are real measurements,
while the synthesised views are functions of those 20.  Giving every view the
same weight would treat interpolated data as independent measurements, so each
view carries an explicit weight ``m_v`` (measured: 1.0, synthesised: rho < 1).

The absolute grey level of an FDK reconstruction depends on the ramp-filter and
``d beta`` discretisation, so the evaluation fits one global scale per volume
(least squares against the ground truth).  This is reported explicitly.
"""

import math

import torch
import torch.nn.functional as F

from .geometry import pixel_spacing, pose_to_frame


def cone_weight(proj, vecs):
    """Cosine (a.k.a. cone-beam) pre-weight ``D / sqrt(D^2 + u^2 + v^2)``."""
    views, height, width = proj.shape
    du, dv = pixel_spacing(vecs)
    cols = torch.arange(width, device=proj.device, dtype=proj.dtype) - (width - 1) * 0.5
    rows = torch.arange(height, device=proj.device, dtype=proj.dtype) - (height - 1) * 0.5
    u = cols.view(1, 1, width) * du.view(-1, 1, 1)
    v = rows.view(1, height, 1) * dv.view(-1, 1, 1)
    source, detector, _ = pose_to_frame(vecs)
    sid = (detector - source).norm(dim=-1).view(-1, 1, 1)
    return proj * sid / (sid * sid + u * u + v * v).sqrt()


def _ramp_kernel(width, du, device, dtype):
    size = 2 * width
    kernel = torch.zeros(size, device=device, dtype=dtype)
    index = torch.arange(1, width, 2, device=device, dtype=dtype)
    value = -1.0 / (math.pi ** 2 * index ** 2 * du * du)
    kernel[width + index.long()] = value
    kernel[width - index.long()] = value
    kernel[width] = 1.0 / (4.0 * du * du)
    return kernel


def ramp_filter(proj, vecs):
    """Filter every view along the detector ``u`` axis (the rotation direction)."""
    du, _ = pixel_spacing(vecs)
    width = proj.shape[-1]
    size = 2 * width
    filtered = []
    for view in range(proj.shape[0]):
        kernel = _ramp_kernel(width, du[view], proj.device, proj.dtype)
        spectrum = torch.fft.rfft(proj[view], n=size, dim=-1) * torch.fft.rfft(kernel)
        # kernel is centred at index ``width``, so the linear convolution
        # result occupies the second half of the circular buffer.
        filtered.append(torch.fft.irfft(spectrum, n=size, dim=-1)[..., width:] * du[view])
    return torch.stack(filtered, dim=0)


def _voxel_grid(volume_origin, volume_phy, resolution, device, dtype):
    axes = []
    for axis in range(3):
        origin = float(volume_origin[axis])
        size = float(volume_phy[axis])
        count = int(resolution[axis])
        axes.append(origin + (torch.arange(count, device=device, dtype=dtype) + 0.5) * (size / count))
    grid = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return torch.stack(grid, dim=-1).reshape(-1, 3)


def backproject(filtered, vecs, volume_origin, volume_phy, resolution,
                angular_weight=None, chunk=1 << 18):
    """Cone-beam back-projection onto the reconstruction grid."""
    views, height, width = filtered.shape
    device, dtype = filtered.device, filtered.dtype
    if angular_weight is None:
        angular_weight = torch.ones(views, device=device, dtype=dtype)

    source, detector, frame = pose_to_frame(vecs)
    u_hat = frame[..., :, 0]
    v_hat = frame[..., :, 1]
    n_hat = frame[..., :, 2]
    du, dv = pixel_spacing(vecs)
    sid = ((detector - source) * n_hat).sum(dim=-1)

    points = _voxel_grid(volume_origin, volume_phy, resolution, device, dtype)
    volume = torch.zeros(points.shape[0], device=device, dtype=dtype)
    delta_beta = 2.0 * math.pi / views

    for view in range(views):
        weight_map = filtered[view].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        centre = detector[view]
        contribution = torch.zeros_like(volume)
        for start in range(0, points.shape[0], chunk):
            block = points[start:start + chunk]
            direction = block - source[view]
            depth = (direction * n_hat[view]).sum(dim=-1)
            valid = depth > 1e-3
            hit = source[view] + direction * (sid[view] / depth.clamp_min(1e-3)).unsqueeze(-1)
            offset = hit - centre
            u_mm = (offset * u_hat[view]).sum(dim=-1)
            v_mm = (offset * v_hat[view]).sum(dim=-1)
            col = (width - 1) * 0.5 + u_mm / du[view]
            row = (height - 1) * 0.5 + v_mm / dv[view]
            sample_grid = torch.stack((
                2.0 * col / max(width - 1, 1) - 1.0,
                2.0 * row / max(height - 1, 1) - 1.0,
            ), dim=-1).view(1, 1, -1, 2)
            sampled = F.grid_sample(
                weight_map, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).view(-1)
            fdk_weight = (sid[view] / depth.clamp_min(1e-3)) ** 2
            contribution[start:start + chunk] = torch.where(valid, sampled * fdk_weight, sampled * 0)
        volume += contribution * angular_weight[view]

    volume = volume * (0.5 * delta_beta)
    return volume.view(*resolution)


def reconstruct(proj, vecs, volume_origin, volume_phy, resolution,
                view_weights=None, chunk=1 << 18, normalize_weights=True):
    """Weighted FDK: cosine weight -> ramp filter -> weighted back-projection.

    ``normalize_weights`` rescales the per-view weights so that their mean is 1
    and ``sum(d beta * m)`` stays ``2 pi``.  Without it the grey level of the
    reconstruction scales with the number of views, so a 20-view and a 360-view
    reconstruction could not be compared directly.
    """
    weighted = cone_weight(proj, vecs)
    filtered = ramp_filter(weighted, vecs)
    if view_weights is not None and normalize_weights:
        view_weights = view_weights * (view_weights.numel() / view_weights.sum())
    return backproject(
        filtered, vecs, volume_origin, volume_phy, resolution,
        angular_weight=view_weights, chunk=chunk,
    )


def fit_scale(reconstruction, target, mask=None):
    """Least-squares grey-level calibration (FDK discretisation constant)."""
    if mask is None:
        numerator = (reconstruction * target).sum()
        denominator = (reconstruction * reconstruction).sum().clamp_min(1e-12)
    else:
        numerator = (reconstruction * target * mask).sum()
        denominator = (reconstruction * reconstruction * mask).sum().clamp_min(1e-12)
    return (numerator / denominator).clamp_min(1e-8)


def psnr(prediction, target, data_range=None):
    if data_range is None:
        data_range = float(target.max() - target.min())
    mse = torch.mean((prediction - target) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(torch.tensor(data_range * data_range) / mse))
