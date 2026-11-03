"""Projection-domain metrics (PSNR / SSIM) used for TensorBoard monitoring.

Everything here is differentiable-free (evaluated under ``torch.no_grad``) and
batched, so a split can be scored on the GPU without a second library.

The dynamic range defaults to the per-image range of the *ground truth*, which
is what the DICOM/NIfTI world usually reports for CT-like data.  Pass an
explicit ``data_range`` to freeze it across epochs and make the curves
comparable.
"""

import torch
import torch.nn.functional as F

_EPS = 1e-12


def _gaussian_window(size, sigma, device, dtype):
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) * 0.5
    profile = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    profile = profile / profile.sum()
    return profile.outer(profile).view(1, 1, size, size)


def _channel_window(channels, size, sigma, device, dtype):
    window = _gaussian_window(size, sigma, device, dtype)
    return window.expand(channels, 1, size, size).contiguous()


def psnr(prediction, target, data_range=None):
    """Mean PSNR over the batch.

    ``prediction`` / ``target`` are ``[B, C, H, W]``.
    """
    if data_range is None:
        flat = target.flatten(1)
        data_range = flat.amax(dim=1) - flat.amin(dim=1)
    mse = (prediction - target).pow(2).flatten(1).mean(dim=1).clamp_min(_EPS)
    return float((10.0 * torch.log10(data_range.pow(2) / mse)).mean())


def ssim(prediction, target, data_range=None, window_size=11, sigma=1.5):
    """Mean SSIM over the batch (Gaussian window, one channel or many)."""
    if data_range is None:
        flat = target.flatten(1)
        data_range = flat.amax(dim=1) - flat.amin(dim=1)
    data_range = data_range.view(-1, 1, 1, 1)

    channels = prediction.shape[1]
    window = _channel_window(channels, window_size, sigma, prediction.device, prediction.dtype)
    padding = window_size // 2

    mu_pred = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)

    var_pred = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_pred.pow(2)
    var_target = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target.pow(2)
    covariance = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_pred * mu_target

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_pred * mu_target + c1) * (2.0 * covariance + c2)
    denominator = (mu_pred.pow(2) + mu_target.pow(2) + c1) * (var_pred + var_target + c2)
    return float((numerator / denominator).flatten(1).mean(dim=1).mean())


def l1(prediction, target):
    return float(F.l1_loss(prediction, target))


METRIC_NAMES = ("psnr", "ssim", "l1")
