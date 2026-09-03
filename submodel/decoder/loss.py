import torch
import torch.nn as nn
import torch.nn.functional as F


def mu_to_hu(volume: torch.Tensor) -> torch.Tensor:
    """Convert the project's linear attenuation coefficient to HU."""
    return (volume / 0.022 - 1.0) * 1000.0


def masked_l1(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error in a GT-defined mask.

    ``mask`` must be made from the target only.  The prediction is deliberately
    not clamped, so an out-of-window/out-of-bone prediction still receives a
    gradient pointing back to its target value.
    """
    mask = mask.to(dtype=error.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def bone_gt_mask_l1(
    prediction_mu: torch.Tensor,
    target_mu: torch.Tensor,
    bone_lower_hu: float,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    """Normalized L1 restricted to voxels whose *GT* is bone."""
    target_hu = mu_to_hu(target_mu.float())
    bone_mask = target_hu >= bone_lower_hu
    error = torch.abs(prediction_mu.float() - target_mu.float())
    return masked_l1(error / (clamp_max - clamp_min), bone_mask)


def soft_tissue_gt_mask_l1(
    prediction_mu: torch.Tensor,
    target_mu: torch.Tensor,
    soft_window_low: float,
    soft_window_high: float,
) -> torch.Tensor:
    """Normalized L1 restricted to voxels whose *GT* lies in a HU window."""
    prediction_hu = mu_to_hu(prediction_mu.float())
    target_hu = mu_to_hu(target_mu.float())
    soft_mask = (target_hu >= soft_window_low) & (target_hu < soft_window_high)
    error = torch.abs(prediction_hu - target_hu)
    return masked_l1(error / (soft_window_high - soft_window_low), soft_mask)


def ssim_loss_3d(
    prediction_mu: torch.Tensor,
    target_mu: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
    window_size: int = 3,
) -> torch.Tensor:
    """Differentiable local 3-D SSIM loss on the fixed attenuation range.

    This is intentionally separate from ``util.get_ssim_3d``.  The latter is
    a non-differentiable NumPy/skimage reporting metric; this implementation is
    a PyTorch loss and therefore can update the reconstruction network.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    data_range = clamp_max - clamp_min
    prediction = (prediction_mu.float() - clamp_min) / data_range
    target = (target_mu.float() - clamp_min) / data_range
    if prediction.ndim == 3:
        prediction = prediction[None, None]
        target = target[None, None]
    elif prediction.ndim == 4:
        prediction = prediction[:, None]
        target = target[:, None]
    elif prediction.ndim != 5:
        raise ValueError(
            "ssim_loss_3d expects [D,H,W], [B,D,H,W], or [B,C,D,H,W] tensors"
        )
    padding = window_size // 2
    mu_prediction = F.avg_pool3d(prediction, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool3d(target, window_size, stride=1, padding=padding)
    variance_prediction = (
        F.avg_pool3d(prediction.square(), window_size, stride=1, padding=padding)
        - mu_prediction.square()
    ).clamp_min(0.0)
    variance_target = (
        F.avg_pool3d(target.square(), window_size, stride=1, padding=padding)
        - mu_target.square()
    ).clamp_min(0.0)
    covariance = (
        F.avg_pool3d(prediction * target, window_size, stride=1, padding=padding)
        - mu_prediction * mu_target
    )

    # Standard SSIM constants for data_range=1 after fixed-range scaling.
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (
        (2.0 * mu_prediction * mu_target + c1)
        * (2.0 * covariance + c2)
        / (
            (mu_prediction.square() + mu_target.square() + c1)
            * (variance_prediction + variance_target + c2)
        )
    )
    return 1.0 - ssim_map.mean()


def gradient1_loss_3d(
    volume_gt: torch.Tensor,
    volume_predict: torch.Tensor,
    loss_func: nn.Module,
) -> torch.Tensor:
    """Batch-aware equivalent of ``models.loss.gradient1_loss``.

    Inputs use [B, C, X, Y, Z]. With a mean-reduction loss, the result equals
    the mean of the original per-volume gradient loss over the batch.
    """
    if volume_gt.ndim != 5 or volume_predict.ndim != 5:
        raise ValueError(
            "gradient1_loss_3d expects [B, C, X, Y, Z] tensors, "
            f"got {tuple(volume_gt.shape)} and {tuple(volume_predict.shape)}."
        )
    if volume_gt.shape != volume_predict.shape:
        raise ValueError(
            "Ground truth and prediction shapes differ: "
            f"{tuple(volume_gt.shape)} vs {tuple(volume_predict.shape)}."
        )

    gdx_real = volume_gt[:, :, 1:, :, :] - volume_gt[:, :, :-1, :, :]
    gdy_real = volume_gt[:, :, :, 1:, :] - volume_gt[:, :, :, :-1, :]
    gdz_real = volume_gt[:, :, :, :, 1:] - volume_gt[:, :, :, :, :-1]

    gdx_fake = volume_predict[:, :, 1:, :, :] - volume_predict[:, :, :-1, :, :]
    gdy_fake = volume_predict[:, :, :, 1:, :] - volume_predict[:, :, :, :-1, :]
    gdz_fake = volume_predict[:, :, :, :, 1:] - volume_predict[:, :, :, :, :-1]

    return (
        loss_func(gdx_real, gdx_fake)
        + loss_func(gdy_real, gdy_fake)
        + loss_func(gdz_real, gdz_fake)
    )
