import torch
import torch.nn as nn


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
