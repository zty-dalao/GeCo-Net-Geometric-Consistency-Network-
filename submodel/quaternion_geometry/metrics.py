"""Reconstruction metrics used by quaternion-aware training.

The root trainer historically computed these values inline.  Keeping the
protocol here makes quaternion experiments explicit and guarantees that the
train/validation/test streams use the same implementation.
"""

import math

import torch


def quaternion_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
) -> dict[str, float]:
    """Return the root-compatible clamped 3-D PSNR and SSIM.

    ``prediction`` and ``target`` are clamped before evaluation.  Each volume
    is then min-max normalized independently, matching the established
    ``get_psnr``/``get_ssim_3d`` protocol used by the original trainer.
    The function is evaluation-only and never builds an autograd graph.
    """
    if clamp_max <= clamp_min:
        raise ValueError("clamp_max must be greater than clamp_min")
    # Import lazily: util.util_func also exposes NIfTI helpers and therefore
    # imports SimpleITK.  Quaternion model construction should remain usable
    # for lightweight smoke tests that do not evaluate volumes.
    from util.util_func import data_norm, get_psnr, get_ssim_3d
    with torch.no_grad():
        prediction = torch.clamp(prediction.detach().float(), clamp_min, clamp_max)
        target = torch.clamp(target.detach().float(), clamp_min, clamp_max)
        prediction = data_norm(prediction)
        target = data_norm(target)
        psnr = get_psnr(prediction, target)
        ssim = get_ssim_3d(prediction, target, data_range=1)
    # get_psnr may return the string ``'INF'`` for an exact match.  Numeric
    # infinity keeps TensorBoard and JSON logging consistent with other paths.
    if isinstance(psnr, str):
        psnr = math.inf
    return {
        "psnr_3d_clamp": float(psnr),
        "ssim_3d_clamp": float(ssim),
    }


__all__ = ["quaternion_reconstruction_metrics"]
