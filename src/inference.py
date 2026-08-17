"""Inference for the projection interpolation model (Fig. 3).

Loads a training checkpoint (model weights + z-score stats + config), loads a
case's projections, and predicts the projection at a requested target angle.

Usage:
    conda activate deeplearning
    python src/inference.py --checkpoint checkpoints/.../interpolation_epoch10.pt \
                            --case 2026-06-04_065713 --target_angle 30.5
"""

from __future__ import annotations

import argparse
import math
import os
import pickle

import numpy as np
import torch

from fig3 import InterpolationModel
from dataloader_projection import _wrap_pi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_data_root(cfg: dict, data_name: str) -> str:
    """Pick the first existing data root for ``data_name`` (same logic as training)."""
    candidates = []
    roots = cfg.get("data_roots", {})
    if isinstance(roots, dict) and data_name in roots:
        val = roots[data_name]
        candidates = [val] if isinstance(val, str) else list(val)
    if not candidates and cfg.get("data_root"):
        candidates = [cfg["data_root"]]

    for c in candidates:
        c = os.path.expanduser(str(c))
        if os.path.isdir(c):
            return c
    return os.path.expanduser(str(candidates[0])) if candidates else ""


def load_case(root: str, case: str):
    """Return ``(projs, angles)`` of one case: projs (K,H,W) float32, angles (K,)."""
    path = os.path.join(root, "processed", "projections", f"{case}.pickle")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["projs"].astype(np.float32), data["angles"].astype(np.float32)


def build_model_from_ckpt(ckpt: dict, device: str):
    """Rebuild the interpolation model from a checkpoint dict."""
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    num_inputs = int(ckpt.get("final_view", model_cfg.get("in_channels", 2)))
    use_cos_sin = bool(cfg.get("training", {}).get("use_cos_sin", True))
    angular_dim = 2 * num_inputs if use_cos_sin else num_inputs

    model = InterpolationModel(
        in_channels=num_inputs,
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        num_down=int(model_cfg.get("num_down", 4)),
        num_regions=int(model_cfg.get("num_regions", 4)),
        angular_dim=angular_dim,
        zscore_mean=float(ckpt["zscore_mean"]),
        zscore_std=float(ckpt["zscore_std"]),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), num_inputs, use_cos_sin


def predict(model, projs: np.ndarray, angles: np.ndarray, target_rad: float,
            num_inputs: int, device: str, use_cos_sin: bool = True) -> np.ndarray:
    """Predict the projection at ``target_rad`` from its nearest views."""
    d = np.abs(_wrap_pi(angles - target_rad))
    src_idx = np.argsort(d)[:num_inputs]
    src_idx = np.sort(src_idx)

    src = projs[src_idx]
    src_angles = angles[src_idx]
    angular = _wrap_pi(target_rad - src_angles).astype(np.float32)
    if use_cos_sin:
        angular = np.concatenate([np.cos(angular) - 1.0, np.sin(angular)])

    src_list = [
        torch.from_numpy(np.ascontiguousarray(src[i])).unsqueeze(0).unsqueeze(0).to(device)
        for i in range(num_inputs)
    ]
    ang = torch.from_numpy(angular).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(src_list, ang)  # (1, 1, H, W)
    return pred.squeeze(0).squeeze(0).cpu().numpy()  # (H, W), raw range


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the interpolation model.")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--case", required=True, help="case id, e.g. 2026-06-04_065713")
    parser.add_argument("--target_angle", type=float, required=True,
                        help="target projection angle in degrees")
    parser.add_argument("--num_inputs", type=int, default=None,
                        help="override number of source views (default: checkpoint final_view)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "inference_results"))
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, num_inputs, use_cos_sin = build_model_from_ckpt(ckpt, device)
    if args.num_inputs is not None:
        num_inputs = args.num_inputs

    cfg = ckpt["config"]
    data_name = cfg.get("data_name", "thorax")
    root = resolve_data_root(cfg, data_name)
    if not root:
        raise FileNotFoundError("no existing data root found in checkpoint config")

    projs, angles = load_case(root, args.case)
    target_rad = math.radians(args.target_angle)

    pred = predict(model, projs, angles, target_rad, num_inputs, device, use_cos_sin)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"{args.case}_angle{args.target_angle:.1f}")
    np.save(base + ".npy", pred)
    try:
        from PIL import Image
        Image.fromarray(np.clip(pred, 0, 255).astype(np.uint8)).save(base + ".png")
    except Exception:
        pass

    print(f"case={args.case} | target_angle={args.target_angle} deg | "
          f"pred shape={tuple(pred.shape)} | range=[{pred.min():.1f}, {pred.max():.1f}]")
    print(f"saved: {base}.npy (+ .png)")


if __name__ == "__main__":
    main()
