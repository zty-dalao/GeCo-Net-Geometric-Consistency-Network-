"""Training script for the interpolation model (Fig. 3).

Implements the training objective and settings from:

    J. Zhang and L. Ren, "Enhance four-dimensional cone-beam computed
    tomography (4D-CBCT) from sparse view acquisitions using a novel deep
    learning model", Biomedical Signal Processing and Control 119 (2026)
    109935.

Loss (Eq. 1):
    Loss = 1 - SSIM(x, y) + alpha * L_VGG(x, y)
    L_VGG(x, y) = MSE(rho(x), rho(y)),  rho = first 9 layers of pre-trained
    VGG16 (perceptual loss, Ledig et al.).  alpha = 1e-2.

Training settings (Table 2):
    optimizer = Adam, lr = 1e-3, batch size = 1, epochs = 300,
    lr decay x0.1 when training loss does not decrease for 10 epochs.

Usage:
    conda activate deeplearning
    python train.py
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fig3 import InterpolationModel


# ---------------------------------------------------------------------------
# 1. Angular encoding (Fig. 3: cos(beta-alpha)-1, sin(beta-alpha))
# ---------------------------------------------------------------------------
def encode_angular(raw_angular: torch.Tensor, use_cos_sin: bool = True) -> torch.Tensor:
    """Encode the angular distances.

    Parameters
    ----------
    raw_angular:
        ``(B, N)`` tensor holding the ``N`` angular distances ``beta - alpha_i``.
        NOTE: ``cos`` / ``sin`` expect **radians**; pass degrees * (pi / 180)
        if your angles are in degrees.
    use_cos_sin:
        If True (paper setting), each distance ``d`` becomes ``[cos(d)-1,
        sin(d)]``, so the output is ``(B, 2N)``.  If False, returns the raw
        ``(B, N)`` distances unchanged.

    Returns
    -------
    ``(B, 2N)`` or ``(B, N)`` tensor.
    """
    if not use_cos_sin:
        return raw_angular
    return torch.cat([torch.cos(raw_angular) - 1.0, torch.sin(raw_angular)], dim=-1)


# ---------------------------------------------------------------------------
# 2. SSIM (differentiable, with Gaussian window; C1/C2 follow the paper)
# ---------------------------------------------------------------------------
def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [math.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2)) for x in range(window_size)],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, val_range: float = 1.0) -> torch.Tensor:
    """Mean structural similarity between ``x`` and ``y`` (0~1, higher=better)."""
    channel = x.size(1)
    window = _create_window(window_size, channel).to(x.device).to(x.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(x, window, padding=pad, groups=channel)
    mu2 = F.conv2d(y, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(x * x, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=pad, groups=channel) - mu1_mu2

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# 3. VGG perceptual loss
# ---------------------------------------------------------------------------
def load_vgg(feature_layers: int = 9) -> Optional[nn.Module]:
    """Return the first ``feature_layers`` layers of pre-trained VGG16 (frozen)."""
    try:
        from torchvision import models

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:feature_layers]
        for p in vgg.parameters():
            p.requires_grad = False
        vgg.eval()
        return vgg
    except Exception as exc:  # torchvision / weights unavailable
        print(f"[warn] VGG unavailable ({exc}); VGG loss disabled.")
        return None


# ---------------------------------------------------------------------------
# 4. Interpolation loss (Eq. 1)
# ---------------------------------------------------------------------------
class InterpolationLoss(nn.Module):
    def __init__(self, alpha: float = 1e-2, vgg: Optional[nn.Module] = None, val_range: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.vgg = vgg
        self.val_range = val_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_ssim = 1.0 - ssim(pred, target, val_range=self.val_range)
        l_vgg = torch.zeros((), device=pred.device, dtype=pred.dtype)
        if self.vgg is not None:
            # VGG16 expects 3-channel input; tile the single-channel projection.
            l_vgg = F.mse_loss(
                self.vgg(pred.repeat(1, 3, 1, 1)),
                self.vgg(target.repeat(1, 3, 1, 1)),
            )
        return l_ssim + self.alpha * l_vgg


# ---------------------------------------------------------------------------
# 5. Dataset (dummy - replace with your real projection data)
# ---------------------------------------------------------------------------
class ProjectionDataset(Dataset):
    """Dummy dataset to demonstrate the data layout.

    Real data layout per sample:
        projections: (num_inputs, 1, H, W)  -- num_inputs source projections
        angular:     (num_inputs,)          -- (beta - alpha_i), one per source
        target:      (1, H, W)              -- ground-truth interpolated projection
    """

    def __init__(self, length: int = 32, num_inputs: int = 6, h: int = 512, w: int = 384):
        self.length = length
        self.num_inputs = num_inputs
        self.h, self.w = h, w

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        projs = torch.randn(self.num_inputs, 1, self.h, self.w)
        angular = torch.rand(self.num_inputs)  # radians in real usage
        target = torch.randn(1, self.h, self.w)
        return projs, angular, target


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------
def main() -> None:
    # -------------------- configuration (paper Table 2) --------------------
    cfg = dict(
        num_inputs=6,           # number of source projections (2 in the paper)
        use_cos_sin=True,       # paper's cos/sin angular encoding
        alpha=1e-2,             # weight of the VGG perceptual loss
        lr=1e-3,
        batch_size=1,
        epochs=300,
        lr_patience=10,         # epochs without training-loss decrease before x0.1
        base_channels=64,
        num_down=4,
        num_regions=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # -------------------- model --------------------
    # When cos/sin encoding is used, each distance -> 2 numbers.
    angular_dim = (2 * cfg["num_inputs"]) if cfg["use_cos_sin"] else cfg["num_inputs"]
    model = InterpolationModel(
        in_channels=cfg["num_inputs"],
        base_channels=cfg["base_channels"],
        num_down=cfg["num_down"],
        num_regions=cfg["num_regions"],
        angular_dim=angular_dim,
    ).to(cfg["device"])

    vgg = load_vgg(feature_layers=9)
    if vgg is not None:
        vgg = vgg.to(cfg["device"])

    criterion = InterpolationLoss(alpha=cfg["alpha"], vgg=vgg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=cfg["lr_patience"]
    )

    loader = DataLoader(
        ProjectionDataset(length=32, num_inputs=cfg["num_inputs"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    # -------------------- train --------------------
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for projs, angular, target in loader:
            projs = projs.squeeze(0).to(cfg["device"])      # (N, 1, H, W) -> list of (1, H, W)
            angular = angular.squeeze(0).to(cfg["device"])  # (N,)
            target = target.squeeze(0).to(cfg["device"])    # (1, H, W)

            # N projections -> batch of N tensors, each (1, 1, H, W)
            proj_list = [projs[i : i + 1] for i in range(projs.size(0))]

            # Angular encoding: (N,) -> (1, angular_dim)
            angular = encode_angular(angular.unsqueeze(0), cfg["use_cos_sin"])
            target = target.unsqueeze(0)  # (1, 1, H, W)

            pred = model(proj_list, angular)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)  # reduce lr when validation loss plateaus

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d}/{cfg['epochs']} | "
                f"loss {avg_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
            )


# ---------------------------------------------------------------------------
# 7. View-selection / rotation-consistency loss (self-supervised)
# ---------------------------------------------------------------------------
def consistency_loss(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    use_ssim: bool = True,
    val_range: float = 1.0,
) -> torch.Tensor:
    """Consistency between two interpolations of the SAME target angle x.

    For a static (non-moving) object the projection at angle x is unique, so
    two interpolations produced from different source-view sets must agree.
    This enforces rotation / view-selection invariance.

    IMPORTANT: use this together with the supervised loss (Eq. 1). On its own
    the model could trivially satisfy it by collapsing to a constant image.
    """
    if use_ssim:
        return 1.0 - ssim(pred_a, pred_b, val_range=val_range)
    return F.mse_loss(pred_a, pred_b)


def sample_rotor_views(
    angles: torch.Tensor,
    projections: torch.Tensor,
    target_angle: torch.Tensor,
    num_inputs: int,
    rotor_offset: float,
):
    """Select ``num_inputs`` equidistant views starting at ``rotor_offset``.

    The ``num_inputs`` views are equally spaced over 360 deg::

        rotor_offset + k * (2*pi / num_inputs),  k = 0 .. num_inputs-1

    Each rotor angle is snapped to the nearest actually-acquired view in
    ``angles``.  Returns ``(projs, angular_dists)`` where::

        projs:         (num_inputs, 1, H, W)
        angular_dists: (num_inputs,) = target_angle - angle_of_selected_view
    """
    spacing = 2 * math.pi / num_inputs
    rotor_angles = rotor_offset + torch.arange(num_inputs, dtype=angles.dtype) * spacing
    idx = [int(torch.argmin(torch.abs(angles - ra)).item()) for ra in rotor_angles]
    idx = torch.tensor(idx, dtype=torch.long)
    return projections[idx], target_angle - angles[idx]


class RotationConsistencyDataset(Dataset):
    """Full-view dataset that yields two view rotors for one target angle.

    Each sample returns:
        proj_a, ang_a : ``num_inputs`` equidistant views (rotor at offset phi)
        proj_b, ang_b : the same rotor rotated clockwise by a random delta
        target        : ground-truth projection at the target angle x

    Both view sets target the SAME angle x; a correct model must output the
    same projection for both (rotation consistency).
    """

    def __init__(self, projections, angles, num_inputs=6, max_delta=0.5):
        self.projections = projections
        self.angles = angles
        self.num_inputs = num_inputs
        self.max_delta = max_delta

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        x = self.angles[idx]
        phi = torch.rand(1).item() * 2 * math.pi
        delta = (torch.rand(1).item() * 2 - 1) * self.max_delta
        proj_a, ang_a = sample_rotor_views(self.angles, self.projections, x, self.num_inputs, phi)
        proj_b, ang_b = sample_rotor_views(
            self.angles, self.projections, x, self.num_inputs, phi + delta
        )
        target = self.projections[idx].clone()
        return proj_a, ang_a, proj_b, ang_b, target


def _to_model_inputs(projs, angular, use_cos_sin, device):
    """(num_inputs,1,H,W),(num_inputs,) -> (list of (1,1,H,W), (1, angular_dim))."""
    proj_list = [projs[i : i + 1].to(device) for i in range(projs.size(0))]
    angular = encode_angular(angular.unsqueeze(0).to(device), use_cos_sin)
    return proj_list, angular


def main_consistency() -> None:
    # -------------------- configuration --------------------
    cfg = dict(
        num_views=491,           # full-view count (dummy demo)
        num_inputs=6,            # equidistant views per rotor
        use_cos_sin=True,
        alpha=1e-2,              # supervised VGG weight (Eq. 1)
        lambda_consistency=1.0,  # consistency-loss weight
        lr=1e-3,
        batch_size=1,
        epochs=5,                # demo; use ~300 for real training
        base_channels=64,
        num_down=4,
        num_regions=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Dummy full-view data over 360 deg (radians). Replace with real data.
    m, h, w = cfg["num_views"], 64, 48
    angles = torch.linspace(0, 2 * math.pi, m)
    full = torch.randn(m, 1, h, w)

    angular_dim = (2 * cfg["num_inputs"]) if cfg["use_cos_sin"] else cfg["num_inputs"]
    model = InterpolationModel(
        in_channels=cfg["num_inputs"], base_channels=cfg["base_channels"],
        num_down=cfg["num_down"], num_regions=cfg["num_regions"],
        angular_dim=angular_dim,
    ).to(cfg["device"])

    vgg = load_vgg(9)
    if vgg is not None:
        vgg = vgg.to(cfg["device"])
    criterion = InterpolationLoss(alpha=cfg["alpha"], vgg=vgg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    loader = DataLoader(
        RotationConsistencyDataset(full, angles, cfg["num_inputs"], max_delta=0.5),
        batch_size=cfg["batch_size"], shuffle=True,
    )

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ep = 0.0
        for proj_a, ang_a, proj_b, ang_b, target in loader:
            proj_a, ang_a = proj_a.squeeze(0), ang_a.squeeze(0)
            proj_b, ang_b = proj_b.squeeze(0), ang_b.squeeze(0)
            target = target.squeeze(0).to(cfg["device"]).unsqueeze(0)

            in_a, d_a = _to_model_inputs(proj_a, ang_a, cfg["use_cos_sin"], cfg["device"])
            in_b, d_b = _to_model_inputs(proj_b, ang_b, cfg["use_cos_sin"], cfg["device"])

            pred_a = model(in_a, d_a)
            pred_b = model(in_b, d_b)

            # supervised (both branches) + rotation consistency
            loss = (
                criterion(pred_a, target)
                + criterion(pred_b, target)
                + cfg["lambda_consistency"] * consistency_loss(pred_a, pred_b)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep += loss.item()

        print(f"epoch {epoch:3d}/{cfg['epochs']} | total loss {ep / len(loader):.4f}")

    # sanity check: consistency on one fresh pair
    model.eval()
    with torch.no_grad():
        x = angles[m // 2]
        pa, da = sample_rotor_views(angles, full, x, cfg["num_inputs"], 0.0)
        pb, db = sample_rotor_views(angles, full, x, cfg["num_inputs"], 0.5)
        A = model(*_to_model_inputs(pa, da, cfg["use_cos_sin"], cfg["device"]))
        B = model(*_to_model_inputs(pb, db, cfg["use_cos_sin"], cfg["device"]))
        print("consistency loss (A vs B):", float(consistency_loss(A, B)))


if __name__ == "__main__":
    main()
    # main_consistency()  # rotation-consistency training demo
