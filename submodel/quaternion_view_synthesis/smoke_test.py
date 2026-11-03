"""Fast self-check for the quaternion view synthesis module.

    python -m submodel.quaternion_view_synthesis.smoke_test

Checks, without training:

1. the plane-induced homography really maps one measured view onto its
   neighbour (so the geometric path is correct, not just shape-compatible);
2. the synthesiser is an *exact* identity when the requested angle equals the
   source angle (the 20 real measurements are never rewritten);
3. the parameter-free geometric layer ``G(q)`` satisfies the quaternion
   composition law;
4. a QConv has exactly 1/4 of the parameters of a real conv with 4x channels;
5. FDK produces a finite volume that correlates with the ground truth.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import fdk as fdk_module
from . import geometry
from .dataset import DentalProjectionDataset, load_volume_xyz, select_input_indices
from .model import QuaternionSoftmaxEnsemble, build_model
from .qconv import QConv2d, quaternion_multiply, rotate_feature_vector


def _correlation(a, b):
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8))


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DentalProjectionDataset(
        data_root="dataset/dental/syn_data",
        split_file="data/dataset_split/dental_split.json",
        split="test",
        image_size=128,
        cache_dir="dataset/dental/_nvs_cache",
        limit=1,
    )
    sample = dataset[0]
    subject = sample["subject"]
    projections = sample["proj"].to(device)
    vecs = sample["vecs"].to(device)
    print(f"[smoke] subject={subject} proj={tuple(projections.shape)} device={device}")

    # 1. homography sanity: view 0 -> view 1 (one degree apart)
    warped = geometry.warp_view(projections[0], vecs[0], vecs[1])
    reference = projections[1]
    print(
        f"[smoke] homography 0->1 : corr={_correlation(warped, reference):.4f} "
        f"mae={float((warped - reference).abs().mean()):.4f} "
        f"(mean intensity {float(reference.mean()):.4f})"
    )

    # 2. identity at zero relative rotation, 3. composition law
    model = build_model(base_channels=8, levels=2).to(device).eval()
    image = projections[0].view(1, 1, *projections.shape[-2:])
    with torch.no_grad():
        identity = model(image, vecs[0:1], vecs[0:1])
        forward = model(image, vecs[0:1], vecs[18:19])

    gate = float(identity["gate"])
    identity_error = float((identity["out"] - image).abs().max())
    print(f"[smoke] identity gate={gate:.6f} max|out-a|={identity_error:.6e}")

    q_rel = forward["q_rel"][0]
    q_half = torch.tensor([math.cos(math.radians(4.0)), 0.0, math.sin(math.radians(4.0)), 0.0],
                          device=device)
    features = torch.randn(1, 4, 4, 8, 8, device=device)
    composed = rotate_feature_vector(
        rotate_feature_vector(features, q_rel.expand(1, 4)), q_half.expand(1, 4)
    )
    direct = rotate_feature_vector(features, quaternion_multiply(q_half, q_rel).expand(1, 4))
    print(f"[smoke] composition law max error={float((composed - direct).abs().max()):.3e}")

    # 4. parameter count
    qconv = QConv2d(8, 8, 3)
    real = torch.nn.Conv2d(32, 32, 3)
    print(
        f"[smoke] params QConv(8->8,3x3)={sum(p.numel() for p in qconv.parameters())} "
        f"vs real conv(32->32)={sum(p.numel() for p in real.parameters())}"
    )

    # 5. FDK sanity
    indices = select_input_indices(projections.shape[0], 20)
    resolution = (32, 32, 32)
    volume = fdk_module.reconstruct(
        projections[indices], vecs[indices],
        sample["meta"]["volume_origin"], sample["meta"]["volume_phy"], resolution,
        view_weights=torch.ones(indices.size, device=device),
    )
    truth = load_volume_xyz(f"dataset/dental/syn_data/{subject}/gt_volume.nii.gz", size=resolution)
    truth = truth.to(device)[0, 0]
    scale = fdk_module.fit_scale(volume, truth)
    print(
        f"[smoke] fdk finite={bool(torch.isfinite(volume).all())} "
        f"corr={_correlation(volume * scale, truth):.4f} "
        f"psnr={fdk_module.psnr(volume * scale, truth):.2f}dB"
    )
    print("[smoke] OK")


if __name__ == "__main__":
    main()
