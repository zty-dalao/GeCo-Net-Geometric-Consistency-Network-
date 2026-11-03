"""Test / evaluate the quaternion view synthesiser on the dental split.

    # metrics only
    python -m submodel.quaternion_view_synthesis.evaluate --ckpt <path>

    # metrics + visualisation
    python -m submodel.quaternion_view_synthesis.evaluate --ckpt <path> --visualize

For every test subject the script builds three cone-beam reconstructions:

    FDK-20          the 20 measured views only
    FDK-fused       the 20 measured views + the synthesised views (weighted)
    FDK-dense       the same number of *real* views, i.e. the information ceiling

and reports PSNR / RMSE against the ground-truth volume plus PNG figures.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import fdk as fdk_module
from ..quaternion_geometry.pose import (
    pose_vec_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
)
from .dataset import (
    DentalProjectionDataset,
    load_volume_xyz,
    select_input_indices,
    select_target_indices,
)
from .model import QuaternionSoftmaxEnsemble, build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate quaternion view synthesis + FDK")
    parser.add_argument("--ckpt", default="submodel/quaternion_view_synthesis/checkpoints/qvs_dental/ckpt_latest.pt")
    parser.add_argument("--data_root", default="dataset/dental/syn_data")
    parser.add_argument("--split_file", default="data/dataset_split/dental_split.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache_dir", default="dataset/dental/_nvs_cache")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n_views", type=int, default=20)
    parser.add_argument("--n_synth", type=int, default=40)
    parser.add_argument("--synth_weight", type=float, default=0.25)
    parser.add_argument("--fdk_res", type=int, default=64)
    parser.add_argument("--out_dir", default="submodel/quaternion_view_synthesis/eval")
    parser.add_argument("--run_name", default="qvs_dental")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


@torch.no_grad()
def synthesize(model, ensemble, projections, vecs, source_indices, target_indices):
    """Return fused predictions and the per-view softmax weights for one target."""
    source_views = projections[source_indices].unsqueeze(1)
    vec_s = vecs[source_indices]
    outputs, weights = [], []
    for target in target_indices:
        vec_t = vecs[target].expand_as(vec_s)
        output = model(source_views, vec_s, vec_t)
        fused, weight = ensemble(
            output["out"].unsqueeze(1),
            output["q_rel"].unsqueeze(1),
            output["confidence"].unsqueeze(1),
            return_weights=True,
        )
        outputs.append(fused.squeeze(1).squeeze(0))
        weights.append(weight.squeeze(1))
    return torch.stack(outputs, dim=0), torch.stack(weights, dim=0)


def _slices(volume):
    """Middle axial / coronal / sagittal slices of a [X, Y, Z] volume."""
    return (
        volume[:, :, volume.shape[2] // 2],
        volume[:, volume.shape[1] // 2, :],
        volume[volume.shape[0] // 2, :, :],
    )


def _max_source_angle(vecs, source_indices, target_index):
    """Largest |Delta theta| between the measured views and one target view."""
    q_s = pose_vec_to_quaternion(vecs[source_indices])
    q_t = pose_vec_to_quaternion(vecs[target_index]).expand_as(q_s)
    q_rel = quaternion_multiply(quaternion_inverse(q_s), q_t)
    return 2.0 * torch.acos(q_rel[:, 0].abs().clamp(max=1.0)).max() * 180.0 / math.pi


def _save_projection_figure(path, real, synth, truth, real_indices, synth_indices):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = len(synth_indices)
    figure, axes = plt.subplots(3, columns, figsize=(3.0 * columns, 9.0))
    rows = (
        ("measured", real, real_indices),
        ("synthesised", synth, synth_indices),
        ("ground truth", truth, synth_indices),
    )
    for row, (name, stack, indices) in enumerate(rows):
        for column in range(columns):
            axis = axes[row, column]
            image = stack[column].detach().cpu().numpy()
            axis.imshow(image, cmap="gray")
            axis.set_title(f"{name} {indices[column]}")
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _save_volume_figure(path, volumes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(volumes.keys())
    figure, axes = plt.subplots(3, len(names), figsize=(3.2 * len(names), 9.0))
    for column, name in enumerate(names):
        for row, image in enumerate(_slices(volumes[name])):
            axis = axes[row, column]
            axis.imshow(image.detach().cpu().numpy(), cmap="gray")
            axis.set_title(name if row == 0 else "")
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})

    model = build_model(
        train_args.get("base_channels", 16), train_args.get("levels", 2)
    ).to(device)
    ensemble = QuaternionSoftmaxEnsemble(train_args.get("angle_scale", 4.0)).to(device)
    model.load_state_dict(checkpoint["model"])
    ensemble.load_state_dict(checkpoint["ensemble"])
    model.eval()
    ensemble.eval()

    dataset = DentalProjectionDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        split=args.split,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
        limit=args.limit,
    )

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for subject_index in range(len(dataset)):
        sample = dataset[subject_index]
        subject = sample["subject"]
        projections = sample["proj"].to(device)
        vecs = sample["vecs"].to(device)
        meta = sample["meta"]
        num_frames = projections.shape[0]

        source_indices = select_input_indices(num_frames, args.n_views)
        synth_stride = max(1, num_frames // args.n_synth)
        synth_indices = select_target_indices(
            num_frames, synth_stride, exclude=source_indices
        )[: args.n_synth]

        synth_proj, synth_weights = synthesize(
            model, ensemble, projections, vecs, source_indices, synth_indices
        )

        resolution = (args.fdk_res, args.fdk_res, args.fdk_res)
        origin, phy = meta["volume_origin"], meta["volume_phy"]
        real_proj = projections[source_indices]
        real_vecs = vecs[source_indices]

        rec_20 = fdk_module.reconstruct(
            real_proj, real_vecs, origin, phy, resolution,
            view_weights=torch.ones(real_proj.shape[0], device=device),
        )
        rec_fused = fdk_module.reconstruct(
            torch.cat((real_proj, synth_proj), dim=0),
            torch.cat((real_vecs, vecs[synth_indices]), dim=0),
            origin, phy, resolution,
            view_weights=torch.cat((
                torch.ones(real_proj.shape[0], device=device),
                torch.full((synth_proj.shape[0],), args.synth_weight, device=device),
            )),
        )
        dense_indices = select_target_indices(num_frames, synth_stride)[: args.n_synth + args.n_views]
        rec_dense = fdk_module.reconstruct(
            projections[dense_indices], vecs[dense_indices], origin, phy, resolution,
            view_weights=torch.ones(dense_indices.size, device=device),
        )

        truth = load_volume_xyz(Path(args.data_root) / subject / "gt_volume.nii.gz",
                                size=resolution).to(device)[0, 0]
        data_range = float(truth.max() - truth.min())

        subject_metrics = {}
        for name, reconstruction in (
            ("fdk_20", rec_20), ("fdk_fused", rec_fused), ("fdk_dense", rec_dense),
        ):
            scale = fdk_module.fit_scale(reconstruction, truth)
            calibrated = reconstruction * scale
            subject_metrics[name] = {
                "psnr": fdk_module.psnr(calibrated, truth, data_range),
                "rmse": float((calibrated - truth).pow(2).mean().sqrt()),
                "scale": float(scale),
            }
        subject_metrics["max_source_angle_deg"] = float(
            _max_source_angle(vecs, source_indices, synth_indices[0])
        )
        results[subject] = subject_metrics

        print(f"[qvs-eval] {subject}: " + " | ".join(
            f"{name} psnr={value['psnr']:.2f}" for name, value in subject_metrics.items()
            if isinstance(value, dict)
        ))

        if args.visualize:
            columns = min(4, synth_indices.size)
            _save_projection_figure(
                out_dir / f"{subject}_projections.png",
                projections[source_indices[:columns]],
                synth_proj[:columns],
                projections[synth_indices[:columns]],
                source_indices[:columns],
                synth_indices[:columns],
            )
            _save_volume_figure(
                out_dir / f"{subject}_volumes.png",
                {
                    "GT": truth,
                    "FDK-20": rec_20 * fdk_module.fit_scale(rec_20, truth),
                    "FDK-fused": rec_fused * fdk_module.fit_scale(rec_fused, truth),
                    "FDK-dense": rec_dense * fdk_module.fit_scale(rec_dense, truth),
                },
            )

    summary = {}
    for name in ("fdk_20", "fdk_fused", "fdk_dense"):
        summary[name] = {
            key: float(np.mean([results[subject][name][key] for subject in results]))
            for key in ("psnr", "rmse")
        }
    results["summary"] = summary
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("[qvs-eval] summary")
    for name, value in summary.items():
        print(f"  {name}: psnr={value['psnr']:.2f} rmse={value['rmse']:.5f}")
    print(f"[qvs-eval] metrics -> {out_dir / 'metrics.json'}")
    return results


if __name__ == "__main__":
    main()
